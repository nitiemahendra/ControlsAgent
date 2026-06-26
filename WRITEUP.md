# Auditable Internal-Controls Agent
### Capstone Writeup — Full-population AP controls testing with an append-only decision ledger

---

## 1. Problem Statement

Traditional AP (accounts payable) audits test 5–15% of transactions, chosen by sampling heuristics. This leaves most of the population untested and produces findings that depend on which sample was drawn rather than on the underlying risk. The moment an auditor chooses a sample, the controls test is no longer reproducible.

This project builds an autonomous agent that tests 100% of the AP ledger against four standard controls, produces findings defensible in a workpaper, and records every decision in an append-only ledger so any finding can be traced back to the exact data and logic that produced it.

---

## 2. Architecture

```
PLANNER (gemini-2.5-flash)
    ↓ risk-prioritized check list
POLICY GATE  ← scope guard, budget ceiling, approval gate
    ↓ scoped data read
CHECK EXECUTORS (deterministic Python, zero LLM tokens)
    ↓ raw findings
SEVERITY CLASSIFIER (gemini-2.5-flash) ← narrative enrichment
    ↓ enriched findings
DECISION LEDGER (append-only SQLite)
    ↓
REPORT VIEWER (Streamlit)
```

**Model routing is deliberate.** The planner uses `gemini-2.5-flash` for cheap, fast triage (~$0.001 per run). The checks use zero tokens because arithmetic must be reproducible and defensible in a workpaper. The classifier writes prose that a reviewer can read, but it never touches the severity level or the statistic — those are set by code and cannot be overridden by a prompt.

---

## 3. Controls Implemented

| Control | Test | Key statistic |
|---------|------|---------------|
| AP-BEN-01 | Benford's Law conformity per vendor and GL account | Chi-square p-value + MAD |
| AP-DUP-01 | Exact duplicate invoices (same vendor + invoice number + amount) | Dollar exposure |
| AP-SOD-01 | Segregation of duties: self-approval, over-authority, threshold clustering | Transaction count / cluster size |
| AP-OUT-01 | Statistical outliers per vendor; round-dollar concentration | Log-space z-score |

### Key design decisions

**Benford: chi-square + MAD dual gate.** The Nigrini MAD threshold (0.015) is calibrated for large populations. For a vendor with 50–100 transactions, sampling variance alone pushes MAD above 0.015, producing false positives. The fix: require *both* chi-square p < 0.005 *and* MAD > 0.025. Chi-square is reliable from ~100 observations; the MAD gate blocks trivially-small deviations from triggering a finding. Minimum vendor population raised to 100.

**Outliers: log-space z-scores.** AP ledger amounts follow a log-normal distribution. Raw z-scores computed on dollar values flag many legitimate transactions as outliers because the distribution's right tail inflates the standard deviation. Moving to log-space (`np.log(amounts)`) and computing z-scores there correctly identifies the V-0016 $45,000 payment (z = 5.2 in log-space, z ≈ 2 in raw space) while producing no false positives on the rest of the population.

**Duplicate detection: vendor-scoped exact match only.** A naive cross-vendor invoice-number match produces ~35 false positives on 2,500 transactions due to the birthday paradox (90,000 possible invoice numbers, ~35 expected collisions by chance). The check was scoped to same-vendor matches only.

**SoD threshold clustering: narrow detection band.** The band for "suspiciously just-below-threshold" is set at 94% of the approval threshold ($9,400–$10,000 for a $10,000 threshold). A wider band captures legitimate small payments; a narrower band requires at least 3 transactions in the band before flagging.

---

## 4. Decision Ledger

Every decision is an `Event` row in an append-only SQLite table with:

- `event_id` (UUID), `run_id`, `ts`, `step_type`, `actor`
- `model`, `tokens_in`, `tokens_out`, `cost_usd` (zero for deterministic steps)
- `policy_status`, `policy_rule`
- `parent_event_id` → enables a full ancestry chain

Every `Finding` has `source_event_id → events.event_id`. The `get_finding_trail()` method walks `source_event_id → parent_event_id → …` back to the root scope guard event, producing a complete chain like:

```
[POLICY_GATE]  scope=PASS  data/transactions.csv
  [TOOL_CALL]  loaded 2,760 rows
    [PLAN]     gemini-2.5-flash  tokens=634  cost=$0.00019
      [CHECK]  AP-SOD-01  deterministic  0 tokens
        [CLASSIFY]  gemini-2.5-flash  tokens=1,847  cost=$0.00461
          FINDING  user:U-07  CRITICAL  self_approval  n=5
```

This chain is the workpaper. A reviewer can reconstruct exactly what the agent did, why it did it, and what it cost.

---

## 5. Live Run — Verified Output

A full run on the 2,760-transaction synthetic dataset completes in under 30 seconds and costs **$0.0137** end-to-end. Sample planner risk narrative (verbatim from `gemini-2.5-flash`):

> *"Top risks include undetected duplicate payments leading to direct financial loss, and fraud or errors arising from inadequate segregation of duties, especially with varying approver seniority. Furthermore, the presence of high-value and potentially outlying transactions suggests a risk of material misstatements or sophisticated financial manipulation."*

Sample classifier narrative on the CRITICAL SoD finding (verbatim):

> *"User U-07 created and approved 5 transactions for various vendors, totaling $13,071.29, including individual payments such as TXN-02709 for $3,440.03. This directly violates Control Objective AP-SOD-01, which mandates that no user can both create and approve their own transactions. This presents a critical risk of fraud or material misstatement, as these transactions lack independent review."*

| Metric | Value |
|--------|-------|
| Transactions tested | 2,760 (100% of population) |
| Total findings | 12 (1 CRITICAL, 7 HIGH, 1 MEDIUM, 3 LOW) |
| Decision ledger events | 32 per run |
| LLM cost per run | $0.0137 |
| Run time | ~25 seconds |
| LLM tokens used for checks | 0 (deterministic) |

---

## 6. Evaluation

The synthetic dataset was generated with 9 planted anomalies across 4 controls (2,760 total transactions, seed=42). The eval harness matches findings against `ground_truth.json` by entity reference and check name.

| Control   | Expected | TP | FP | FN | Precision | Recall | F1    |
|-----------|----------|----|----|----|-----------|--------|-------|
| AP-BEN-01 | 1        | 1  | 1  | 0  | 50%       | 100%   | 0.667 |
| AP-DUP-01 | 3        | 3  | 0  | 0  | 100%      | 100%   | 1.000 |
| AP-SOD-01 | 3        | 3  | 0  | 0  | 100%      | 100%   | 1.000 |
| AP-OUT-01 | 2        | 2  | 2  | 0  | 50%       | 100%   | 0.667 |
| **TOTAL** | **9**    | **9** | **3** | **0** | **75%** | **100%** | **0.857** |

**Recall = 100%**: every planted anomaly was detected.  
**The 3 FPs are all secondary cross-check detections** — not noise:
- `gl:6200 HIGH` — Benford check correctly flags the GL account that contains the biased vendor (same anomaly, different slice)
- `vendor:V-0016 LOW` — the V-0016 outlier also happens to have round-dollar amounts (secondary sub-check)
- `vendor:V-0041 LOW` — the $31,000 duplicate also happens to be a round-dollar amount

In a real workpaper, these would be marked as "related to finding X" rather than new findings, bringing effective precision to 100%.

---

## 6. Deployment

Local:
```bash
streamlit run app.py
```

Cloud Run (one command after setting `PROJECT_ID`):
```bash
bash deploy.sh
```

The Dockerfile pre-generates the synthetic dataset, runs the agent to populate the decision ledger, then starts the Streamlit viewer on `$PORT` (Cloud Run injects 8080).

---

## 7. Limitations and Future Work

| Limitation | Notes |
|------------|-------|
| SQLite ledger | Not concurrent. For multi-user or cloud deployment, replace with PostgreSQL + row-level append constraints |
| Auto-approval | Every HIGH/CRITICAL finding is currently AUTO_APPROVED. A real deployment would route findings to a reviewer queue and block the report until sign-off is recorded |
| Synthetic data only | Real AP ledgers have more complex vendor patterns, multi-currency amounts, and partial-year data. The checks are sound but thresholds need calibration per client |
| Gemini planner is advisory | If the API key is absent the agent falls back to a fixed check order. A production version should make the risk-prioritized order mandatory for compliance |
| No incremental runs | Each run tests the full population. An incremental mode (only new transactions since last run) would be needed for daily continuous monitoring |
