# Auditable Internal-Controls Agent

**Track: Agents for Business**  
*An AI agent that runs 100% of an AP ledger through four controls tests, records every decision in an append-only ledger, and produces workpaper-quality findings that a human auditor can independently verify.*

---

## Problem

Enterprise AP audits are sample-based by necessity: auditors inspect 5–15% of transactions. Whatever falls outside the sample goes untested, and every finding is dependent on which 15% was drawn, not on where the actual risk sits.

This creates two failure modes:

1. **Coverage gap** — A duplicate payment buried in the un-sampled 85% is invisible until the vendor calls.
2. **Defensibility gap** — When a finding is challenged, an auditor can describe what they checked but not *how* they decided. There is no decision trail.

The problem is not that auditors lack skill. It's that no tool makes full-population continuous monitoring operationally practical, and no tool records its own reasoning in a form that survives audit review.

---

## Solution

A single-command AI agent that:

- Reads a raw AP ledger (no labels, no annotations)
- Runs four controls tests across every transaction
- Records every decision — from policy gate to LLM call to check statistic — in an append-only SQLite ledger
- Produces findings with both a machine-verifiable statistic and a workpaper-quality narrative

The agent tests **100% of the ledger** for under $0.015 per run.

---

## Architecture

The pipeline has five stages, each producing events in the decision ledger:

```
AP Ledger CSV
    │
    ▼
PLANNER (gemini-2.5-flash)
  → reads dataset summary
  → returns risk-prioritised check list
  → PLAN event logged (tokens, cost, model, rationale)
    │
    ▼
POLICY GATE
  → scope_guard: agent may only read pre-approved file paths
  → budget_guard: halts run if LLM spend exceeds ceiling
  → approval_gate: HIGH/CRITICAL findings routed for human sign-off
  → POLICY_GATE events logged per rule
    │
    ▼
TOOL CALL
  → scoped data read (MCP-style: path validated against scope)
  → TOOL_CALL event logged (row count, column list, path)
    │
    ▼
CHECK EXECUTORS (deterministic Python — zero LLM tokens)
  → AP-BEN-01: Benford's Law (chi-square + MAD dual gate)
  → AP-DUP-01: Exact duplicates (vendor-scoped)
  → AP-SOD-01: Segregation of duties (3 sub-checks)
  → AP-OUT-01: Statistical outliers (log-space z-scores + round-dollar)
  → CHECK events logged per control
    │
    ▼
CLASSIFIER (gemini-2.5-flash)
  → writes severity_rationale and recommended_action prose
  → does NOT change severity (set deterministically by check)
  → CLASSIFY events logged (tokens, cost, model)
    │
    ▼
DECISION LEDGER (append-only SQLite)
  → every finding links to its source CHECK event
  → every CHECK event links to its PLAN event
  → full chain: POLICY_GATE → TOOL_CALL → PLAN → CHECK → CLASSIFY → FINDING
```

**Key design choice — zero LLM tokens for checks:**  
The arithmetic (chi-square, z-score, duplicate match) is always deterministic Python. An LLM writing "this vendor's p-value is 0.003" can hallucinate a number. An LLM that writes a narrative about a p-value already computed by scipy cannot. This separation is what makes the workpaper defensible.

---

## Controls Design

### AP-BEN-01 — Benford's Law Conformity
Benford's Law predicts the first-digit distribution of naturally occurring numbers. Fabricated invoice amounts typically concentrate on "convenient" digits (5, 6, 9). The check uses a **chi-square test** (H₀: distribution matches Benford) AND a **mean absolute deviation gate** — chi-square alone fires on small samples; the MAD gate eliminates those.

Thresholds: chi-square p < 0.005, MAD > 0.025, minimum 20 invoices per vendor.

### AP-DUP-01 — Exact Duplicate Detection
Vendor-scoped match on `(vendor_id, invoice_number, amount)`. Cross-vendor matching was rejected: birthday paradox produces ~35 false positives on 60 vendors. Dollar exposure is computed per duplicate pair.

### AP-SOD-01 — Segregation of Duties
Three sub-checks run in parallel:
- **Self-approval**: `created_by == approved_by`
- **Over-authority**: `amount > role_limit[approver_role]`
- **Threshold clustering**: Kernel density estimation on per-user transactions finds clusters of invoices within a configurable band below the $10,000 approval limit (a known fraud pattern)

### AP-OUT-01 — Statistical Outliers
Log-space z-scores per vendor (amounts are log-normally distributed — raw z-scores produce massive false positives on log-uniform data). Separate round-dollar concentration check flags vendors where ≥4 transactions are exact round amounts above $1,000.

---

## Agent Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Agent system** | `agent/loop.py` — full orchestration loop driven by Gemini planner output |
| **Security features** | `agent/policy.py` — `PolicyGate` with scope guard, budget guard, approval gate; each produces an auditable event |
| **Deployability** | `Dockerfile` + `deploy.sh` — one command deploys to Google Cloud Run |
| **Agent CLI / skills** | `run_agent.py` — full CLI with flags (`--budget`, `--data`, `--db`) |
| **MCP-style scoped tools** | Data reads validated against approved paths, logged as `TOOL_CALL` events; agent cannot read outside its scope |

---

## Decision Ledger

The ledger is the differentiator. Every run writes 32 events. Every finding has a `source_event_id` that chains back to the `CHECK` event, which chains back to the `PLAN` event, which chains back to the `TOOL_CALL` event and `POLICY_GATE` events. The complete chain for one finding looks like:

```
POLICY_GATE  scope=PASS       path=data/transactions.csv
POLICY_GATE  budget=PASS      ceiling=$1.00  spent=$0.000
TOOL_CALL    rows=2760        cols=[txn_id, vendor_id, ...]
PLAN         model=gemini-2.5-flash  tokens=634  cost=$0.00019
  CHECK      AP-SOD-01        deterministic  0 tokens
    CLASSIFY model=gemini-2.5-flash  tokens=1,847  cost=$0.00461
      FINDING  user:U-07  CRITICAL  self_approval  n=5  $13,071
```

A reviewer can open the database, query by `finding_id`, and reconstruct every step without trusting any black box.

---

## Evaluation

The synthetic dataset (2,760 transactions, seed=42) was generated with 9 planted anomalies across all four controls. Anomaly labels were stripped from the CSV — the agent has no ground truth.

| Control | Planted | Detected | Recall | Notes |
|---------|---------|----------|--------|-------|
| AP-BEN-01 | 1 | 1 | 100% | V-0010, 200 biased transactions |
| AP-DUP-01 | 3 | 3 | 100% | Pairs: $14.2k, $8.75k, $31k |
| AP-SOD-01 | 3 | 3 | 100% | U-07 self-approval, U-03 over-authority, clustering |
| AP-OUT-01 | 2 | 2 | 100% | $45k outlier, round-dollar concentration |
| **Total** | **9** | **9** | **100%** | |

**Overall: Recall = 100%, Precision = 75%, F1 = 0.857**

3 false positives are secondary cross-check detections — entities already flagged by one control that also trigger a second. In a real workpaper these would be marked "related to finding #X".

Run cost: **$0.0137** (planner $0.002 + classifier $0.012 + checks $0.000).

---

## Limitations and What Comes Next

- **Synthetic data only.** Real AP ledgers require field mapping and a data dictionary.
- **SQLite is single-tenant.** A production deployment would use Cloud Spanner or AlloyDB with row-level audit access controls.
- **4 controls.** A full COSO AP control suite would add PO matching, three-way match, vendor master change detection, and payment timing analysis.
- **Planner is stateless.** Adding memory across runs would let the agent track whether a vendor's Benford deviation is worsening over time.

---

## Build Journey

The project was built over 7 days following the course arc:

- **Days 1–2:** Data model, check executors, eval harness — getting the arithmetic right before touching any LLM
- **Day 3:** Policy gate + planner integration — the first full agent loop
- **Day 4:** Classifier + live API debugging (OAuth tokens vs API keys; quota routing to gemini-2.5-flash)
- **Day 5:** Streamlit report viewer with interactive audit trail drill-down
- **Day 6–7:** Writeup, demo script, reproducibility verification

The most important design lesson: **separate what the LLM does from what the code does.** Once those boundaries were clear, the architecture became straightforward.
