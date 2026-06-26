# Auditable Internal-Controls Agent

**Track: Agents for Business**  
Full-population AP ledger controls testing — every transaction, every run, every decision logged.

---

## The Problem

Traditional AP (accounts payable) audits sample 5–15% of transactions. Whatever falls outside the sample goes untested, and findings depend on which sample was drawn, not on actual risk. A duplicate payment buried in the un-sampled 90% is invisible until the vendor calls.

This agent tests **100% of the ledger** on every run and records every decision in an append-only ledger, making each finding traceable back to the exact data and logic that produced it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CONTROLS AGENT PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AP Ledger CSV                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐   risk-prioritised    ┌──────────────────┐   │
│  │ PLANNER  │──── check list ──────▶│  POLICY GATE     │   │
│  │ (Gemini  │                       │  • scope guard   │   │
│  │  flash)  │                       │  • budget guard  │   │
│  └──────────┘                       │  • approval gate │   │
│                                     └────────┬─────────┘   │
│                                              │ scoped read  │
│                                              ▼             │
│                                     ┌──────────────────┐   │
│                                     │  CHECK EXECUTORS │   │
│                                     │  AP-BEN-01       │   │
│                                     │  AP-DUP-01       │   │
│                                     │  AP-SOD-01       │   │
│                                     │  AP-OUT-01       │   │
│                                     └────────┬─────────┘   │
│                                  raw findings│             │
│                                              ▼             │
│                                     ┌──────────────────┐   │
│                                     │   CLASSIFIER     │   │
│                                     │  (Gemini flash)  │   │
│                                     │  narrative prose │   │
│                                     └────────┬─────────┘   │
│                                              │             │
│                                              ▼             │
│                                     ┌──────────────────┐   │
│                                     │ DECISION LEDGER  │   │
│                                     │ (append-only DB) │   │
│                                     └────────┬─────────┘   │
│                                              │             │
│                                              ▼             │
│                                     ┌──────────────────┐   │
│                                     │  REPORT VIEWER   │   │
│                                     │   (Streamlit)    │   │
│                                     └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Model routing:**
| Step | Model | Tokens |
|------|-------|--------|
| Risk triage (planner) | gemini-2.5-flash | ~600 |
| Benford / Duplicate / SoD / Outlier checks | Deterministic Python | **0** |
| Narrative classifier | gemini-2.5-flash | ~1,800 |

Checks use zero LLM tokens — arithmetic must be reproducible and defensible.

---

## Controls Implemented

| ID | Control | Method | Statistic |
|----|---------|--------|-----------|
| AP-BEN-01 | Benford's Law conformity | Chi-square + MAD dual gate | p-value, MAD |
| AP-DUP-01 | Exact duplicate invoices | Vendor-scoped exact match | Dollar exposure |
| AP-SOD-01 | Segregation of duties | Self-approval / over-authority / threshold clustering | Count, cluster size |
| AP-OUT-01 | Statistical outliers | Log-space z-scores + round-dollar concentration | z-score |

---

## Course Concepts Demonstrated

| Concept | Where | Detail |
|---------|-------|--------|
| **Agent system** | `agent/loop.py` | Full orchestration loop: planner → policy → tool → checks → classify → ledger |
| **Security features** | `agent/policy.py` | `PolicyGate`: scope guard (data access control), budget guard (cost ceiling), approval gate (HIGH/CRITICAL sign-off) |
| **Deployability** | `Dockerfile`, `deploy.sh` | Cloud Run-ready; `bash deploy.sh` deploys in one command |
| **Agent skills / CLI** | `run_agent.py` | CLI agent entry point with `--budget`, `--data`, `--db` flags |
| **MCP-style tool layer** | `agent/loop.py` | Scoped data reads logged as `TOOL_CALL` events; agent cannot read outside allowed paths |

---

## Evaluation Results

Tested against 9 planted anomalies in a 2,760-transaction synthetic dataset:

| Control | Expected | Detected | Recall | Precision |
|---------|----------|----------|--------|-----------|
| AP-BEN-01 | 1 | 1 | 100% | 50% |
| AP-DUP-01 | 3 | 3 | 100% | 100% |
| AP-SOD-01 | 3 | 3 | 100% | 100% |
| AP-OUT-01 | 2 | 2 | 100% | 50% |
| **TOTAL** | **9** | **9** | **100%** | **75%** |

3 false positives are secondary cross-check detections of already-flagged entities (same anomaly seen by a second control). Effective precision: 100%.

---

## Quick Start

### Prerequisites
- Python 3.12+
- Gemini API key from [Google Cloud Console](https://console.cloud.google.com/apis/credentials) with the [Gemini API enabled](https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com)

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd ControlsAgent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY=AIzaSy...

# 4. Generate the synthetic dataset
python -m synthetic.generate --output-dir data

# 5. Run the agent (CLI)
python run_agent.py

# 6. Launch the report viewer
streamlit run app.py
# Open http://localhost:8501
```

### Run the eval harness
```bash
python run_eval.py
```
Outputs precision/recall per control against planted anomalies.

### Deploy to Cloud Run
```bash
export PROJECT_ID=your-gcp-project-id
bash deploy.sh
```

---

## Project Structure

```
ControlsAgent/
├── agent/
│   ├── models.py        # Event + Finding dataclasses
│   ├── ledger.py        # Append-only SQLite ledger
│   ├── checks.py        # Four deterministic AP check executors
│   ├── classifier.py    # Gemini narrative classifier
│   ├── planner.py       # Gemini risk-triage planner
│   ├── policy.py        # PolicyGate (scope / budget / approval)
│   └── loop.py          # Main agent orchestration loop
├── synthetic/
│   └── generate.py      # Synthetic dataset generator (seed=42)
├── data/
│   ├── transactions.csv # 2,760 AP transactions (generated)
│   ├── ground_truth.json# Planted anomaly ground truth
│   └── config.json      # Role limits + approval thresholds
├── app.py               # Streamlit report viewer
├── run_agent.py         # CLI entry point
├── run_eval.py          # Precision/recall eval harness
├── Dockerfile           # Cloud Run container
├── deploy.sh            # One-command Cloud Run deploy
├── WRITEUP.md           # Technical writeup
└── DEMO_SCRIPT.md       # Video demo script
```

---

## Decision Ledger — Audit Trail

Every run produces a chain of 32 events in SQLite. Each finding links back to its source event, enabling full traceability:

```
[POLICY_GATE]  scope=PASS  data/transactions.csv
  [TOOL_CALL]  loaded 2,760 rows
    [PLAN]     gemini-2.5-flash  tokens=634  cost=$0.00019
      [CHECK]  AP-SOD-01  deterministic  0 tokens
        [CLASSIFY]  gemini-2.5-flash  tokens=1,847  cost=$0.00461
          FINDING  user:U-07  CRITICAL  self_approval  n=5
```

---

## Cost

| Item | Cost per run |
|------|-------------|
| Planner (gemini-2.5-flash) | ~$0.002 |
| Classifier — 4 batches (gemini-2.5-flash) | ~$0.012 |
| Checks (deterministic) | $0.000 |
| **Total** | **~$0.014** |

---

## Security Notes

- `.env` is in `.gitignore` — API keys are never committed
- `PolicyGate.scope_guard()` enforces that the agent can only read pre-approved file paths
- `PolicyGate.budget_guard()` halts the run if LLM spend exceeds the configured ceiling
- Severity levels are set by deterministic code — the LLM cannot escalate or downgrade a finding
