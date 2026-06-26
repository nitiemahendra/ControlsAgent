"""
Main agent loop: planner → policy gate → MCP data tool → check executors → ledger.

Event chain for a single check (illustrates the traceability the ledger captures):

  POLICY_GATE (scope)                  ← run root
    TOOL_CALL (data load)              ← parent: scope event
      PLAN  (Gemini flash)             ← parent: tool_call event
        POLICY_GATE (budget)           ← parent: plan event
          CHECK (AP-BEN-01, etc.)      ← parent: plan event
            CLASSIFY (gemini-2.5-pro)  ← parent: check event  (narrative enrichment)
            APPROVAL_REQUEST           ← parent: check event  (HIGH/CRITICAL only)
            FINDING (FK source_event_id → CHECK event)
      REPORT                           ← parent: plan event
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent.checks import run_benford, run_duplicates, run_outliers, run_sod
from agent.classifier import enrich_findings
from agent.ledger import Ledger
from agent.models import Event, Finding
from agent.planner import RunPlan, create_plan
from agent.policy import PolicyGate, PolicyViolation

load_dotenv()

_CHECK_FNS = {
    "AP-BEN-01": run_benford,
    "AP-DUP-01": run_duplicates,
    "AP-SOD-01": run_sod,
    "AP-OUT-01": run_outliers,
}

_DEFAULT_BUDGET = float(os.environ.get("COST_CEILING_USD", "1.00"))


@dataclass
class RunResult:
    run_id: str
    status: str               # COMPLETE | PARTIAL | FAILED
    finding_count: int
    by_severity: dict[str, int]
    total_cost_usd: float
    plan_reasoning: str
    risk_narrative: str
    plan_fallback: bool = False


def run_agent(
    dataset_path: str = "data/transactions.csv",
    config_path: str = "data/config.json",
    db_path: str = "controls_agent.db",
    budget_usd: float | None = None,
    allowed_paths: list[str] | None = None,
) -> RunResult:
    """Full agent loop: scope-guard → load → plan → checks → findings → report."""
    if budget_usd is None:
        budget_usd = _DEFAULT_BUDGET
    if allowed_paths is None:
        allowed_paths = [dataset_path]

    config = json.loads(Path(config_path).read_text())
    ledger = Ledger(db_path)
    run_id = str(uuid.uuid4())

    gate = PolicyGate(
        ledger=ledger,
        run_id=run_id,
        allowed_paths=allowed_paths,
        budget_usd=budget_usd,
    )

    all_findings: list[Finding] = []
    plan: RunPlan | None = None
    status = "COMPLETE"

    try:
        # ── 1. Scope guard (Rule 1) ───────────────────────────────────────────
        scope_event_id = gate.scope_guard(dataset_path, parent_event_id=None)

        # ── 2. MCP data tool: load dataset ───────────────────────────────────
        tool_event = ledger.append_event(Event(
            run_id=run_id,
            step_type="TOOL_CALL",
            actor="EXECUTOR",
            model="deterministic",
            input_summary={"tool": "read_dataset", "path": dataset_path},
            output_summary={},
            reasoning="MCP data tool: read AP transaction ledger.",
            parent_event_id=scope_event_id,
        ))

        df = pd.read_csv(dataset_path)
        df["amount"] = df["amount"].astype(float)

        ledger.append_event(Event(
            run_id=run_id,
            step_type="TOOL_CALL",
            actor="EXECUTOR",
            model="deterministic",
            input_summary={},
            output_summary={"rows": len(df), "vendors": int(df["vendor_id"].nunique())},
            reasoning=f"Loaded {len(df):,} transactions from {dataset_path}.",
            parent_event_id=tool_event.event_id,
        ))

        # ── 3. Budget guard before LLM call (Rule 3) ─────────────────────────
        gate.budget_guard(parent_event_id=tool_event.event_id)

        # ── 4. Planner (Gemini flash) ─────────────────────────────────────────
        plan = create_plan(df, run_id, ledger, parent_event_id=tool_event.event_id)

        # ── 5. Check executors ────────────────────────────────────────────────
        for check_plan in plan.checks:
            check_fn = _CHECK_FNS.get(check_plan.check_id)
            if check_fn is None:
                continue

            # Budget guard before each check (in case prior LLM calls were costly)
            gate.budget_guard(parent_event_id=plan.plan_event_id)

            check_event = ledger.append_event(Event(
                run_id=run_id,
                step_type="CHECK",
                actor="EXECUTOR",
                model="deterministic",
                input_summary={
                    "check": check_plan.check_id,
                    "rows": len(df),
                    "rationale": check_plan.rationale,
                    "priority": check_plan.priority,
                },
                output_summary={},
                reasoning=(
                    f"Deterministic {check_plan.check_id} executor. "
                    f"Priority {check_plan.priority}. Zero LLM tokens."
                ),
                parent_event_id=plan.plan_event_id,
            ))

            findings = check_fn(df, config, run_id, check_event.event_id)

            # Classifier: enrich findings with LLM narrative (no-op without API key)
            findings = enrich_findings(
                findings, check_plan.check_id, run_id, ledger,
                parent_event_id=check_event.event_id,
            )

            for f in findings:
                # Approval gate (Rule 2) for HIGH/CRITICAL before persisting
                if f.severity in ("HIGH", "CRITICAL"):
                    review_status, _ = gate.approval_gate(
                        f, parent_event_id=check_event.event_id
                    )
                    f.review_status = review_status
                ledger.append_finding(f)

            all_findings.extend(findings)

        # ── 6. Final budget check ─────────────────────────────────────────────
        gate.budget_guard(parent_event_id=plan.plan_event_id)

    except PolicyViolation as exc:
        status = "PARTIAL"
        ledger.append_event(Event(
            run_id=run_id,
            step_type="REPORT",
            actor="EXECUTOR",
            model="deterministic",
            input_summary={"policy_violation": str(exc)},
            output_summary={"status": "PARTIAL", "findings_so_far": len(all_findings)},
            reasoning=f"Run halted by policy: {exc}",
        ))

    except Exception as exc:
        status = "FAILED"
        ledger.append_event(Event(
            run_id=run_id,
            step_type="REPORT",
            actor="EXECUTOR",
            model="deterministic",
            input_summary={"error": str(exc)},
            output_summary={"status": "FAILED"},
            reasoning=f"Unexpected error: {type(exc).__name__}: {exc}",
        ))
        raise

    # ── Report ────────────────────────────────────────────────────────────────
    by_sev: dict[str, int] = {}
    for f in all_findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1

    total_cost = ledger.get_run_cost(run_id)

    ledger.append_event(Event(
        run_id=run_id,
        step_type="REPORT",
        actor="EXECUTOR",
        model="deterministic",
        input_summary={"total_findings": len(all_findings)},
        output_summary={
            "status": status,
            "by_severity": by_sev,
            "total_cost_usd": round(total_cost, 6),
        },
        reasoning=f"{status}: {len(all_findings)} finding(s), ${total_cost:.4f} LLM cost.",
        parent_event_id=plan.plan_event_id if plan else None,
    ))

    return RunResult(
        run_id=run_id,
        status=status,
        finding_count=len(all_findings),
        by_severity=by_sev,
        total_cost_usd=total_cost,
        plan_reasoning=plan.reasoning if plan else "",
        risk_narrative=plan.risk_narrative if plan else "",
        plan_fallback=plan.fallback if plan else True,
    )
