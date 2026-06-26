"""
Day-2 runner: exercises all four check executors against the synthetic ledger.

Usage:
    python run_checks.py
    python run_checks.py --db controls_agent.db --data data/transactions.csv
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from agent.checks import run_benford, run_duplicates, run_outliers, run_sod
from agent.ledger import Ledger
from agent.models import Event

_CHECKS = [
    ("AP-BEN-01", "Benford first-digit conformity      ", run_benford),
    ("AP-DUP-01", "Duplicate / near-duplicate payments ", run_duplicates),
    ("AP-SOD-01", "Segregation-of-duties               ", run_sod),
    ("AP-OUT-01", "Round-dollar & outlier              ", run_outliers),
]

_SEV_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _fmt_stat(stat: dict) -> str:
    if "mad" in stat:
        return f"MAD={stat['mad']:.4f}  chi2={stat.get('chi2', '?')}"
    if "dollars_at_risk" in stat:
        return f"${stat['dollars_at_risk']:,.0f} at risk  n={stat['cluster_size']}"
    if "txn_count" in stat and "total_amount" in stat:
        return f"n={stat['txn_count']}  ${stat['total_amount']:,.0f} total"
    if "txn_count" in stat and "max_excess" in stat:
        return f"n={stat['txn_count']}  excess=${stat['max_excess']:,.0f}"
    if "z" in stat:
        z = stat["z"]
        return f"z={z:.1f}" + (f"  iqr_mult={stat['iqr_multiple']}x" if stat.get("iqr_multiple") else "")
    if "round_count" in stat:
        return f"n={stat['round_count']}  ${stat.get('total_round_amount', 0):,.0f}"
    if "cluster_size" in stat and "total_amount" in stat:
        return f"n={stat['cluster_size']}  ${stat['total_amount']:,.0f} total"
    return str(stat)[:50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",     default="controls_agent.db")
    ap.add_argument("--data",   default="data/transactions.csv")
    ap.add_argument("--config", default="data/config.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["amount"] = df["amount"].astype(float)
    config = json.loads(Path(args.config).read_text())

    ledger = Ledger(args.db)
    run_id = str(uuid.uuid4())

    W = 62
    SEP = "=" * W
    DIV = "-" * W

    print(f"\n{SEP}")
    print("  CONTROLS AGENT  --  DAY 2 CHECK RUNNER")
    print(SEP)
    print(f"  Run    : {run_id[:8]}...")
    print(f"  Dataset: {args.data}  ({len(df):,} transactions)")
    print(f"  DB     : {args.db}")
    print()

    plan_event = ledger.append_event(Event(
        run_id=run_id,
        step_type="PLAN",
        actor="PLANNER",
        input_summary={"dataset": args.data, "checks": [c[0] for c in _CHECKS]},
        output_summary={"planned": len(_CHECKS)},
        model="deterministic",
        reasoning="Day-2 direct runner: execute all four checks sequentially.",
    ))

    all_findings = []

    for ctrl_id, desc, fn in _CHECKS:
        check_event = ledger.append_event(Event(
            run_id=run_id,
            step_type="CHECK",
            actor="EXECUTOR",
            input_summary={"check": ctrl_id, "rows": len(df)},
            output_summary={},
            model="deterministic",
            reasoning=f"Deterministic {ctrl_id} executor; zero LLM tokens.",
            parent_event_id=plan_event.event_id,
        ))

        findings = fn(df, config, run_id, check_event.event_id)

        for f in findings:
            ledger.append_finding(f)

        all_findings.extend(findings)
        print(f"  {ctrl_id}  {desc}  {len(findings):>3} finding(s)")

    by_sev: dict[str, int] = {}
    for f in all_findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1

    ledger.append_event(Event(
        run_id=run_id,
        step_type="REPORT",
        actor="EXECUTOR",
        input_summary={"total_findings": len(all_findings)},
        output_summary={"by_severity": by_sev},
        model="deterministic",
        parent_event_id=plan_event.event_id,
    ))

    print()
    print(DIV)
    print(f"  {'SEV':<10}  {'CONTROL':<12}  {'ENTITY':<28}  KEY STAT")
    print(DIV)

    sorted_findings = sorted(
        all_findings,
        key=lambda x: (_SEV_RANK.get(x.severity, 9), x.control_id),
    )
    for f in sorted_findings:
        entity = f.entity_ref[:26]
        stat = _fmt_stat(f.statistic)
        print(f"  {f.severity:<10}  {f.control_id:<12}  {entity:<28}  {stat}")

    print(DIV)
    print(f"\n  Total findings : {len(all_findings)}")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = by_sev.get(sev, 0)
        if n:
            print(f"    {sev:<10}: {n}")
    run_cost = ledger.get_run_cost(run_id)
    print(f"\n  Run cost       : ${run_cost:.4f}  (all deterministic)")
    print(f"  Decision ledger: {args.db}")

    # Demo: show full trail for the first HIGH/CRITICAL finding
    high = next((f for f in sorted_findings if f.severity in ("CRITICAL", "HIGH")), None)
    if high:
        print(f"\n{DIV}")
        print(f"  TRAIL DEMO -- finding {high.finding_id[:8]}  ({high.control_id}  {high.entity_ref})")
        trail = ledger.get_finding_trail(high.finding_id)
        for ev in trail["event_trail"]:
            parent = f"  <- {ev['parent_event_id'][:8]}" if ev.get("parent_event_id") else ""
            print(f"    [{ev['step_type']:<14}] {ev['event_id'][:8]}  {ev['ts']}{parent}")
        print(f"    [FINDING       ] {high.finding_id[:8]}  sev={high.severity}")
        print(DIV)

    print()


if __name__ == "__main__":
    main()
