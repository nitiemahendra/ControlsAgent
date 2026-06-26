"""
Day-3 entry point: full agent loop (planner + policy gate + checks + ledger).

Usage:
    python run_agent.py
    python run_agent.py --budget 0.50 --data data/transactions.csv

Set GEMINI_API_KEY in .env or environment to enable the Gemini planner.
Without a key the planner falls back to a default full-population plan.
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from agent.ledger import Ledger
from agent.loop import run_agent

_SEV_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",     default="controls_agent.db")
    ap.add_argument("--data",   default="data/transactions.csv")
    ap.add_argument("--config", default="data/config.json")
    ap.add_argument("--budget", type=float, default=1.00, help="LLM cost ceiling in USD")
    args = ap.parse_args()

    has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())

    W = 66
    SEP = "=" * W
    DIV = "-" * W

    print(f"\n{SEP}")
    print("  CONTROLS AGENT  --  DAY 3 FULL LOOP")
    print(SEP)
    print(f"  Dataset : {args.data}")
    print(f"  Budget  : ${args.budget:.2f}")
    print(f"  Gemini  : {'API key set -- using Gemini planner' if has_key else 'no key -- default plan'}")
    print()

    result = run_agent(
        dataset_path=args.data,
        config_path=args.config,
        db_path=args.db,
        budget_usd=args.budget,
    )

    # ── Planner output ────────────────────────────────────────────────────────
    if result.risk_narrative:
        print("  Planner risk narrative:")
        # Wrap at ~60 chars
        words = result.risk_narrative.split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 64:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  Status         : {result.status}")
    print(f"  Total findings : {result.finding_count}")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = result.by_severity.get(sev, 0)
        if n:
            print(f"    {sev:<10}: {n}")
    planner_note = "(default plan)" if result.plan_fallback else f"${result.total_cost_usd:.4f}"
    print(f"  LLM cost       : {planner_note}")
    print(f"  Run ID         : {result.run_id[:8]}...")
    print(f"  Decision ledger: {args.db}")

    # ── Event-type breakdown from ledger ──────────────────────────────────────
    ledger = Ledger(args.db)
    events = ledger.get_events(result.run_id)
    event_types: dict[str, int] = {}
    for ev in events:
        event_types[ev["step_type"]] = event_types.get(ev["step_type"], 0) + 1

    print()
    print(f"  {DIV}")
    print(f"  Decision ledger events ({len(events)} total):")
    for etype in ["PLAN", "POLICY_GATE", "TOOL_CALL", "CHECK", "APPROVAL_REQUEST", "REPORT"]:
        n = event_types.get(etype, 0)
        if n:
            print(f"    {etype:<18}: {n}")

    # ── Trail demo: first CRITICAL/HIGH finding ───────────────────────────────
    findings = ledger.get_findings(result.run_id)
    top = next((f for f in findings if f["severity"] in ("CRITICAL", "HIGH")), None)
    if top:
        print()
        print(f"  {DIV}")
        trail = ledger.get_finding_trail(top["finding_id"])
        print(
            f"  TRAIL DEMO  {top['control_id']}  {top['entity_ref']}  "
            f"sev={top['severity']}"
        )
        for ev in trail["event_trail"]:
            parent = f"  <- {ev['parent_event_id'][:8]}" if ev.get("parent_event_id") else ""
            model_tag = f"  [{ev['model']}]" if ev["model"] != "deterministic" else ""
            pol = f"  policy={ev['policy_status']}" if ev["policy_status"] != "NA" else ""
            print(
                f"    [{ev['step_type']:<18}] {ev['event_id'][:8]}"
                f"{parent}{model_tag}{pol}"
            )
        print(f"    [FINDING           ] {top['finding_id'][:8]}")
        print(f"  {DIV}")

    print()


if __name__ == "__main__":
    main()
