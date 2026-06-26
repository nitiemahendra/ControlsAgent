"""
Day-4 eval harness: precision/recall on planted anomalies.

Runs a fresh agent pass, then compares findings against ground_truth.json.

Usage:
    python run_eval.py
    python run_eval.py --data data/transactions.csv --gt data/ground_truth.json

Reframes testing the agent as testing the controls themselves:
  - Seed the dataset with known anomalies
  - Measure detection rate (recall) and false-positive rate (1 - precision)
  - 100% recall means every planted fraud indicator was caught
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from agent.ledger import Ledger
from agent.loop import run_agent


# ── Expected positives from ground truth ──────────────────────────────────────

@dataclass
class ExpectedPositive:
    control_id: str
    entity_key: str        # substring to match against finding.entity_ref
    description: str
    check_name: str = ""   # optional: match finding.check_name too


def build_expected(gt: dict) -> list[ExpectedPositive]:
    ep: list[ExpectedPositive] = []

    # AP-BEN-01: one planted vendor
    ep.append(ExpectedPositive(
        "AP-BEN-01", gt["AP-BEN-01"]["vendor"],
        f"Benford non-conformity ({gt['AP-BEN-01']['vendor']})",
    ))

    # AP-DUP-01: three duplicate pairs
    for pair in gt["AP-DUP-01"]:
        ep.append(ExpectedPositive(
            "AP-DUP-01", pair["vendor"],
            f"Dup pair {pair['invoice_number']} ${pair['amount']:,.2f} ({pair['vendor']})",
            check_name="duplicate_payments_exact",
        ))

    # AP-SOD-01: self-approval, over-authority, threshold clustering
    sod = gt["AP-SOD-01"]
    ep.append(ExpectedPositive("AP-SOD-01", "U-07",              "Self-approval (U-07)", "sod_self_approval"))
    ep.append(ExpectedPositive("AP-SOD-01", "U-03",              "Over-authority (U-03)", "sod_over_authority"))
    ep.append(ExpectedPositive("AP-SOD-01", sod["threshold_vendor"],
                               f"Threshold clustering ({sod['threshold_vendor']})", "sod_threshold_clustering"))

    # AP-OUT-01: vendor outlier + round-dollar
    out = gt["AP-OUT-01"]
    ep.append(ExpectedPositive(
        "AP-OUT-01", out["vendor_outlier"]["vendor"],
        f"Vendor outlier ${out['vendor_outlier']['amount']:,.0f} ({out['vendor_outlier']['vendor']})",
        check_name="outlier_vendor_amount",
    ))
    # Round-dollar: planted for V-0006 (the vendor used in _round_dollar())
    round_vendor = "V-0006"  # VENDORS[5] in generate.py
    ep.append(ExpectedPositive(
        "AP-OUT-01", round_vendor,
        f"Round-dollar txns ({round_vendor})",
        check_name="outlier_round_dollar",
    ))

    return ep


# ── Matching logic ────────────────────────────────────────────────────────────

def match_findings(
    findings: list[dict],
    expected: list[ExpectedPositive],
) -> tuple[list[tuple], list[dict], list[ExpectedPositive]]:
    """Returns (tp_pairs, fps, fns).

    tp_pairs: list of (ExpectedPositive, finding_dict)
    fps:      findings that don't match any expected positive
    fns:      expected positives with no matching finding
    """
    remaining_expected = list(expected)
    tps: list[tuple] = []
    fps: list[dict] = []

    for f in findings:
        matched_ep = None
        for ep in remaining_expected:
            if f["control_id"] != ep.control_id:
                continue
            if ep.entity_key not in f["entity_ref"]:
                continue
            if ep.check_name and f.get("check_name", "") != ep.check_name:
                continue
            matched_ep = ep
            break

        if matched_ep:
            tps.append((matched_ep, f))
            remaining_expected.remove(matched_ep)
        else:
            fps.append(f)

    fns = remaining_expected
    return tps, fps, fns


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",     default="controls_agent.db")
    ap.add_argument("--data",   default="data/transactions.csv")
    ap.add_argument("--config", default="data/config.json")
    ap.add_argument("--gt",     default="data/ground_truth.json")
    ap.add_argument("--budget", type=float, default=1.00)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    expected = build_expected(gt)

    W = 70
    SEP = "=" * W
    DIV = "-" * W

    print(f"\n{SEP}")
    print("  CONTROLS AGENT  --  DAY 4 EVAL HARNESS")
    print(SEP)
    print(f"  Dataset     : {args.data}")
    print(f"  Ground truth: {args.gt}")
    print(f"  Planted positives:")
    for ctrl in ["AP-BEN-01", "AP-DUP-01", "AP-SOD-01", "AP-OUT-01"]:
        n = sum(1 for e in expected if e.control_id == ctrl)
        print(f"    {ctrl}: {n}")
    print(f"    TOTAL   : {len(expected)}")
    print()

    # ── Run the agent ──────────────────────────────────────────────────────────
    print("  Running agent...")
    result = run_agent(
        dataset_path=args.data,
        config_path=args.config,
        db_path=args.db,
        budget_usd=args.budget,
    )
    print(f"  Run {result.run_id[:8]}...  status={result.status}  findings={result.finding_count}")
    print()

    # ── Load findings ──────────────────────────────────────────────────────────
    ledger = Ledger(args.db)
    findings = ledger.get_findings(result.run_id)

    # ── Global match ──────────────────────────────────────────────────────────
    tps, fps, fns = match_findings(findings, expected)

    # ── Per-control breakdown ─────────────────────────────────────────────────
    controls = ["AP-BEN-01", "AP-DUP-01", "AP-SOD-01", "AP-OUT-01"]
    rows = []
    for ctrl in controls:
        exp_ctrl  = [e for e in expected if e.control_id == ctrl]
        tp_ctrl   = [(e, f) for e, f in tps if e.control_id == ctrl]
        fp_ctrl   = [f for f in fps if f["control_id"] == ctrl]
        fn_ctrl   = [e for e in fns if e.control_id == ctrl]
        p, r, f1  = metrics(len(tp_ctrl), len(fp_ctrl), len(fn_ctrl))
        rows.append((ctrl, len(exp_ctrl), len(tp_ctrl), len(fp_ctrl), len(fn_ctrl), p, r, f1))

    # Totals
    total_exp = len(expected)
    total_tp  = len(tps)
    total_fp  = len(fps)
    total_fn  = len(fns)
    tot_p, tot_r, tot_f1 = metrics(total_tp, total_fp, total_fn)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"  {DIV}")
    hdr = f"  {'Control':<12} {'Exp':>4} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>7}  {'Recall':>7}  {'F1':>6}"
    print(hdr)
    print(f"  {DIV}")
    for ctrl, exp, tp, fp, fn, p, r, f1 in rows:
        print(f"  {ctrl:<12} {exp:>4} {tp:>4} {fp:>4} {fn:>4}  {p:>6.1%}  {r:>7.1%}  {f1:>6.3f}")
    print(f"  {DIV}")
    print(f"  {'TOTAL':<12} {total_exp:>4} {total_tp:>4} {total_fp:>4} {total_fn:>4}  {tot_p:>6.1%}  {tot_r:>7.1%}  {tot_f1:>6.3f}")
    print(f"  {DIV}")

    # ── TP details ────────────────────────────────────────────────────────────
    print(f"\n  True Positives ({total_tp}/{len(expected)} planted anomalies detected):")
    for ep, f in tps:
        sev = f["severity"]
        stat = _fmt_stat(f["statistic"])
        print(f"    [TP] {f['control_id']}  {ep.description:<45}  {sev}  {stat}")

    # ── FP details ────────────────────────────────────────────────────────────
    if fps:
        print(f"\n  False Positives ({total_fp} findings not in ground truth):")
        for f in fps:
            label = _fp_label(f)
            print(f"    [FP] {f['control_id']}  {f['entity_ref']:<30}  {f['severity']}  {label}")

    # ── FN details ────────────────────────────────────────────────────────────
    if fns:
        print(f"\n  False Negatives ({total_fn} planted anomalies NOT detected):")
        for ep in fns:
            print(f"    [FN] {ep.control_id}  {ep.description}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {DIV}")
    print(f"  Recall    {tot_r:.1%}  -- {'all planted anomalies detected' if tot_r == 1.0 else 'some missed'}")
    print(f"  Precision {tot_p:.1%}  -- {_fp_note(fps)}")
    print(f"  F1        {tot_f1:.3f}")
    print(f"  Run cost  ${result.total_cost_usd:.4f}")
    print(f"  {DIV}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_stat(stat: dict) -> str:
    if "mad" in stat:
        return f"MAD={stat['mad']:.4f}"
    if "dollars_at_risk" in stat:
        return f"${stat['dollars_at_risk']:,.0f} at risk"
    if "z" in stat:
        return f"z={stat['z']:.1f}"
    if "round_count" in stat:
        return f"n={stat['round_count']}"
    if "txn_count" in stat:
        return f"n={stat['txn_count']}"
    if "cluster_size" in stat:
        return f"n={stat['cluster_size']}"
    return ""


def _fp_label(f: dict) -> str:
    """Human-readable label for a false-positive finding."""
    entity = f["entity_ref"]
    check  = f.get("check_name", "")
    if "gl:" in entity:
        return "secondary detection (same anomaly, GL-account slice)"
    if check == "outlier_round_dollar":
        return "secondary detection (round-dollar sub-check on already-flagged vendor)"
    return "review required"


def _fp_note(fps: list[dict]) -> str:
    if not fps:
        return "no false positives"
    secondary = sum(1 for f in fps if "gl:" in f["entity_ref"] or f.get("check_name") == "outlier_round_dollar")
    if secondary == len(fps):
        return f"{len(fps)} FP(s) are secondary cross-check detections on already-flagged entities"
    return f"{len(fps)} FP(s) — review individually"


if __name__ == "__main__":
    main()
