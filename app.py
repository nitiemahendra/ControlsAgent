"""
Streamlit report viewer for the Auditable Internal-Controls Agent.

Usage (local):
    streamlit run app.py

Usage (Cloud Run):
    docker build -t controls-agent . && docker run -p 8080:8080 controls-agent
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.ledger import Ledger
from agent.loop import run_agent

# ── Config ─────────────────────────────────────────────────────────────────────

DB_PATH   = os.environ.get("DB_PATH",   "controls_agent.db")
DATA_PATH = os.environ.get("DATA_PATH", "data/transactions.csv")
GT_PATH   = "data/ground_truth.json"

_SEV_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
_SEV_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
_STEP_EMOJI = {
    "POLICY_GATE": "🛡️", "TOOL_CALL": "🔧", "PLAN": "🗺️",
    "CHECK": "✅", "CLASSIFY": "🧠", "APPROVAL_REQUEST": "👤", "REPORT": "📄",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sev_badge(sev: str) -> str:
    return f"{_SEV_EMOJI.get(sev, '')} {sev}"


def _fmt_stat(stat: dict) -> str:
    if not stat:
        return ""
    if "mad" in stat:
        return f"MAD={stat['mad']:.4f}  p={stat.get('chi2_p', 0):.4f}"
    if "dollars_at_risk" in stat:
        return f"${stat['dollars_at_risk']:,.0f} at risk"
    if "z" in stat:
        return f"z={stat['z']:.1f}  IQR-fence={stat.get('iqr_fence', '')}"
    if "round_count" in stat:
        return f"n={stat['round_count']} round-dollar"
    if "txn_count" in stat:
        return f"n={stat['txn_count']}"
    if "cluster_size" in stat:
        return f"n={stat['cluster_size']}"
    return json.dumps(stat)[:60]


def _fmt_check(name: str) -> str:
    labels = {
        "benford_vendor":          "Benford vendor",
        "benford_gl":              "Benford GL slice",
        "duplicate_payments_exact":"Duplicate exact",
        "sod_self_approval":       "Self-approval",
        "sod_over_authority":      "Over-authority",
        "sod_threshold_clustering":"Threshold cluster",
        "outlier_vendor_amount":   "Vendor outlier",
        "outlier_round_dollar":    "Round-dollar",
    }
    return labels.get(name, name)


def _fmt_evidence(ev: dict | list | str) -> str:
    if isinstance(ev, str):
        return ev
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        lines = []
        for item in ev[:8]:
            lines.append("  " + "  ·  ".join(f"{k}: {v}" for k, v in item.items()))
        if len(ev) > 8:
            lines.append(f"  … {len(ev) - 8} more")
        return "\n".join(lines)
    return json.dumps(ev, indent=2)[:400]


# ── Eval helpers (inline, no import from run_eval) ─────────────────────────────

@dataclass
class _EP:
    control_id: str
    entity_key: str
    description: str
    check_name: str = ""


def _build_expected(gt: dict) -> list[_EP]:
    eps: list[_EP] = []
    eps.append(_EP("AP-BEN-01", gt["AP-BEN-01"]["vendor"], f"Benford non-conformity ({gt['AP-BEN-01']['vendor']})"))
    for p in gt["AP-DUP-01"]:
        eps.append(_EP("AP-DUP-01", p["vendor"], f"Dup {p['invoice_number']} ${p['amount']:,.0f}", "duplicate_payments_exact"))
    sod = gt["AP-SOD-01"]
    eps.append(_EP("AP-SOD-01", "U-07",               "Self-approval (U-07)",                     "sod_self_approval"))
    eps.append(_EP("AP-SOD-01", "U-03",               "Over-authority (U-03)",                    "sod_over_authority"))
    eps.append(_EP("AP-SOD-01", sod["threshold_vendor"], f"Threshold ({sod['threshold_vendor']})", "sod_threshold_clustering"))
    out = gt["AP-OUT-01"]
    eps.append(_EP("AP-OUT-01", out["vendor_outlier"]["vendor"], f"Outlier ${out['vendor_outlier']['amount']:,.0f}", "outlier_vendor_amount"))
    eps.append(_EP("AP-OUT-01", "V-0006", "Round-dollar (V-0006)", "outlier_round_dollar"))
    return eps


def _match(findings: list[dict], expected: list[_EP]):
    remaining = list(expected)
    tps, fps = [], []
    for f in findings:
        m = next((e for e in remaining
                  if f["control_id"] == e.control_id
                  and e.entity_key in f["entity_ref"]
                  and (not e.check_name or f.get("check_name") == e.check_name)), None)
        if m:
            tps.append((m, f))
            remaining.remove(m)
        else:
            fps.append(f)
    return tps, fps, remaining


def _metrics(tp: int, fp: int, fn: int):
    p  = tp / (tp + fp) if tp + fp else 0.0
    r  = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r)  if p + r  else 0.0
    return p, r, f1


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Controls Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 Auditable Internal-Controls Agent")
st.caption("Full-population AP ledger testing · Append-only decision ledger · Workpaper-ready findings")

# ── Run Audit / Reset buttons ──────────────────────────────────────────────────

col_btn, col_reset, col_msg = st.columns([1, 1, 4])
with col_btn:
    if st.button("▶  Run Audit", type="primary", use_container_width=True):
        with st.spinner("Running agent on 2,760 transactions…"):
            result = run_agent(dataset_path=DATA_PATH, db_path=DB_PATH)
        st.session_state["selected_run"] = result.run_id
        col_msg.success(
            f"Run `{result.run_id[:8]}` · **{result.status}** · "
            f"{result.finding_count} findings · ${result.total_cost_usd:.4f}"
        )
        st.rerun()

with col_reset:
    if st.button("🗑  Reset", use_container_width=True):
        Path(DB_PATH).unlink(missing_ok=True)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────

ledger = Ledger(DB_PATH)
runs   = ledger.list_runs()

if not runs:
    st.info("No audit runs found. Click **▶ Run Audit** to start.")
    st.stop()

with st.sidebar:
    st.header("Run history")
    run_options = list(reversed(runs))
    run_labels  = {
        r["run_id"]: f"{r['started_at'][:16]}  ·  {r['event_count']}ev"
        for r in run_options
    }
    default_id = st.session_state.get("selected_run", run_options[0]["run_id"])
    if default_id not in run_labels:
        default_id = run_options[0]["run_id"]

    selected_run_id = st.selectbox(
        "Run",
        options=list(run_labels),
        format_func=lambda r: run_labels[r],
        index=list(run_labels).index(default_id),
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Severity filter")
    sev_filter = st.multiselect(
        "Severities",
        options=_SEV_ORDER,
        default=_SEV_ORDER,
        label_visibility="collapsed",
    )

# ── Load run data ─────────────────────────────────────────────────────────────

findings = ledger.get_findings(selected_run_id)
events   = ledger.get_events(selected_run_id)
run_meta = next((r for r in runs if r["run_id"] == selected_run_id), {})

# ── Metrics ────────────────────────────────────────────────────────────────────

sev_counts = {s: sum(1 for f in findings if f["severity"] == s) for s in _SEV_ORDER}
cost_usd   = run_meta.get("total_cost_usd", 0.0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("🔴 CRITICAL", sev_counts["CRITICAL"])
c2.metric("🟠 HIGH",     sev_counts["HIGH"])
c3.metric("🟡 MEDIUM",   sev_counts["MEDIUM"])
c4.metric("🔵 LOW",      sev_counts["LOW"])
c5.metric("Events",      run_meta.get("event_count", len(events)))
c6.metric("💰 LLM cost", f"${cost_usd:.4f}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_findings, tab_events, tab_eval = st.tabs(["📋 Findings", "📜 Event Log", "📊 Eval"])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Findings
# ═══════════════════════════════════════════════════════════════════════════════

with tab_findings:
    filtered = [f for f in findings if f["severity"] in sev_filter]

    if not filtered:
        st.info("No findings match the active severity filter.")
    else:
        tbl_rows = []
        for f in filtered:
            tbl_rows.append({
                "Severity":  _sev_badge(f["severity"]),
                "Control":   f["control_id"],
                "Entity":    f["entity_ref"],
                "Check":     _fmt_check(f.get("check_name", "")),
                "Statistic": _fmt_stat(f["statistic"]),
                "Status":    f.get("review_status", "PENDING"),
                "_id":       f["finding_id"],
            })
        df = pd.DataFrame(tbl_rows)

        table_key = f"tbl_{selected_run_id[:8]}"
        sel_state = st.dataframe(
            df.drop(columns=["_id"]),
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key=table_key,
        )

        sel_rows = sel_state.selection.rows if sel_state.selection else []

        if sel_rows:
            fid     = tbl_rows[sel_rows[0]]["_id"]
            finding = next(f for f in filtered if f["finding_id"] == fid)
            trail   = ledger.get_finding_trail(fid)

            st.divider()

            # Finding detail
            left, right = st.columns([3, 2])

            with left:
                st.subheader(
                    f"{_sev_badge(finding['severity'])}  ·  "
                    f"{finding['control_id']}  ·  {finding['entity_ref']}"
                )
                st.caption(f"**Objective:** {finding['control_objective']}")

                st.markdown("**Evidence**")
                st.code(_fmt_evidence(finding["evidence"]), language=None)

                st.markdown("**Severity rationale**")
                st.markdown(finding.get("severity_rationale", "_Not available_"))

                st.markdown("**Recommended action**")
                st.markdown(finding.get("recommended_action", "_Not available_"))

            with right:
                st.subheader("Audit trail")
                st.caption("Event chain from root → this finding")
                for i, ev in enumerate(trail["event_trail"]):
                    icon = _STEP_EMOJI.get(ev["step_type"], "•")
                    depth = "  " * i
                    with st.expander(
                        f"{depth}{icon} `{ev['step_type']}`  ·  {ev['actor']}  ·  {ev['ts'][:19]}",
                        expanded=(i == len(trail["event_trail"]) - 1),
                    ):
                        st.markdown(f"**Reasoning:** {ev.get('reasoning', '')}")
                        if ev.get("model") and ev["model"] != "deterministic":
                            st.markdown(
                                f"Model: `{ev['model']}`  ·  "
                                f"Tokens: {ev.get('tokens_in', 0)}↑ {ev.get('tokens_out', 0)}↓  ·  "
                                f"Cost: ${ev.get('cost_usd', 0):.5f}"
                            )
                        pol = ev.get("policy_status")
                        if pol:
                            badge = "✅" if pol == "PASS" else "❌"
                            st.markdown(f"Policy: {badge} `{pol}`  rule: `{ev.get('policy_rule', '')}`")
        else:
            st.caption("Click a row to see finding detail and the complete audit trail.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Event Log
# ═══════════════════════════════════════════════════════════════════════════════

with tab_events:
    if not events:
        st.info("No events found for this run.")
    else:
        ev_rows = []
        for ev in events:
            ev_rows.append({
                "Timestamp":  ev["ts"][:19],
                "Step":       f"{_STEP_EMOJI.get(ev['step_type'], '')} {ev['step_type']}",
                "Actor":      ev["actor"],
                "Model":      ev.get("model", "") or "",
                "Tok↑":       ev.get("tokens_in", 0) or 0,
                "Tok↓":       ev.get("tokens_out", 0) or 0,
                "Cost $":     round(ev.get("cost_usd", 0) or 0, 5),
                "Policy":     ev.get("policy_status", "") or "",
                "Reasoning":  (ev.get("reasoning", "") or "")[:80],
            })
        ev_df = pd.DataFrame(ev_rows)
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

        # Event type breakdown
        step_counts = {}
        for ev in events:
            step_counts[ev["step_type"]] = step_counts.get(ev["step_type"], 0) + 1
        st.caption("  ·  ".join(f"`{k}` ×{v}" for k, v in sorted(step_counts.items())))


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Eval (precision / recall on planted anomalies)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    gt_path = Path(GT_PATH)
    if not gt_path.exists():
        st.info(f"Ground truth file not found at `{GT_PATH}`. Run `python -m synthetic.generate` first.")
    else:
        gt       = json.loads(gt_path.read_text())
        expected = _build_expected(gt)
        tps, fps, fns = _match(findings, expected)

        total_tp = len(tps)
        total_fp = len(fps)
        total_fn = len(fns)
        tot_p, tot_r, tot_f1 = _metrics(total_tp, total_fp, total_fn)

        # Per-control table
        controls = ["AP-BEN-01", "AP-DUP-01", "AP-SOD-01", "AP-OUT-01"]
        rows = []
        for ctrl in controls:
            exp_n = sum(1 for e in expected if e.control_id == ctrl)
            tp_n  = sum(1 for e, _ in tps if e.control_id == ctrl)
            fp_n  = sum(1 for f in fps if f["control_id"] == ctrl)
            fn_n  = sum(1 for e in fns if e.control_id == ctrl)
            p, r, f1 = _metrics(tp_n, fp_n, fn_n)
            rows.append({
                "Control":   ctrl,
                "Expected":  exp_n,
                "TP":        tp_n,
                "FP":        fp_n,
                "FN":        fn_n,
                "Precision": f"{p:.0%}",
                "Recall":    f"{r:.0%}",
                "F1":        f"{f1:.3f}",
            })
        rows.append({
            "Control":   "TOTAL",
            "Expected":  len(expected),
            "TP":        total_tp,
            "FP":        total_fp,
            "FN":        total_fn,
            "Precision": f"{tot_p:.0%}",
            "Recall":    f"{tot_r:.0%}",
            "F1":        f"{tot_f1:.3f}",
        })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Summary callouts
        r_col, p_col = st.columns(2)
        r_col.metric("Recall",    f"{tot_r:.0%}",  help="All planted anomalies detected / total planted")
        p_col.metric("Precision", f"{tot_p:.0%}",  help="Findings confirmed by ground truth / total findings")

        if total_fn == 0:
            st.success("✅ All planted anomalies detected (recall = 100%)")

        # TP detail
        if tps:
            with st.expander(f"True positives ({total_tp})", expanded=True):
                for ep, f in tps:
                    st.markdown(
                        f"**[TP]** `{f['control_id']}`  {ep.description}  "
                        f"— {_sev_badge(f['severity'])}  ·  {_fmt_stat(f['statistic'])}"
                    )

        # FP detail
        if fps:
            with st.expander(f"False positives ({total_fp}) — secondary cross-check detections"):
                for f in fps:
                    note = "secondary GL-account slice" if "gl:" in f["entity_ref"] else "secondary sub-check on already-flagged entity"
                    st.markdown(
                        f"**[FP]** `{f['control_id']}`  {f['entity_ref']}  "
                        f"— {_sev_badge(f['severity'])}  ·  {note}"
                    )

        # FN detail
        if fns:
            with st.expander(f"False negatives ({total_fn}) — missed anomalies"):
                for ep in fns:
                    st.markdown(f"**[FN]** `{ep.control_id}`  {ep.description}")
