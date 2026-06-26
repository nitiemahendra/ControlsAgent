"""
Four deterministic check executors for AP ledger controls testing.

Each function is pure (no side effects):
  inputs:  pandas DataFrame, config dict, run_id str, source_event_id str
  output:  list[Finding]

The agent loop creates events, applies policy gates, and persists findings.
Checks never touch the DB directly.

Why deterministic Python, not an LLM: arithmetic must be defensible in a
workpaper; a stochastic model introduces variance you cannot sign off on.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from agent.models import Finding

# ── AP-BEN-01 ─────────────────────────────────────────────────────────────────

BENFORD = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
_BENFORD_ARR = np.array([BENFORD[d] for d in range(1, 10)])
# Use chi-square p-value (not raw MAD) as the primary gate: MAD is sample-size-
# sensitive and fires on small Benford-conforming populations. Chi-square at
# p<0.005 is robust down to ~100 observations.
CHI2_P_THRESHOLD = 0.005
MIN_BENFORD_POP = 100


def _leading_digit(val: float) -> int | None:
    s = str(abs(val)).replace(".", "").lstrip("0")
    return int(s[0]) if s else None


def run_benford(
    df: pd.DataFrame,
    config: dict,
    run_id: str,
    source_event_id: str,
) -> list[Finding]:
    """AP-BEN-01: Benford first-digit conformity, sliced by vendor and GL account."""
    findings: list[Finding] = []
    materiality = float(config.get("materiality_floor", 5_000))

    def _check_slice(grp: pd.DataFrame, slice_key: str) -> Finding | None:
        if len(grp) < MIN_BENFORD_POP:
            return None

        digits = grp["amount"].apply(_leading_digit).dropna().astype(int)
        n = len(digits)
        obs_counts = np.array([(digits == d).sum() for d in range(1, 10)], dtype=float)
        obs_freq = obs_counts / n

        mad = float(np.abs(obs_freq - _BENFORD_ARR).mean())
        exp_counts = _BENFORD_ARR * n
        chi2_stat, p_val = stats.chisquare(obs_counts, f_exp=exp_counts)
        # Require both statistical significance AND practical significance.
        # Chi-square alone fires on tiny deviations in large GL-account slices;
        # MAD > 0.025 filters those out while keeping genuine anomalies (MAD ~0.05+).
        MAD_FLOOR = 0.025
        if p_val >= CHI2_P_THRESHOLD or mad <= MAD_FLOOR:
            return None

        per_digit = {
            str(d): {
                "observed": round(float(obs_freq[d - 1]), 4),
                "expected": round(float(_BENFORD_ARR[d - 1]), 4),
                "delta":    round(float(obs_freq[d - 1] - _BENFORD_ARR[d - 1]), 4),
            }
            for d in range(1, 10)
        }
        dominant = [str(d) for d in range(1, 10)
                    if float(obs_freq[d - 1] - _BENFORD_ARR[d - 1]) > 0.04]

        total_amount = float(grp["amount"].sum())
        if mad > 0.04 and total_amount >= materiality:
            severity = "HIGH"
        elif mad > 0.025:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return Finding(
            run_id=run_id,
            control_id="AP-BEN-01",
            control_objective="Detect systematic manipulation or fabrication in transaction amounts",
            check_name="benford_conformity",
            entity_ref=slice_key,
            evidence={
                "slice": slice_key,
                "population_size": n,
                "per_digit": per_digit,
                "dominant_digits": dominant,
                "total_amount": round(total_amount, 2),
            },
            statistic={
                "mad": round(mad, 6),
                "chi2": round(float(chi2_stat), 3),
                "p_value": round(float(p_val), 6),
                "population": n,
            },
            severity=severity,
            severity_rationale=(
                f"MAD {mad:.4f} (chi2 p={p_val:.4f} < {CHI2_P_THRESHOLD}); "
                f"slice={slice_key}, n={n}, total=${total_amount:,.2f}. "
                "Benford deviation is a smoke signal, not proof; investigate digit concentration."
            ),
            recommended_action=(
                f"Review transactions in slice '{slice_key}'; compare to prior periods; "
                "focus on over-represented digits: "
                + (", ".join(dominant) if dominant else "none above threshold")
            ),
            source_event_id=source_event_id,
        )

    for vendor, grp in df.groupby("vendor_id"):
        f = _check_slice(grp, f"vendor:{vendor}")
        if f:
            findings.append(f)

    for gl, grp in df.groupby("gl_account"):
        f = _check_slice(grp, f"gl:{gl}")
        if f:
            findings.append(f)

    return findings


# ── AP-DUP-01 ─────────────────────────────────────────────────────────────────

FUZZY_WINDOW_DAYS = 7


def run_duplicates(
    df: pd.DataFrame,
    config: dict,
    run_id: str,
    source_event_id: str,
) -> list[Finding]:
    """AP-DUP-01: Exact and fuzzy duplicate payment detection."""
    findings: list[Finding] = []
    materiality = float(config.get("materiality_floor", 5_000))

    # (a) Exact: same vendor_id + invoice_number + amount
    exact_keys = ["vendor_id", "invoice_number", "amount"]
    dup_mask = df.duplicated(subset=exact_keys, keep=False)
    exact_pairs: set[frozenset] = set()

    for key, grp in df[dup_mask].groupby(exact_keys):
        vendor, inv_num, amount = key
        amount = float(amount)
        txn_ids = grp["txn_id"].tolist()
        pay_dates = grp["payment_date"].tolist()
        severity = "HIGH" if amount >= materiality else "MEDIUM"

        for a, b in combinations(txn_ids, 2):
            exact_pairs.add(frozenset([a, b]))

        findings.append(Finding(
            run_id=run_id,
            control_id="AP-DUP-01",
            control_objective="Prevent duplicate or double payments to the same vendor for the same invoice",
            check_name="duplicate_payments_exact",
            entity_ref=f"vendor:{vendor}",
            evidence={
                "vendor_id": str(vendor),
                "invoice_number": str(inv_num),
                "txn_ids": txn_ids,
                "amount": amount,
                "payment_dates": pay_dates,
            },
            statistic={
                "cluster_size": len(txn_ids),
                "dollars_at_risk": round(amount * (len(txn_ids) - 1), 2),
            },
            severity=severity,
            severity_rationale=(
                f"Exact match: vendor={vendor}, invoice={inv_num}, amount=${amount:,.2f}; "
                f"{len(txn_ids)} payments on {pay_dates}. "
                f"${amount * (len(txn_ids) - 1):,.2f} at risk."
            ),
            recommended_action=(
                f"Confirm with AP whether all {len(txn_ids)} payments were authorized; "
                f"recover ${amount * (len(txn_ids) - 1):,.2f} if confirmed duplicate."
            ),
            source_event_id=source_event_id,
        ))

    # (b) Fuzzy: same vendor + same amount, different invoice, within FUZZY_WINDOW_DAYS
    df2 = df.copy()
    df2["_pay_dt"] = pd.to_datetime(df2["payment_date"])
    seen: set[frozenset] = set()

    for (vendor, amount), grp in df2.groupby(["vendor_id", "amount"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("_pay_dt").reset_index(drop=True)
        rows = grp.to_dict("records")

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                days = int((rows[j]["_pay_dt"] - rows[i]["_pay_dt"]).days)
                if days > FUZZY_WINDOW_DAYS:
                    break
                if rows[i]["invoice_number"] == rows[j]["invoice_number"]:
                    continue
                pair = frozenset([rows[i]["txn_id"], rows[j]["txn_id"]])
                if pair in seen or pair in exact_pairs:
                    continue
                seen.add(pair)

                findings.append(Finding(
                    run_id=run_id,
                    control_id="AP-DUP-01",
                    control_objective="Prevent near-duplicate payments (same vendor + amount, different invoice, within 7 days)",
                    check_name="duplicate_payments_fuzzy",
                    entity_ref=f"vendor:{vendor}",
                    evidence={
                        "vendor_id": str(vendor),
                        "txn_ids": [rows[i]["txn_id"], rows[j]["txn_id"]],
                        "invoice_numbers": [rows[i]["invoice_number"], rows[j]["invoice_number"]],
                        "amount": float(amount),
                        "payment_dates": [str(rows[i]["_pay_dt"].date()), str(rows[j]["_pay_dt"].date())],
                        "days_apart": days,
                    },
                    statistic={
                        "cluster_size": 2,
                        "dollars_at_risk": float(amount),
                        "days_between": days,
                    },
                    severity="MEDIUM",
                    severity_rationale=(
                        f"Vendor {vendor} paid ${float(amount):,.2f} twice within {days} day(s) "
                        f"on different invoices. May be legitimate or near-duplicate."
                    ),
                    recommended_action=(
                        "Verify both invoices support independent deliverables; "
                        "check if this vendor has a pattern of same-amount billing."
                    ),
                    source_event_id=source_event_id,
                ))

    return findings


# ── AP-SOD-01 ─────────────────────────────────────────────────────────────────

_THRESHOLD_BAND_FACTOR = 0.94   # lower bound = threshold * 0.94 (e.g. $9,400 for a $10k threshold)
_MIN_CLUSTER = 3                # minimum txns before flagging threshold clustering


def run_sod(
    df: pd.DataFrame,
    config: dict,
    run_id: str,
    source_event_id: str,
) -> list[Finding]:
    """AP-SOD-01: Self-approval, over-authority, and threshold clustering."""
    findings: list[Finding] = []
    role_limits: dict[str, float] = {k: float(v) for k, v in config["role_limits"].items()}
    threshold = float(config["approval_threshold"])
    materiality = float(config.get("materiality_floor", 5_000))
    band_lo = threshold * _THRESHOLD_BAND_FACTOR

    def _limit(role: str) -> float:
        return role_limits.get(str(role).lower(), 0.0)

    # (a) Self-approval
    self_app = df[df["created_by"] == df["approved_by"]]
    for user, grp in self_app.groupby("created_by"):
        txn_ids = grp["txn_id"].tolist()
        amounts = [float(a) for a in grp["amount"]]
        total = sum(amounts)
        severity = "CRITICAL" if total >= materiality else "HIGH"
        findings.append(Finding(
            run_id=run_id,
            control_id="AP-SOD-01",
            control_objective="Ensure no user can both create and approve their own transactions",
            check_name="sod_self_approval",
            entity_ref=f"user:{user}",
            evidence={
                "user": str(user),
                "txn_ids": txn_ids,
                "amounts": amounts,
                "vendors": grp["vendor_id"].tolist(),
            },
            statistic={"txn_count": len(txn_ids), "total_amount": round(total, 2)},
            severity=severity,
            severity_rationale=(
                f"User {user} created AND approved {len(txn_ids)} transaction(s) "
                f"totaling ${total:,.2f}; no independent reviewer."
            ),
            recommended_action=(
                f"Escalate all self-approved transactions by {user} to management; "
                "investigate whether SoD system controls were bypassed or misconfigured."
            ),
            source_event_id=source_event_id,
        ))

    # (b) Over-authority: amount > approver's role limit
    over_auth = df[df.apply(lambda r: float(r["amount"]) > _limit(r["approver_role"]), axis=1)]
    for approver, grp in over_auth.groupby("approved_by"):
        role = str(grp.iloc[0]["approver_role"])
        lim = _limit(role)
        txn_ids = grp["txn_id"].tolist()
        max_amt = float(grp["amount"].max())
        findings.append(Finding(
            run_id=run_id,
            control_id="AP-SOD-01",
            control_objective="Ensure approvals do not exceed the approver's delegated authority limit",
            check_name="sod_over_authority",
            entity_ref=f"user:{approver}",
            evidence={
                "approver": str(approver),
                "role": role,
                "authorized_limit": lim,
                "txn_ids": txn_ids,
                "amounts": [float(a) for a in grp["amount"]],
            },
            statistic={
                "txn_count": len(txn_ids),
                "max_amount": round(max_amt, 2),
                "authorized_limit": lim,
                "max_excess": round(max_amt - lim, 2),
            },
            severity="HIGH",
            severity_rationale=(
                f"Approver {approver} (role: {role}, limit: ${lim:,.0f}) approved "
                f"{len(txn_ids)} transaction(s); max ${max_amt:,.2f} exceeds limit by ${max_amt - lim:,.2f}."
            ),
            recommended_action=(
                f"Re-approve via a qualified approver; verify {approver}'s role limit is correctly configured."
            ),
            source_event_id=source_event_id,
        ))

    # (c) Threshold clustering: amounts in [band_lo, threshold) per (creator, vendor)
    near = df[(df["amount"] >= band_lo) & (df["amount"] < threshold)]
    for (creator, vendor), grp in near.groupby(["created_by", "vendor_id"]):
        if len(grp) < _MIN_CLUSTER:
            continue
        txn_ids = grp["txn_id"].tolist()
        amounts = [float(a) for a in grp["amount"]]
        findings.append(Finding(
            run_id=run_id,
            control_id="AP-SOD-01",
            control_objective="Detect threshold gaming — transactions clustered just below an approval limit",
            check_name="sod_threshold_clustering",
            entity_ref=f"user:{creator}|vendor:{vendor}",
            evidence={
                "creator": str(creator),
                "vendor": str(vendor),
                "threshold": threshold,
                "band_lo": round(band_lo, 2),
                "txn_ids": txn_ids,
                "amounts": amounts,
            },
            statistic={
                "cluster_size": len(grp),
                "mean_amount": round(float(grp["amount"].mean()), 2),
                "total_amount": round(sum(amounts), 2),
                "threshold": threshold,
            },
            severity="MEDIUM",
            severity_rationale=(
                f"{len(grp)} transactions by {creator} for vendor {vendor} cluster "
                f"in ${band_lo:,.0f}–${threshold:,.0f}, just below the ${threshold:,.0f} threshold. "
                "Pattern is consistent with threshold gaming but requires judgment to confirm."
            ),
            recommended_action=(
                f"Review all {len(grp)} transactions by {creator} for vendor {vendor}; "
                "determine whether aggregate exceeds approval limit; consider escalated review."
            ),
            source_event_id=source_event_id,
        ))

    return findings


# ── AP-OUT-01 ─────────────────────────────────────────────────────────────────

_Z_THRESHOLD = 3.0
_IQR_FACTOR = 1.5
_MIN_VENDOR_POP = 10
_ROUND_MULTIPLE = 1_000.0


def run_outliers(
    df: pd.DataFrame,
    config: dict,
    run_id: str,
    source_event_id: str,
) -> list[Finding]:
    """AP-OUT-01: Round-dollar (elevated rate) and per-vendor amount outliers."""
    findings: list[Finding] = []
    materiality = float(config.get("materiality_floor", 5_000))

    n_total = len(df)
    n_round_pop = int((df["amount"] % _ROUND_MULTIPLE == 0).sum())
    pop_rate = n_round_pop / n_total if n_total > 0 else 0.0

    # (a) Round-dollar: exact multiple of $1,000, above materiality, elevated vs population rate
    for vendor, grp in df.groupby("vendor_id"):
        round_mask = grp["amount"] % _ROUND_MULTIPLE == 0
        large_round = grp[round_mask & (grp["amount"] >= materiality)]
        if len(large_round) == 0:
            continue

        vendor_rate = float(round_mask.mean())
        rate_ratio = (vendor_rate / pop_rate) if pop_rate > 0 else float("inf")

        if len(large_round) < 3 and rate_ratio < 5.0:
            continue

        findings.append(Finding(
            run_id=run_id,
            control_id="AP-OUT-01",
            control_objective="Surface potentially fabricated or abnormal round-dollar transactions",
            check_name="outlier_round_dollar",
            entity_ref=f"vendor:{vendor}",
            evidence={
                "vendor_id": str(vendor),
                "txn_ids": large_round["txn_id"].tolist(),
                "amounts": [float(a) for a in large_round["amount"]],
                "vendor_round_rate": round(vendor_rate, 4),
                "population_round_rate": round(pop_rate, 4),
            },
            statistic={
                "round_count": len(large_round),
                "vendor_rate": round(vendor_rate, 4),
                "population_rate": round(pop_rate, 4),
                "rate_ratio": round(rate_ratio, 2) if rate_ratio != float("inf") else None,
                "total_round_amount": round(float(large_round["amount"].sum()), 2),
            },
            severity="LOW",
            severity_rationale=(
                f"Vendor {vendor}: {len(large_round)} large round-dollar transactions "
                f"(rate={vendor_rate:.1%} vs pop. {pop_rate:.1%}). "
                "Round amounts are noisy — kept LOW; escalate only if corroborated by another finding."
            ),
            recommended_action=(
                "Verify each round-dollar amount is supported by an original invoice; "
                "cross-reference AP-BEN-01 and AP-DUP-01 findings for this vendor."
            ),
            source_event_id=source_event_id,
        ))

    # (b) Per-vendor outlier: z-score on LOG amounts.
    # Log-transform is essential here: transaction amounts are log-normally
    # distributed, so raw z-scores produce massive false-positive rates on
    # the right tail. In log space, uniform/log-normal data has bounded z-scores
    # and genuine outliers (e.g. a $45k transaction against a $3k vendor mean)
    # remain clearly detectable.
    for vendor, grp in df.groupby("vendor_id"):
        if len(grp) < _MIN_VENDOR_POP:
            continue

        amounts = grp["amount"].astype(float)
        log_amounts = np.log(amounts)
        log_mean = float(log_amounts.mean())
        log_std = float(log_amounts.std(ddof=1))
        if log_std == 0:
            continue

        # IQR fence computed in log space too
        lq1 = float(log_amounts.quantile(0.25))
        lq3 = float(log_amounts.quantile(0.75))
        liqr = lq3 - lq1
        log_upper = (lq3 + _IQR_FACTOR * liqr) if liqr > 0 else float("inf")
        log_lower = (lq1 - _IQR_FACTOR * liqr) if liqr > 0 else float("-inf")

        z_scores = (log_amounts - log_mean) / log_std
        outlier_mask = (
            (z_scores.abs() > _Z_THRESHOLD)
            | (log_amounts > log_upper)
            | (log_amounts < log_lower)
        )
        outliers = grp[outlier_mask].copy()
        if outliers.empty:
            continue

        outliers = outliers.copy()
        outliers["_z"] = z_scores[outlier_mask]

        # Report raw-space stats for readability
        raw_mean = float(amounts.mean())
        raw_std = float(amounts.std(ddof=1))

        for _, row in outliers.iterrows():
            amount = float(row["amount"])
            z = float(row["_z"])
            iqr_mult = (
                round((math.log(amount) - lq3) / liqr, 2)
                if liqr > 0 and math.log(amount) > lq3
                else None
            )

            if abs(z) >= 5.0 and amount >= materiality:
                severity = "HIGH"
            elif abs(z) >= 4.0:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            findings.append(Finding(
                run_id=run_id,
                control_id="AP-OUT-01",
                control_objective="Surface abnormal transaction amounts relative to vendor payment history",
                check_name="outlier_vendor_amount",
                entity_ref=f"vendor:{vendor}",
                evidence={
                    "txn_id": str(row["txn_id"]),
                    "vendor_id": str(vendor),
                    "amount": amount,
                    "vendor_mean": round(raw_mean, 2),
                    "vendor_std": round(raw_std, 2),
                    "vendor_median": round(float(amounts.median()), 2),
                    "population_size": int(len(grp)),
                },
                statistic={
                    "z": round(z, 3),
                    "log_z": round(z, 3),
                    "iqr_multiple": iqr_mult,
                    "vendor_mean": round(raw_mean, 2),
                    "vendor_std": round(raw_std, 2),
                },
                severity=severity,
                severity_rationale=(
                    f"Transaction {row['txn_id']} (${amount:,.2f}) for vendor {vendor} "
                    f"is {abs(z):.1f}σ in log-space from vendor log-mean "
                    f"(raw mean=${raw_mean:,.2f}, raw std=${raw_std:,.2f}, n={len(grp)})."
                    + (f" Log-IQR multiple: {iqr_mult}x." if iqr_mult else "")
                ),
                recommended_action=(
                    f"Request original invoice for {row['txn_id']}; "
                    f"compare against prior transactions with vendor {vendor}."
                ),
                source_event_id=source_event_id,
            ))

    return findings
