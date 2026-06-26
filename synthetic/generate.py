"""
Synthetic AP transaction ledger generator with planted anomalies.

Produces two files in --output-dir (default: data/):
  transactions.csv   – clean ledger visible to the agent (no planted labels)
  ground_truth.json  – anomaly ground truth used by the eval harness
  config.json        – role-limit and threshold config loaded by checks

Planted anomaly map
  AP-BEN-01  Vendor V-0010: 200 txns biased toward 5/6 first digits (Benford non-conformity)
  AP-DUP-01  3 exact duplicate pairs (same vendor + invoice_number + amount, paid twice)
  AP-SOD-01  U-07 self-approval (5 txns); U-03 over-authority (3 txns >$5k); V-0021 threshold clustering (9 txns at $9.4k–$9.9k)
  AP-OUT-01  V-0016: 30 normal ~$3k txns + 1 outlier at $45k; 6 round-dollar txns (≥$5k)

Usage:
    python -m synthetic.generate
    python -m synthetic.generate --output-dir data
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
N_BASE = 2_500
N_VENDORS = 60

ROLE_LIMITS: dict[str, int] = {
    "junior":   5_000,
    "senior":  25_000,
    "manager": 100_000,
    "director": 500_000,
}
APPROVAL_THRESHOLD = 10_000   # threshold clustering detection band: $9k–$10k
MATERIALITY_FLOOR  =  5_000

START = datetime(2025, 1, 1)
END   = datetime(2026, 3, 31)

VENDORS = [f"V-{i:04d}" for i in range(1, N_VENDORS + 1)]

# Deterministic role assignments (no random — avoids seed-ordering issues)
_ROLE_CYCLE = (
    "junior junior senior senior manager "
    "junior junior senior manager senior "
    "junior director senior junior senior "
    "junior manager senior junior senior"
).split()
USERS: dict[str, str] = {
    f"U-{i:02d}": _ROLE_CYCLE[i - 1] for i in range(1, 21)
}
USERS["U-07"] = "junior"   # self-approval target
USERS["U-03"] = "junior"   # over-authority target
USERS["U-15"] = "manager"  # reliable approver used in planted txns ($100k limit)

# Vendor aliases used in planted anomalies
BENFORD_VENDOR   = VENDORS[9]    # V-0010
OUTLIER_VENDOR   = VENDORS[15]   # V-0016
THRESHOLD_VENDOR = VENDORS[20]   # V-0021
DUP_VENDORS      = [VENDORS[10], VENDORS[25], VENDORS[40]]

CSV_FIELDS = [
    "txn_id", "vendor_id", "invoice_number", "amount",
    "invoice_date", "payment_date", "posting_ts",
    "created_by", "approved_by", "approver_role", "gl_account",
]

GL_ACCOUNTS = ["6100", "6200", "6300", "7100", "7200"]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _rand_date(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(
        days=rng.randint(0, delta),
        hours=rng.randint(8, 18),
        minutes=rng.randint(0, 59),
    )


def _log_uniform(rng: random.Random, lo: float = 100, hi: float = 50_000) -> float:
    """Log-uniform distribution → naturally follows Benford's law."""
    return round(10 ** rng.uniform(math.log10(lo), math.log10(hi)), 2)


def _non_benford(rng: random.Random, lo: float = 5_000, hi: float = 15_000) -> float:
    """Reject-sample to bias toward 5/6 first digits, breaking Benford conformity."""
    for _ in range(500):
        val = round(rng.uniform(lo, hi), 2)
        first = int(str(abs(val)).lstrip("0").replace(".", "")[0])
        if first in (5, 6):
            return val
    return round(rng.uniform(lo, hi), 2)


def _pick_approver(rng: random.Random, creator: str, min_limit: float) -> str:
    candidates = [
        u for u, role in USERS.items()
        if u != creator and ROLE_LIMITS[role] >= min_limit
    ]
    return rng.choice(candidates) if candidates else "U-15"


# ── Generator class ───────────────────────────────────────────────────────────

class _Gen:
    def __init__(self, seed: int = SEED) -> None:
        self._rng = random.Random(seed)
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"TXN-{self._seq:05d}"

    def _inv_num(self) -> str:
        return f"INV-{self._rng.randint(10_000, 99_999)}"

    def _txn(
        self,
        vendor: str,
        amount: float,
        creator: str,
        approver: str,
        gl: str,
        invoice_number: str | None = None,
        inv_date: datetime | None = None,
        pay_date: datetime | None = None,
        planted_type: str | None = None,
    ) -> dict:
        if inv_date is None:
            inv_date = _rand_date(self._rng, START, END - timedelta(days=15))
        if pay_date is None:
            pay_date = inv_date + timedelta(days=self._rng.randint(15, 45))
        return {
            "txn_id":        self._next_id(),
            "vendor_id":     vendor,
            "invoice_number": invoice_number or self._inv_num(),
            "amount":        amount,
            "invoice_date":  inv_date.strftime("%Y-%m-%d"),
            "payment_date":  pay_date.strftime("%Y-%m-%d"),
            "posting_ts":    inv_date.isoformat(),
            "created_by":    creator,
            "approved_by":   approver,
            "approver_role": USERS[approver],
            "gl_account":    gl,
            "_planted":      planted_type,   # stripped before CSV write
        }

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        txns: list[dict] = []
        gt: dict = {
            "AP-BEN-01": {},
            "AP-DUP-01": [],
            "AP-SOD-01": {
                "self_approval":        [],
                "over_authority":       [],
                "threshold_clustering": [],
                "threshold_vendor":     THRESHOLD_VENDOR,
                "threshold_value":      APPROVAL_THRESHOLD,
            },
            "AP-OUT-01": {
                "round_dollar":   [],
                "vendor_outlier": {},
            },
        }

        self._base(txns)
        self._benford(txns, gt)
        self._duplicates(txns, gt)
        self._sod_self_approval(txns, gt)
        self._sod_over_authority(txns, gt)
        self._sod_threshold(txns, gt)
        self._vendor_outlier(txns, gt)
        self._round_dollar(txns, gt)

        self._rng.shuffle(txns)
        self._write_csv(txns, output_dir / "transactions.csv")
        self._write_json(gt, output_dir / "ground_truth.json")
        self._write_json(
            {
                "role_limits":          ROLE_LIMITS,
                "approval_threshold":   APPROVAL_THRESHOLD,
                "materiality_floor":    MATERIALITY_FLOOR,
            },
            output_dir / "config.json",
        )
        self._print_summary(txns, gt)

    # ── sections ──────────────────────────────────────────────────────────────

    def _base(self, txns: list) -> None:
        """2 500 normal transactions covering all non-planted vendors."""
        # Exclude BENFORD_VENDOR and OUTLIER_VENDOR so planted histories are clean.
        # OUTLIER_VENDOR needs a pure Gaussian history for z-score detection to work.
        normal_vendors = [v for v in VENDORS if v not in (BENFORD_VENDOR, OUTLIER_VENDOR)]
        for _ in range(N_BASE):
            vendor  = self._rng.choice(normal_vendors)
            amount  = _log_uniform(self._rng)
            creator = self._rng.choice(list(USERS))
            approver = _pick_approver(self._rng, creator, amount)
            gl      = self._rng.choice(GL_ACCOUNTS)
            txns.append(self._txn(vendor, amount, creator, approver, gl))

    def _benford(self, txns: list, gt: dict) -> None:
        """200 transactions for V-0010 with 5/6-biased amounts (AP-BEN-01)."""
        ids: list[str] = []
        for _ in range(200):
            amount   = _non_benford(self._rng)
            creator  = self._rng.choice([u for u in USERS if u != "U-07"])
            approver = _pick_approver(self._rng, creator, amount)
            t = self._txn(BENFORD_VENDOR, amount, creator, approver, "6200", planted_type="BENFORD")
            txns.append(t)
            ids.append(t["txn_id"])
        gt["AP-BEN-01"] = {"vendor": BENFORD_VENDOR, "txn_ids": ids}

    def _duplicates(self, txns: list, gt: dict) -> None:
        """3 duplicate pairs: same vendor + invoice_number + amount, paid twice (AP-DUP-01)."""
        specs = [
            (DUP_VENDORS[0], "INV-88120", 14_200.00),
            (DUP_VENDORS[1], "INV-77431",  8_750.50),
            (DUP_VENDORS[2], "INV-92011", 31_000.00),
        ]
        for vendor, inv_num, amount in specs:
            pair_ids: list[str] = []
            inv_date = _rand_date(self._rng, START, END - timedelta(days=30))
            for k in range(2):
                pay_date = inv_date + timedelta(days=k * 3)
                t = self._txn(
                    vendor, amount, "U-05", "U-15", "6100",
                    invoice_number=inv_num,
                    inv_date=inv_date, pay_date=pay_date,
                    planted_type="DUPLICATE",
                )
                txns.append(t)
                pair_ids.append(t["txn_id"])
            gt["AP-DUP-01"].append({
                "vendor": vendor, "invoice_number": inv_num,
                "amount": amount, "txn_ids": pair_ids,
            })

    def _sod_self_approval(self, txns: list, gt: dict) -> None:
        """5 txns where U-07 is both creator and approver (AP-SOD-01)."""
        for _ in range(5):
            amount = round(self._rng.uniform(500, 4_500), 2)
            vendor = self._rng.choice(VENDORS)
            t = self._txn(vendor, amount, "U-07", "U-07", "6300", planted_type="SOD_SELF")
            txns.append(t)
            gt["AP-SOD-01"]["self_approval"].append(t["txn_id"])

    def _sod_over_authority(self, txns: list, gt: dict) -> None:
        """3 txns approved by U-03 (junior, $5k limit) above that limit (AP-SOD-01)."""
        for _ in range(3):
            amount = round(self._rng.uniform(8_000, 20_000), 2)
            vendor = self._rng.choice(VENDORS)
            t = self._txn(vendor, amount, "U-01", "U-03", "7100", planted_type="SOD_AUTHORITY")
            txns.append(t)
            gt["AP-SOD-01"]["over_authority"].append(t["txn_id"])

    def _sod_threshold(self, txns: list, gt: dict) -> None:
        """9 txns for V-0021 clustered at $9 400–$9 999 (just below $10k threshold, AP-SOD-01)."""
        for _ in range(9):
            amount = round(self._rng.uniform(9_400, 9_999), 2)
            t = self._txn(THRESHOLD_VENDOR, amount, "U-09", "U-15", "6200", planted_type="SOD_THRESHOLD")
            txns.append(t)
            gt["AP-SOD-01"]["threshold_clustering"].append(t["txn_id"])

    def _vendor_outlier(self, txns: list, gt: dict) -> None:
        """30 normal ~$3k txns for V-0016 then one $45k outlier (z >> 3σ, AP-OUT-01)."""
        for _ in range(30):
            amount = max(100.0, round(self._rng.gauss(3_000, 400), 2))
            t = self._txn(OUTLIER_VENDOR, amount, "U-04", "U-15", "6100")
            txns.append(t)
        # outlier
        t = self._txn(OUTLIER_VENDOR, 45_000.00, "U-04", "U-15", "6100", planted_type="OUTLIER")
        txns.append(t)
        gt["AP-OUT-01"]["vendor_outlier"] = {
            "vendor": OUTLIER_VENDOR,
            "txn_id": t["txn_id"],
            "amount": 45_000.00,
        }

    def _round_dollar(self, txns: list, gt: dict) -> None:
        """6 large exact round-dollar amounts for V-0006 (AP-OUT-01)."""
        round_vendor = VENDORS[5]
        for amt in [5_000, 10_000, 15_000, 25_000, 50_000, 100_000]:
            t = self._txn(round_vendor, float(amt), "U-12", "U-15", "7200", planted_type="ROUND")
            txns.append(t)
            gt["AP-OUT-01"]["round_dollar"].append(t["txn_id"])

    # ── I/O ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_csv(txns: list, path: Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for t in txns:
                writer.writerow({k: t[k] for k in CSV_FIELDS})

    @staticmethod
    def _write_json(data: object, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _print_summary(txns: list, gt: dict) -> None:
        total   = len(txns)
        planted = sum(1 for t in txns if t.get("_planted"))
        sod     = gt["AP-SOD-01"]
        out     = gt["AP-OUT-01"]
        print(f"Generated {total} transactions  ({planted} planted anomalies)\n")
        print(f"  AP-BEN-01  Benford non-conformity  {len(gt['AP-BEN-01']['txn_ids']):>4} txns  vendor={gt['AP-BEN-01']['vendor']}")
        print(f"  AP-DUP-01  Duplicate pairs         {len(gt['AP-DUP-01']):>4} pairs ({len(gt['AP-DUP-01']) * 2} txns)")
        print(f"  AP-SOD-01  Self-approval           {len(sod['self_approval']):>4} txns  user=U-07")
        print(f"  AP-SOD-01  Over-authority          {len(sod['over_authority']):>4} txns  user=U-03")
        print(f"  AP-SOD-01  Threshold clustering    {len(sod['threshold_clustering']):>4} txns  vendor={sod['threshold_vendor']}")
        print(f"  AP-OUT-01  Vendor outlier          {1:>4} txns  vendor={out['vendor_outlier']['vendor']}  amount=${out['vendor_outlier']['amount']:,.2f}")
        print(f"  AP-OUT-01  Round-dollar            {len(out['round_dollar']):>4} txns  vendor={VENDORS[5]}")
        print(f"\nWrote: transactions.csv  ground_truth.json  config.json")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic AP ledger")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed (default: 42)")
    args = parser.parse_args()
    _Gen(seed=args.seed).run(Path(args.output_dir))
