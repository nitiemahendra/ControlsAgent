"""Append-only SQLite decision ledger.

Two tables:
  events   – every agent action (plan, tool call, policy gate, check, classify, report)
  findings – control exceptions surfaced by checks

Finding.source_event_id → Event.event_id, and Event.parent_event_id chains steps
back to the plan that triggered them. get_finding_trail() walks this chain for
full traceability – the key demo moment.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent.models import Event, Finding

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    ts              TEXT NOT NULL,
    step_type       TEXT NOT NULL,
    actor           TEXT NOT NULL,
    input_summary   TEXT NOT NULL,
    output_summary  TEXT NOT NULL,
    model           TEXT NOT NULL,
    reasoning       TEXT    DEFAULT '',
    tokens_in       INTEGER DEFAULT 0,
    tokens_out      INTEGER DEFAULT 0,
    cost_usd        REAL    DEFAULT 0.0,
    policy_status   TEXT    DEFAULT 'NA',
    policy_rule     TEXT    DEFAULT '',
    parent_event_id TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id          TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    control_id          TEXT NOT NULL,
    control_objective   TEXT NOT NULL,
    check_name          TEXT NOT NULL,
    entity_ref          TEXT NOT NULL,
    evidence            TEXT NOT NULL,
    statistic           TEXT NOT NULL,
    severity            TEXT NOT NULL,
    severity_rationale  TEXT NOT NULL,
    recommended_action  TEXT NOT NULL,
    source_event_id     TEXT NOT NULL REFERENCES events(event_id),
    review_status       TEXT DEFAULT 'AUTO_FLAGGED'
);
"""

_SEVERITY_ORDER = "CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END"


class Ledger:
    def __init__(self, db_path: str | Path = "controls_agent.db") -> None:
        self._path = str(db_path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── writes ───────────────────────────────────────────────────────────────

    def append_event(self, event: Event) -> Event:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event.event_id, event.run_id, event.ts,
                    event.step_type, event.actor,
                    json.dumps(event.input_summary),
                    json.dumps(event.output_summary),
                    event.model, event.reasoning,
                    event.tokens_in, event.tokens_out, event.cost_usd,
                    event.policy_status, event.policy_rule,
                    event.parent_event_id,
                ),
            )
        return event

    def append_finding(self, finding: Finding) -> Finding:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    finding.finding_id, finding.run_id,
                    finding.control_id, finding.control_objective,
                    finding.check_name, finding.entity_ref,
                    json.dumps(finding.evidence),
                    json.dumps(finding.statistic),
                    finding.severity, finding.severity_rationale,
                    finding.recommended_action, finding.source_event_id,
                    finding.review_status,
                ),
            )
        return finding

    # ── reads ────────────────────────────────────────────────────────────────

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY ts", (run_id,)
            ).fetchall()
        return [_deser_event(dict(r)) for r in rows]

    def get_findings(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM findings WHERE run_id=? ORDER BY {_SEVERITY_ORDER}",
                (run_id,),
            ).fetchall()
        return [_deser_finding(dict(r)) for r in rows]

    def get_finding_trail(self, finding_id: str) -> dict[str, Any]:
        """Return the finding plus its full event ancestry chain (newest → root)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM findings WHERE finding_id=?", (finding_id,)
            ).fetchone()
            if row is None:
                raise KeyError(finding_id)
            finding = _deser_finding(dict(row))

            chain: list[dict] = []
            current = finding["source_event_id"]
            seen: set[str] = set()
            while current and current not in seen:
                seen.add(current)
                ev_row = conn.execute(
                    "SELECT * FROM events WHERE event_id=?", (current,)
                ).fetchone()
                if ev_row is None:
                    break
                ev = _deser_event(dict(ev_row))
                chain.append(ev)
                current = ev.get("parent_event_id")

        return {"finding": finding, "event_trail": list(reversed(chain))}

    def get_run_cost(self, run_id: str) -> float:
        with self._conn() as conn:
            result = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM events WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return float(result[0])

    def list_runs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id,
                    MIN(ts)                         AS started_at,
                    MAX(ts)                         AS finished_at,
                    COUNT(DISTINCT event_id)        AS event_count,
                    COALESCE(SUM(cost_usd), 0.0)    AS total_cost_usd
                FROM events
                GROUP BY run_id
                ORDER BY started_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]


# ── helpers ───────────────────────────────────────────────────────────────────

def _deser_event(row: dict) -> dict:
    row["input_summary"] = json.loads(row["input_summary"])
    row["output_summary"] = json.loads(row["output_summary"])
    return row


def _deser_finding(row: dict) -> dict:
    row["evidence"] = json.loads(row["evidence"])
    row["statistic"] = json.loads(row["statistic"])
    return row
