"""
Policy gate — three rules enforced before and after key agent actions.

Every gate call appends a POLICY_GATE event to the ledger so the full
decision trail is auditable even when no violation occurs.
"""
from __future__ import annotations

from agent.ledger import Ledger
from agent.models import Event, Finding


class PolicyViolation(Exception):
    """Raised when a gate returns FAIL; the caller should halt the run."""


class PolicyGate:
    def __init__(
        self,
        ledger: Ledger,
        run_id: str,
        allowed_paths: list[str],
        budget_usd: float,
    ) -> None:
        self._ledger = ledger
        self._run_id = run_id
        self._allowed = [str(p).replace("\\", "/").lower() for p in allowed_paths]
        self._budget = budget_usd

    # ── Rule 1: scope guard ───────────────────────────────────────────────────

    def scope_guard(
        self, requested_path: str, parent_event_id: str | None = None
    ) -> str:
        """Verify the requested dataset path is in the engagement allowlist.

        Returns the event_id of the gate event so the caller can chain it as
        a parent for subsequent events.
        """
        path_key = str(requested_path).replace("\\", "/").lower()
        passed = any(path_key.endswith(a) or a.endswith(path_key) for a in self._allowed)
        status = "PASS" if passed else "FAIL"
        rule = "SCOPE_GUARD: dataset path must match engagement allowlist"

        event = self._ledger.append_event(Event(
            run_id=self._run_id,
            step_type="POLICY_GATE",
            actor="POLICY",
            input_summary={"requested_path": str(requested_path)},
            output_summary={"allowed_paths": self._allowed, "result": status},
            model="deterministic",
            reasoning=(
                f"{status}: '{path_key}' "
                + ("is in allowlist." if passed else "is NOT in allowlist — halting.")
            ),
            policy_status=status,
            policy_rule=rule,
            parent_event_id=parent_event_id,
        ))

        if not passed:
            raise PolicyViolation(
                f"SCOPE_GUARD FAIL: '{requested_path}' not in allowlisted paths {self._allowed}"
            )
        return event.event_id

    # ── Rule 2: cost ceiling ──────────────────────────────────────────────────

    def budget_guard(self, parent_event_id: str | None = None) -> str:
        """Halt if cumulative run cost has reached or exceeded the budget ceiling.

        Returns the event_id so the caller can chain it.
        """
        current_cost = self._ledger.get_run_cost(self._run_id)
        passed = current_cost < self._budget
        status = "PASS" if passed else "FAIL"
        rule = f"COST_CEILING: cumulative cost < ${self._budget:.2f}"

        event = self._ledger.append_event(Event(
            run_id=self._run_id,
            step_type="POLICY_GATE",
            actor="POLICY",
            input_summary={"current_cost_usd": round(current_cost, 6), "budget_usd": self._budget},
            output_summary={"result": status},
            model="deterministic",
            reasoning=(
                f"{status}: ${current_cost:.4f} "
                + ("< budget." if passed else f">= ${self._budget:.2f} — halting run.")
            ),
            policy_status=status,
            policy_rule=rule,
            parent_event_id=parent_event_id,
        ))

        if not passed:
            raise PolicyViolation(
                f"COST_CEILING FAIL: ${current_cost:.4f} >= budget ${self._budget:.2f}"
            )
        return event.event_id

    # ── Rule 3: approval gate ─────────────────────────────────────────────────

    def approval_gate(
        self, finding: Finding, parent_event_id: str | None = None
    ) -> tuple[str, str]:
        """Log an APPROVAL_REQUEST for HIGH/CRITICAL findings and auto-approve (MVP).

        Production would pause for human sign-off. Here the gate is logged so
        the audit trail shows it existed.

        Returns (review_status, event_id).
        """
        if finding.severity not in ("HIGH", "CRITICAL"):
            return finding.review_status, ""

        event = self._ledger.append_event(Event(
            run_id=self._run_id,
            step_type="APPROVAL_REQUEST",
            actor="POLICY",
            input_summary={
                "finding_id": finding.finding_id,
                "control_id": finding.control_id,
                "entity_ref": finding.entity_ref,
                "severity": finding.severity,
            },
            output_summary={
                "decision": "AUTO_APPROVED",
                "note": "MVP: auto-approve; gate logged for audit trail.",
            },
            model="deterministic",
            reasoning=(
                f"{finding.severity} finding ({finding.control_id} / {finding.entity_ref}) "
                "requires approval gate per policy. "
                "MVP mode: auto-approved. Production pauses for human reviewer."
            ),
            policy_status="PASS",
            policy_rule="APPROVAL_GATE: HIGH/CRITICAL findings require approval before final report",
            parent_event_id=parent_event_id,
        ))

        return "APPROVED", event.event_id
