from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uid() -> str:
    return str(uuid.uuid4())


@dataclass
class Event:
    run_id: str
    step_type: str          # PLAN | TOOL_CALL | POLICY_GATE | CHECK | CLASSIFY | APPROVAL_REQUEST | REPORT
    actor: str              # PLANNER | EXECUTOR | POLICY | CLASSIFIER | HUMAN
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    model: str              # gemini-2.0-flash | gemini-2.5-pro | deterministic
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    policy_status: str = "NA"       # PASS | FAIL | NA
    policy_rule: str = ""
    parent_event_id: str | None = None
    event_id: str = field(default_factory=_uid)
    ts: str = field(default_factory=_now)


@dataclass
class Finding:
    run_id: str
    control_id: str             # AP-BEN-01 | AP-DUP-01 | AP-SOD-01 | AP-OUT-01
    control_objective: str
    check_name: str
    entity_ref: str
    evidence: dict[str, Any]
    statistic: dict[str, Any]
    severity: str               # LOW | MEDIUM | HIGH | CRITICAL
    severity_rationale: str
    recommended_action: str
    source_event_id: str        # FK → events.event_id
    finding_id: str = field(default_factory=_uid)
    review_status: str = "AUTO_FLAGGED"     # AUTO_FLAGGED | APPROVED | DISMISSED
