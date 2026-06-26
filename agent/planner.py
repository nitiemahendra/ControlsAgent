"""
Gemini-flash planner: analyzes the dataset and returns a risk-prioritized check plan.

Model routing rule: planner uses the cheap/fast model (gemini-2.0-flash).
The checks themselves are deterministic Python — never an LLM.
The narrative classifier (Day 4) uses the strong model (gemini-2.5-pro).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from agent.ledger import Ledger
from agent.models import Event

# ── Model config ──────────────────────────────────────────────────────────────

PLANNER_MODEL = "gemini-2.5-flash"

# Pricing (USD per token).
_PRICE = {
    "gemini-2.5-flash": {"in": 0.30e-6,  "out": 2.50e-6},
    "gemini-2.5-pro":   {"in": 1.25e-6,  "out": 10.00e-6},
    "gemini-2.0-flash": {"in": 0.075e-6, "out": 0.30e-6},
}

# ── Available checks ──────────────────────────────────────────────────────────

AVAILABLE_CHECKS: dict[str, str] = {
    "AP-BEN-01": "Benford first-digit conformity — detect amount fabrication/manipulation",
    "AP-DUP-01": "Duplicate / near-duplicate payments — detect double payments",
    "AP-SOD-01": "Segregation-of-duties — detect self-approval, over-authority, threshold gaming",
    "AP-OUT-01": "Round-dollar & vendor outlier — detect abnormal amounts",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert AP (Accounts Payable) audit planner. Analyze a transaction dataset
and produce a risk-prioritized controls testing plan. Be specific to the numbers given.
Return ONLY valid JSON — no markdown, no commentary."""

_USER_TEMPLATE = """\
Dataset summary:
  Transactions   : {n_txns:,}
  Distinct vendors: {n_vendors}
  Invoice dates  : {date_min} to {date_max}
  Amount range   : ${amount_min:,.2f} – ${amount_max:,.2f}
  Amount mean    : ${amount_mean:,.2f}  |  median: ${amount_median:,.2f}
  GL accounts    : {gl_accounts}
  Approver roles : {roles}

Available controls checks:
{checks_list}

Return JSON with this exact structure:
{{
  "reasoning": "<one paragraph: your risk assessment of this specific dataset>",
  "risk_narrative": "<2–3 sentences: top risks for the executive summary>",
  "checks": [
    {{"check_id": "<AP-XXX-01>", "priority": <1=highest>, "rationale": "<why this check matters here>"}}
  ]
}}"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CheckPlan:
    check_id: str
    priority: int
    rationale: str


@dataclass
class RunPlan:
    reasoning: str
    risk_narrative: str
    checks: list[CheckPlan]
    plan_event_id: str
    model: str = PLANNER_MODEL
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    fallback: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_plan(plan_event_id: str) -> RunPlan:
    return RunPlan(
        reasoning="Gemini unavailable — defaulting to full check of all four controls.",
        risk_narrative="No AI risk triage available. All controls tested at equal priority.",
        checks=[
            CheckPlan(cid, i + 1, "Default plan: all checks")
            for i, cid in enumerate(AVAILABLE_CHECKS)
        ],
        plan_event_id=plan_event_id,
        model="default",
        fallback=True,
    )


def _calc_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = _PRICE.get(model, {"in": 0.0, "out": 0.0})
    return tokens_in * p["in"] + tokens_out * p["out"]


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text, stripping markdown fences if present."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    # Find the outermost {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _retry_generate(client, model: str, prompt: str, config, max_retries: int = 3):
    """Call generate_content with exponential backoff on 429 RESOURCE_EXHAUSTED."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = 4 ** attempt   # 1s → 4s → 16s
                time.sleep(wait)
                continue
            raise
    raise last_exc  # unreachable but satisfies type checkers


# ── Public API ────────────────────────────────────────────────────────────────

def create_plan(
    df: pd.DataFrame,
    run_id: str,
    ledger: Ledger,
    parent_event_id: str | None = None,
) -> RunPlan:
    """Call Gemini flash to create a risk-prioritized run plan.

    Falls back gracefully if the API key is absent or the call fails.
    Always creates a PLAN event in the ledger.
    """
    checks_list = "\n".join(f"  {k}: {v}" for k, v in AVAILABLE_CHECKS.items())

    dataset_info = {
        "n_txns":       len(df),
        "n_vendors":    int(df["vendor_id"].nunique()),
        "date_min":     str(df["invoice_date"].min()),
        "date_max":     str(df["invoice_date"].max()),
        "amount_min":   float(df["amount"].min()),
        "amount_max":   float(df["amount"].max()),
        "amount_mean":  float(df["amount"].mean()),
        "amount_median":float(df["amount"].median()),
        "gl_accounts":  ", ".join(sorted(str(x) for x in df["gl_account"].unique())),
        "roles":        ", ".join(sorted(str(x) for x in df["approver_role"].unique())),
    }

    plan_event = ledger.append_event(Event(
        run_id=run_id,
        step_type="PLAN",
        actor="PLANNER",
        model=PLANNER_MODEL,
        input_summary=dataset_info,
        output_summary={},
        reasoning="Requesting risk-prioritized plan from Gemini flash.",
        parent_event_id=parent_event_id,
    ))

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    # AQ.* keys are OAuth bearer tokens — not usable as API keys.
    # Fall back to Application Default Credentials (gcloud auth application-default login).
    _use_adc = bool(api_key) and not api_key.startswith("AIza")
    if not api_key:
        ledger.append_event(Event(
            run_id=run_id,
            step_type="PLAN",
            actor="PLANNER",
            model="default",
            input_summary={},
            output_summary={"fallback": True, "reason": "GEMINI_API_KEY not set"},
            reasoning="No API key — using default plan.",
            parent_event_id=plan_event.event_id,
        ))
        return _default_plan(plan_event.event_id)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client() if _use_adc else genai.Client(api_key=api_key)
        prompt = _USER_TEMPLATE.format(checks_list=checks_list, **dataset_info)

        response = _retry_generate(
            client, PLANNER_MODEL, prompt,
            types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                response_mime_type="application/json",
            ),
        )

        plan_json = _extract_json(response.text)

        raw_checks = plan_json.get("checks", [])
        checks = [
            CheckPlan(
                check_id=str(c["check_id"]),
                priority=int(c.get("priority", i + 1)),
                rationale=str(c.get("rationale", "")),
            )
            for i, c in enumerate(raw_checks)
        ]
        # Ensure all four checks are present (model may omit some)
        present = {c.check_id for c in checks}
        for i, cid in enumerate(AVAILABLE_CHECKS):
            if cid not in present:
                checks.append(CheckPlan(cid, len(checks) + 1, "Added by fallback — not in model plan"))

        usage = response.usage_metadata
        tok_in  = getattr(usage, "prompt_token_count", 0) or 0
        tok_out = getattr(usage, "candidates_token_count", 0) or 0
        cost    = _calc_cost(PLANNER_MODEL, tok_in, tok_out)

        ledger.append_event(Event(
            run_id=run_id,
            step_type="PLAN",
            actor="PLANNER",
            model=PLANNER_MODEL,
            input_summary={},
            output_summary={
                "checks": [c.check_id for c in checks],
                "risk_narrative": plan_json.get("risk_narrative", ""),
            },
            reasoning=plan_json.get("reasoning", ""),
            tokens_in=tok_in,
            tokens_out=tok_out,
            cost_usd=cost,
            parent_event_id=plan_event.event_id,
        ))

        return RunPlan(
            reasoning=plan_json.get("reasoning", ""),
            risk_narrative=plan_json.get("risk_narrative", ""),
            checks=sorted(checks, key=lambda c: c.priority),
            plan_event_id=plan_event.event_id,
            model=PLANNER_MODEL,
            tokens_in=tok_in,
            tokens_out=tok_out,
            cost_usd=cost,
        )

    except Exception as exc:
        ledger.append_event(Event(
            run_id=run_id,
            step_type="PLAN",
            actor="PLANNER",
            model=PLANNER_MODEL,
            input_summary={},
            output_summary={"fallback": True, "error": str(exc)},
            reasoning=f"Gemini call failed ({type(exc).__name__}); using default plan.",
            parent_event_id=plan_event.event_id,
        ))
        return _default_plan(plan_event.event_id)
