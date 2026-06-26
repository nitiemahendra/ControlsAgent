"""
Severity classifier: uses a strong model to write narrative rationale and
recommended_action for each finding from the deterministic check executors.

Design rule: the classifier DOES NOT change severity — that stays deterministic
and defensible. It only enriches the narrative text for the workpaper.

Model routing:
  - Planner (triage)   → gemini-2.0-flash  (cheap, fast)
  - Checks             → deterministic Python (zero tokens)
  - Classifier (prose) → gemini-2.5-pro  (strong, slow, expensive)
"""
from __future__ import annotations

import json
import os
import time

from agent.ledger import Ledger
from agent.models import Event, Finding

CLASSIFIER_MODEL = "gemini-2.5-flash"
_FALLBACK_MODEL  = "gemini-2.5-flash"   # same model; kept for compatibility
_PRICE = {
    "gemini-2.5-flash": {"in": 0.30e-6,  "out": 2.50e-6},
    "gemini-2.5-pro":   {"in": 1.25e-6,  "out": 10.00e-6},
    "gemini-2.0-flash": {"in": 0.075e-6, "out": 0.30e-6},
}

_SYSTEM = """\
You are a senior AP auditor writing findings for an internal controls workpaper.
For each finding supplied, write:
1. severity_rationale — 2–4 sentences explaining the specific risk using the exact
   figures in evidence. Cite control objectives and materiality thresholds.
2. recommended_action — 3–5 numbered steps for the reviewer.

IMPORTANT: Do NOT change the severity level — it is already determined by
deterministic analytics that will be cited in the workpaper.
Return ONLY valid JSON, no commentary."""

_USER_TEMPLATE = """\
Control: {control_id}
Objective: {objective}

{n} finding(s) to narrate:
{findings_json}

Return a JSON array, one object per finding:
[
  {{
    "finding_id": "<exact id>",
    "severity_rationale": "<2-4 sentences, exact figures, cite materiality>",
    "recommended_action": "<numbered steps, 3-5 items>"
  }}
]"""


def enrich_findings(
    findings: list[Finding],
    check_id: str,
    run_id: str,
    ledger: Ledger,
    parent_event_id: str | None = None,
) -> list[Finding]:
    """Use CLASSIFIER_MODEL to write narrative for a batch of findings from one check.

    Falls back silently (keeps check-generated text) if no API key or call fails.
    """
    if not findings:
        return findings

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    # AQ.* keys are OAuth bearer tokens — not usable as API keys.
    # Fall back to Application Default Credentials (gcloud auth application-default login).
    _use_adc = bool(api_key) and not api_key.startswith("AIza")
    if not api_key and not _use_adc:
        return findings  # no credentials at all — skip silently

    classify_event = ledger.append_event(Event(
        run_id=run_id,
        step_type="CLASSIFY",
        actor="CLASSIFIER",
        model=CLASSIFIER_MODEL,
        input_summary={"check_id": check_id, "n_findings": len(findings)},
        output_summary={},
        reasoning=f"Requesting {CLASSIFIER_MODEL} narrative for {len(findings)} finding(s).",
        parent_event_id=parent_event_id,
    ))

    try:
        from google import genai
        from google.genai import types

        client = genai.Client() if _use_adc else genai.Client(api_key=api_key)

        findings_data = [
            {
                "finding_id": f.finding_id,
                "entity_ref": f.entity_ref,
                "severity": f.severity,
                "evidence": f.evidence,
                "statistic": f.statistic,
            }
            for f in findings
        ]

        prompt = _USER_TEMPLATE.format(
            control_id=check_id,
            objective=findings[0].control_objective,
            n=len(findings),
            findings_json=json.dumps(findings_data, indent=2),
        )

        cfg = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            response_mime_type="application/json",
        )
        # Try strong model first; fall back to flash on quota exhaustion.
        model_used = CLASSIFIER_MODEL
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_used, contents=prompt, config=cfg
                )
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg and attempt < 2:
                    if model_used == CLASSIFIER_MODEL:
                        model_used = _FALLBACK_MODEL   # degrade to flash on quota
                    time.sleep(4 ** attempt)
                    continue
                raise

        narratives: list[dict] = json.loads(response.text)
        nmap = {n["finding_id"]: n for n in narratives}
        for f in findings:
            if f.finding_id in nmap:
                entry = nmap[f.finding_id]
                rationale = entry.get("severity_rationale", f.severity_rationale)
                action    = entry.get("recommended_action",  f.recommended_action)
                # Model sometimes returns a list of steps — join to a single string
                if isinstance(rationale, list):
                    rationale = " ".join(str(s) for s in rationale)
                if isinstance(action, list):
                    action = "\n".join(f"{i+1}. {s}" for i, s in enumerate(action))
                f.severity_rationale = str(rationale)
                f.recommended_action = str(action)

        usage   = response.usage_metadata
        tok_in  = getattr(usage, "prompt_token_count",     0) or 0
        tok_out = getattr(usage, "candidates_token_count", 0) or 0
        cost = tok_in * _PRICE[model_used]["in"] + tok_out * _PRICE[model_used]["out"]

        ledger.append_event(Event(
            run_id=run_id,
            step_type="CLASSIFY",
            actor="CLASSIFIER",
            model=model_used,
            input_summary={},
            output_summary={
                "narratives_written": len(narratives),
                "model_used": model_used,
                "degraded": model_used != CLASSIFIER_MODEL,
            },
            reasoning=f"Narrated {len(narratives)} finding(s) for {check_id} via {model_used}.",
            tokens_in=tok_in,
            tokens_out=tok_out,
            cost_usd=cost,
            parent_event_id=classify_event.event_id,
        ))

    except Exception as exc:
        ledger.append_event(Event(
            run_id=run_id,
            step_type="CLASSIFY",
            actor="CLASSIFIER",
            model=CLASSIFIER_MODEL,
            input_summary={},
            output_summary={"error": str(exc)[:200], "kept_original": True},
            reasoning=f"Classifier failed ({type(exc).__name__}); keeping check-generated rationale.",
            parent_event_id=classify_event.event_id,
        ))

    return findings
