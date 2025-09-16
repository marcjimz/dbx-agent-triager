# pip install rapidfuzz
from typing import Any, Optional
import json
from rapidfuzz import fuzz
import mlflow
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, Expectation, AssessmentError

name = "custom_fuzzy_evaluator"
THRESHOLD = 75

def _extract_payload(outputs: Any) -> Optional[dict]:
    """
    Accepts either a JSON string or a dict.
    Handles both:
      { "incident_number": ..., "short_description": ..., "recommended_assignment_group": ..., ... }
    and
      { "message": { ...same keys... } }
    """
    if outputs is None:
        return None
    if isinstance(outputs, str):
        try:
            obj = json.loads(outputs)
        except Exception:
            return None
    elif isinstance(outputs, dict):
        obj = outputs
    else:
        return None

    if isinstance(obj.get("message"), dict):
        return obj["message"]
    return obj

@scorer(name=name)
def fuzzy_evaluator(outputs: Any, inputs: dict[str, Any]) -> Feedback:
    """
    Returns True if fuzzy ratio >= 75%, else False.
    Includes a rationale summarizing the comparison.
    """
    payload = _extract_payload(outputs)
    actual = None
    if isinstance(inputs, dict):
        actual = inputs.get("assignment_group")

    if not payload or "recommended_assignment_group" not in payload:
        error = AssessmentError(
            error_code="INCORRECT_PAYLOAD",
            error_message="Missing payload proper formatting or recommended_assignment_group column: %s" % payload,
        )
        return Feedback(
            name="fuzzy_similarity",
            value=None,
            rationale="Could not parse 'recommended_assignment_group' from model output JSON.",
            error=error
        )

    if not actual:
        error = AssessmentError(
            error_code="INCORRECT_PAYLOAD",
            error_message="Missing payload proper formatting or assignment_group column: %s" % payload,
        )
        return Feedback(
            name="fuzzy_similarity",
            value=None,
            rationale="Missing 'assignment_group' in inputs for this row.",
            error=error
        )

    rec = payload.get("recommended_assignment_group") or ""
    ratio = fuzz.ratio(str(actual).strip().lower(), str(rec).strip().lower())
    passed = ratio >= THRESHOLD  # boolean condition

    incident = payload.get("incident_number", "n/a")
    sd = payload.get("short_description", "n/a")
    reason = payload.get("reason", "n/a")

    rationale = (
        f"Incident {incident}: '{sd}'. "
        f"Model recommended '{rec}' vs actual '{actual}'. "
        f"Reason: {reason}. Confidence (fuzzy ratio)={ratio}%. "
        f"Pass? {passed}"
    )

    return Feedback(
        name="fuzzy_similarity",
        value=passed,                         # True/False for pass/fail
        rationale=rationale,                  # Human-readable explanation
        metadata={
            "ratio": str(ratio),
            "threshold": str(THRESHOLD),
            "actual": actual,
            "recommended": rec,
            "incident_number": str(payload.get("incident_number", "")),
        },
    )