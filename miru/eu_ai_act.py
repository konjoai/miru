"""EU AI Act compliance report generator.

The AI Act (Regulation (EU) 2024/1689) imposes documentation,
transparency, and robustness obligations on providers of high-risk AI
systems.  Miru's recorded analyses already capture most of the required
fields.  This module assembles them into a regulator-ready report.

Coverage
--------

- **Article 11 — Technical documentation.** Identifies the system,
  its providers, the model behind it, and the data flow that produced
  a given analysis.  Refers to the recorded JSONL record as the
  evidentiary source.
- **Article 13 — Transparency & information to deployers.** Surfaces
  the answer returned to the user, the confidence the model reported,
  the explanation method used, and the fidelity score (when available)
  — i.e. the information a deployer needs to evaluate the output.
- **Article 15 — Accuracy, robustness, cybersecurity.** Includes any
  benchmark scores attached to the record, the fidelity test outcome,
  and a list of detected risks ("low fidelity", "low confidence",
  "method disagreement") so an auditor can spot weak explanations
  quickly.

This module is deliberately *fact-only* — it does not assert compliance,
it just structures the facts.  A human compliance officer signs off.

References:
    Regulation (EU) 2024/1689.  Article texts:
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

REPORT_VERSION = "1.0"
COMPLIANCE_DEADLINE = "2026-08-02"  # high-risk obligations apply from this date.


def generate_report(
    record: dict[str, Any],
    *,
    system_name: str = "Miru explainability surface",
    provider: str = "Miru — see /health for runtime version",
    use_case_category: str = "general-purpose vision-language inference",
) -> dict[str, Any]:
    """Build a structured EU AI Act report from a recorded analysis.

    Args:
        record: A recorded JSONL line as produced by
            :func:`miru.recorder.build_record` — must contain at least
            ``analysis_id``, ``ts``, ``question``, ``image_sha256`` and
            ``trace``.
        system_name: Free-text identifier of the AI system.
        provider: Free-text provider identifier.
        use_case_category: High-risk category (e.g. "biometric
            identification", "critical infrastructure" — see Annex III
            of the Act).

    Returns:
        Nested dict with ``article_11``, ``article_13``, ``article_15``,
        ``compliance_status`` blocks plus envelope metadata.
    """
    trace = record.get("trace", {}) or {}
    analysis_id = record.get("analysis_id", "")
    image_sha256 = record.get("image_sha256")
    question = record.get("question", "")
    timestamp = record.get("ts", "")

    answer = trace.get("answer", "")
    confidence = _safe_float(trace.get("confidence"))
    backend = trace.get("backend", "")
    method = trace.get("method", trace.get("explanation_method", ""))
    fidelity = trace.get("fidelity") or trace.get("fidelity_score")
    fidelity_score = _safe_float(fidelity.get("fidelity_score") if isinstance(fidelity, dict) else fidelity)
    benchmark = trace.get("benchmark") or {}
    consensus = trace.get("consensus") or {}

    article_11 = {
        "title": "Technical documentation",
        "system_name": system_name,
        "provider": provider,
        "use_case_category": use_case_category,
        "analysis_id": analysis_id,
        "timestamp": timestamp,
        "model": backend or "unknown",
        "method": method or "unknown",
        "input_artefact_sha256": image_sha256,
        "data_flow": (
            "user → image_b64 → backend.infer → ReasoningTracer → "
            "explainer → recorder (privacy-stripped JSONL)"
        ),
        "evidentiary_record": "see recorded JSONL by analysis_id",
    }

    article_13 = {
        "title": "Transparency and information to deployers",
        "user_facing_question": question,
        "user_facing_answer": answer,
        "model_confidence": confidence,
        "explanation_method": method or "unknown",
        "fidelity_score": fidelity_score,
        "intended_use_disclosure": (
            "Saliency-map explanations are *approximations* of the "
            "model's decision; consult fidelity_score and method "
            "consensus before relying on a specific region."
        ),
    }

    risks: list[str] = []
    if fidelity_score is not None and fidelity_score < 0.5:
        risks.append(
            f"low_fidelity: fidelity_score={fidelity_score:.3f} < 0.5; "
            "explanation may not reflect model reasoning"
        )
    if confidence is not None and confidence < 0.5:
        risks.append(
            f"low_confidence: confidence={confidence:.3f} < 0.5"
        )
    consensus_score = _safe_float(consensus.get("consensus_score"))
    if consensus_score is not None and consensus_score < 0.3:
        risks.append(
            f"method_disagreement: consensus_score={consensus_score:.3f} "
            "< 0.3 across explanation methods"
        )

    article_15 = {
        "title": "Accuracy, robustness and cybersecurity",
        "benchmark": benchmark or {"status": "not_attached"},
        "fidelity": {
            "score": fidelity_score,
            "method": "deletion test (Petsiuk et al. 2018)",
            "status": (
                "ok" if fidelity_score is None or fidelity_score >= 0.5
                else "warning"
            ),
        },
        "consensus": consensus or {"status": "not_attached"},
        "detected_risks": risks,
    }

    compliance_status = _compliance_status(article_11, article_13, article_15)

    return {
        "report_version": REPORT_VERSION,
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "compliance_deadline": COMPLIANCE_DEADLINE,
        "regulation": "Regulation (EU) 2024/1689 (AI Act)",
        "article_11": article_11,
        "article_13": article_13,
        "article_15": article_15,
        "compliance_status": compliance_status,
    }


def _compliance_status(
    a11: dict[str, Any], a13: dict[str, Any], a15: dict[str, Any]
) -> dict[str, Any]:
    """Per-article completeness summary.

    Honest: ``ok`` if all listed required fields are present and
    non-empty, ``incomplete`` otherwise with the list of missing
    fields.  This is *completeness*, not legal compliance — only a
    human auditor can sign off on the latter.
    """
    required_11 = (
        "system_name", "provider", "use_case_category", "analysis_id",
        "timestamp", "model", "method", "input_artefact_sha256",
    )
    required_13 = (
        "user_facing_question", "user_facing_answer", "model_confidence",
        "explanation_method",
    )
    required_15 = ("benchmark", "fidelity")

    return {
        "article_11": _check_block(a11, required_11),
        "article_13": _check_block(a13, required_13),
        "article_15": _check_block(a15, required_15),
        "note": (
            "Completeness check only; final legal compliance requires "
            "human sign-off."
        ),
    }


def _check_block(block: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    missing = [
        k for k in required
        if block.get(k) in (None, "", "unknown")
    ]
    return {
        "status": "ok" if not missing else "incomplete",
        "missing_fields": missing,
    }


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = [
    "REPORT_VERSION",
    "COMPLIANCE_DEADLINE",
    "generate_report",
]
