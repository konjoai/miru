"""EU AI Act compliance report generator.

The AI Act (Regulation (EU) 2024/1689) imposes documentation,
transparency, record-keeping, and robustness obligations on providers and
deployers of high-risk AI systems.  Miru's recorded analyses already
capture most of the required fields.  This module assembles them into a
regulator-ready report.

Coverage
--------

- **Article 11 — Technical documentation.** Identifies the system,
  its providers, the model behind it, and the data flow that produced
  a given analysis.  Refers to the recorded JSONL record as the
  evidentiary source.
- **Article 12 — Record-keeping (logging).** Documents that the analysis
  is captured as an immutable, privacy-stripped log entry, traceable and
  reproducible from its ``analysis_id``.
- **Article 13 — Transparency & information to deployers.** Surfaces
  the answer returned to the user, the confidence the model reported,
  the explanation method used, the fidelity score (when available), and
  the documented feature importance (the ranked salient regions) — i.e.
  the information a deployer needs to evaluate the output.
- **Article 15 — Accuracy, robustness, cybersecurity.** Includes any
  benchmark scores attached to the record, the fidelity test outcome,
  the cross-modal synergy probe, and a list of detected risks ("low
  fidelity", "low confidence", "method disagreement", "visual-only
  salience") so an auditor can spot weak explanations quickly.
- **Article 86 — Right to explanation of individual decision-making.**
  A plain-language, person-facing rationale citing the single most
  influential image region, plus a contestability note.

This module is deliberately *fact-only* — it does not assert compliance,
it just structures the facts.  A human compliance officer signs off.

References:
    Regulation (EU) 2024/1689.  Article texts:
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

REPORT_VERSION = "1.1"
COMPLIANCE_DEADLINE = "2026-08-02"  # high-risk obligations apply from this date.

LOW_FIDELITY_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.5
LOW_CONSENSUS_THRESHOLD = 0.3
LOW_SYNERGY_THRESHOLD = 0.3
MAX_FEATURE_IMPORTANCE = 5


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
        Nested dict with ``article_11``, ``article_12``, ``article_13``,
        ``article_15``, ``article_86`` and ``compliance_status`` blocks
        plus envelope metadata.
    """
    facts = _extract_facts(record)

    article_11 = _build_article_11(facts, system_name, provider, use_case_category)
    article_12 = _build_article_12(facts)
    article_13 = _build_article_13(facts)
    article_15 = _build_article_15(facts)
    article_86 = _build_article_86(facts)

    return {
        "report_version": REPORT_VERSION,
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "compliance_deadline": COMPLIANCE_DEADLINE,
        "regulation": "Regulation (EU) 2024/1689 (AI Act)",
        "article_11": article_11,
        "article_12": article_12,
        "article_13": article_13,
        "article_15": article_15,
        "article_86": article_86,
        "compliance_status": _compliance_status(
            article_11, article_12, article_13, article_15, article_86
        ),
    }


def _extract_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten the fields the article builders need into one dict."""
    trace = record.get("trace", {}) or {}
    fidelity = trace.get("fidelity") or trace.get("fidelity_score")
    fidelity_score = _safe_float(
        fidelity.get("fidelity_score") if isinstance(fidelity, dict) else fidelity
    )
    synergy = trace.get("synergy") if isinstance(trace.get("synergy"), dict) else None
    return {
        "analysis_id": record.get("analysis_id", ""),
        "image_sha256": record.get("image_sha256"),
        "question": record.get("question", ""),
        "timestamp": record.get("ts", ""),
        "answer": trace.get("answer", ""),
        "confidence": _safe_float(trace.get("confidence")),
        "backend": trace.get("backend", ""),
        "method": trace.get("method", trace.get("explanation_method", "")),
        "fidelity_score": fidelity_score,
        "synergy": synergy,
        "top_regions": trace.get("top_regions") or [],
        "benchmark": trace.get("benchmark") or {},
        "consensus": trace.get("consensus") or {},
    }


def _build_article_11(
    facts: dict[str, Any],
    system_name: str,
    provider: str,
    use_case_category: str,
) -> dict[str, Any]:
    return {
        "title": "Technical documentation",
        "system_name": system_name,
        "provider": provider,
        "use_case_category": use_case_category,
        "analysis_id": facts["analysis_id"],
        "timestamp": facts["timestamp"],
        "model": facts["backend"] or "unknown",
        "method": facts["method"] or "unknown",
        "input_artefact_sha256": facts["image_sha256"],
        "data_flow": (
            "user → image_b64 → backend.infer → ReasoningTracer → "
            "explainer → recorder (privacy-stripped JSONL)"
        ),
        "evidentiary_record": "see recorded JSONL by analysis_id",
    }


def _build_article_12(facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Record-keeping (logging)",
        "analysis_id": facts["analysis_id"],
        "timestamp": facts["timestamp"],
        "log_record_format": (
            "privacy-stripped JSONL; raw image bytes reduced to a SHA-256 "
            "digest before persistence"
        ),
        "log_retention": (
            "governed by the deployer's recorder store (MIRU_RECORD_PATH)"
        ),
        "traceability": (
            "fully reproducible from analysis_id via "
            "GET /analysis/{id}/export?format=json"
        ),
    }


def _build_article_13(facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Transparency and information to deployers",
        "user_facing_question": facts["question"],
        "user_facing_answer": facts["answer"],
        "model_confidence": facts["confidence"],
        "explanation_method": facts["method"] or "unknown",
        "fidelity_score": facts["fidelity_score"],
        "feature_importance": _feature_importance(facts["top_regions"]),
        "intended_use_disclosure": (
            "Saliency-map explanations are *approximations* of the "
            "model's decision; consult fidelity_score and method "
            "consensus before relying on a specific region."
        ),
    }


def _build_article_15(facts: dict[str, Any]) -> dict[str, Any]:
    fidelity_score = facts["fidelity_score"]
    return {
        "title": "Accuracy, robustness and cybersecurity",
        "benchmark": facts["benchmark"] or {"status": "not_attached"},
        "fidelity": {
            "score": fidelity_score,
            "method": "deletion test (Petsiuk et al. 2018)",
            "status": (
                "ok"
                if fidelity_score is None or fidelity_score >= LOW_FIDELITY_THRESHOLD
                else "warning"
            ),
        },
        "synergy": facts["synergy"] or {"status": "not_attached"},
        "consensus": facts["consensus"] or {"status": "not_attached"},
        "detected_risks": _detect_risks(facts),
    }


def _build_article_86(facts: dict[str, Any]) -> dict[str, Any]:
    region = _top_region(facts["top_regions"])
    confidence = facts["confidence"]
    conf_str = f"{confidence:.0%}" if confidence is not None else "an unreported"
    if region is not None:
        region_clause = (
            f" The decision was most influenced by the image region at grid "
            f"cell (row {region['row']}, col {region['col']})."
        )
    else:
        region_clause = " No per-region influence was recorded for this analysis."
    question = facts["question"] or "(no question recorded)"
    answer = facts["answer"] or "(no answer recorded)"
    return {
        "title": "Right to explanation of an individual decision",
        "plain_language_explanation": (
            f"The system was asked: '{question}'. It responded: '{answer}' "
            f"with {conf_str} confidence.{region_clause} This explanation is "
            "an approximation of the model's internal reasoning, not a "
            "guarantee of it."
        ),
        "most_influential_region": region,
        "model_confidence": confidence,
        "contestability_note": (
            "An affected person may request human review of this decision; "
            "the saliency explanation does not replace that right."
        ),
    }


def _detect_risks(facts: dict[str, Any]) -> list[str]:
    """Collect plain-language risk flags an auditor should see first."""
    risks: list[str] = []
    fidelity_score = facts["fidelity_score"]
    if fidelity_score is not None and fidelity_score < LOW_FIDELITY_THRESHOLD:
        risks.append(
            f"low_fidelity: fidelity_score={fidelity_score:.3f} < "
            f"{LOW_FIDELITY_THRESHOLD}; explanation may not reflect model reasoning"
        )
    confidence = facts["confidence"]
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        risks.append(
            f"low_confidence: confidence={confidence:.3f} < {LOW_CONFIDENCE_THRESHOLD}"
        )
    consensus_score = _safe_float(facts["consensus"].get("consensus_score"))
    if consensus_score is not None and consensus_score < LOW_CONSENSUS_THRESHOLD:
        risks.append(
            f"method_disagreement: consensus_score={consensus_score:.3f} "
            f"< {LOW_CONSENSUS_THRESHOLD} across explanation methods"
        )
    risks.extend(_synergy_risk(facts["synergy"]))
    return risks


def _synergy_risk(synergy: dict[str, Any] | None) -> list[str]:
    """Flag visual-only salience when the synergy probe ran and scored low."""
    if not synergy:
        return []
    score = _safe_float(synergy.get("synergy_score"))
    low = bool(synergy.get("low_synergy")) or (
        score is not None and score < LOW_SYNERGY_THRESHOLD
    )
    if not low:
        return []
    score_str = f"{score:.3f}" if score is not None else "unknown"
    return [
        f"visual_only_salience: synergy_score={score_str} < {LOW_SYNERGY_THRESHOLD}; "
        "saliency may track visual salience rather than cross-modal reasoning"
    ]


def _feature_importance(top_regions: list[Any]) -> list[dict[str, Any]]:
    """Document the ranked salient regions as feature-importance evidence."""
    ranked: list[dict[str, Any]] = []
    for rank, region in enumerate(top_regions[:MAX_FEATURE_IMPORTANCE], start=1):
        if not isinstance(region, dict):
            continue
        ranked.append(
            {
                "rank": rank,
                "row": region.get("row"),
                "col": region.get("col"),
                "score": _safe_float(region.get("score")),
            }
        )
    return ranked


def _top_region(top_regions: list[Any]) -> dict[str, Any] | None:
    """Return the single highest-scoring region, or None when absent."""
    fi = _feature_importance(top_regions)
    return fi[0] if fi else None


def _compliance_status(
    a11: dict[str, Any],
    a12: dict[str, Any],
    a13: dict[str, Any],
    a15: dict[str, Any],
    a86: dict[str, Any],
) -> dict[str, Any]:
    """Per-article completeness summary.

    Honest: ``ok`` if all listed required fields are present and
    non-empty, ``incomplete`` otherwise with the list of missing
    fields.  This is *completeness*, not legal compliance — only a
    human auditor can sign off on the latter.
    """
    return {
        "article_11": _check_block(
            a11,
            (
                "system_name",
                "provider",
                "use_case_category",
                "analysis_id",
                "timestamp",
                "model",
                "method",
                "input_artefact_sha256",
            ),
        ),
        "article_12": _check_block(a12, ("analysis_id", "timestamp")),
        "article_13": _check_block(
            a13,
            (
                "user_facing_question",
                "user_facing_answer",
                "model_confidence",
                "explanation_method",
            ),
        ),
        "article_15": _check_block(a15, ("benchmark", "fidelity")),
        "article_86": _check_block(a86, ("plain_language_explanation",)),
        "note": (
            "Completeness check only; final legal compliance requires human sign-off."
        ),
    }


def _check_block(block: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    missing = [k for k in required if block.get(k) in (None, "", "unknown")]
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
