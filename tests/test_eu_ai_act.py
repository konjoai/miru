"""Tests for the EU AI Act compliance report generator."""
from __future__ import annotations

from datetime import datetime

import pytest

from miru.eu_ai_act import COMPLIANCE_DEADLINE, REPORT_VERSION, generate_report


def _full_record() -> dict:
    """A complete recorded analysis dict with all Article fields populated."""
    return {
        "analysis_id": "abc12345-...-uuid",
        "ts": "2026-05-12T10:00:00+00:00",
        "question": "What is in the image?",
        "image_sha256": "0" * 64,
        "trace": {
            "answer": "A circular bright object.",
            "confidence": 0.83,
            "backend": "mock",
            "method": "attention",
            "latency_ms": 0.5,
            "fidelity": {
                "fidelity_score": 0.72,
                "baseline_confidence": 0.83,
                "masked_confidence": 0.23,
                "k_pct": 0.1,
                "low_fidelity": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_report_envelope_has_all_three_articles() -> None:
    r = generate_report(_full_record())
    assert "article_11" in r
    assert "article_13" in r
    assert "article_15" in r
    assert r["regulation"].startswith("Regulation (EU) 2024/1689")
    assert r["report_version"] == REPORT_VERSION
    assert r["compliance_deadline"] == COMPLIANCE_DEADLINE
    # report_generated_utc must parse as ISO-8601.
    datetime.fromisoformat(r["report_generated_utc"])


# ---------------------------------------------------------------------------
# Article 11 — technical documentation
# ---------------------------------------------------------------------------


def test_article_11_carries_model_and_data_flow() -> None:
    r = generate_report(_full_record(), system_name="X", provider="Y")
    a11 = r["article_11"]
    assert a11["system_name"] == "X"
    assert a11["provider"] == "Y"
    assert a11["model"] == "mock"
    assert a11["method"] == "attention"
    assert a11["input_artefact_sha256"] == "0" * 64
    assert "backend.infer" in a11["data_flow"]


def test_article_11_incomplete_when_fields_missing() -> None:
    record = _full_record()
    record["trace"]["backend"] = ""
    record["image_sha256"] = None
    r = generate_report(record)
    status = r["compliance_status"]["article_11"]
    assert status["status"] == "incomplete"
    assert "model" in status["missing_fields"]
    assert "input_artefact_sha256" in status["missing_fields"]


# ---------------------------------------------------------------------------
# Article 13 — transparency
# ---------------------------------------------------------------------------


def test_article_13_includes_user_facing_outputs() -> None:
    r = generate_report(_full_record())
    a13 = r["article_13"]
    assert a13["user_facing_question"] == "What is in the image?"
    assert a13["user_facing_answer"] == "A circular bright object."
    assert a13["model_confidence"] == pytest.approx(0.83)
    assert a13["fidelity_score"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# Article 15 — accuracy / robustness / risks
# ---------------------------------------------------------------------------


def test_article_15_flags_low_fidelity() -> None:
    record = _full_record()
    record["trace"]["fidelity"]["fidelity_score"] = 0.3
    r = generate_report(record)
    a15 = r["article_15"]
    assert a15["fidelity"]["status"] == "warning"
    assert any("low_fidelity" in risk for risk in a15["detected_risks"])


def test_article_15_flags_low_confidence() -> None:
    record = _full_record()
    record["trace"]["confidence"] = 0.31
    r = generate_report(record)
    risks = r["article_15"]["detected_risks"]
    assert any("low_confidence" in risk for risk in risks)


def test_article_15_flags_method_disagreement() -> None:
    record = _full_record()
    record["trace"]["consensus"] = {"consensus_score": 0.15}
    r = generate_report(record)
    risks = r["article_15"]["detected_risks"]
    assert any("method_disagreement" in risk for risk in risks)


def test_article_15_no_risks_on_clean_record() -> None:
    r = generate_report(_full_record())
    assert r["article_15"]["detected_risks"] == []


# ---------------------------------------------------------------------------
# Defensive — minimal / malformed records
# ---------------------------------------------------------------------------


def test_generate_report_tolerates_minimal_record() -> None:
    minimal = {
        "analysis_id": "x" * 8,
        "ts": "2026-05-12T00:00:00+00:00",
        "question": "",
        "image_sha256": None,
        "trace": {},
    }
    r = generate_report(minimal)
    # Doesn't raise; produces a structurally valid envelope.
    assert "article_11" in r
    assert r["compliance_status"]["article_11"]["status"] == "incomplete"
