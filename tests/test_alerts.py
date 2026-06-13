"""Unit tests for miru.alerts — rule store, evaluation logic, webhook helpers."""

from __future__ import annotations

import pytest

from miru.alerts import (
    AlertStore,
    SUPPORTED_FIELDS,
    SUPPORTED_OPS,
    validate_webhook_url,
    _evaluate_op,
    _new_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> AlertStore:
    return AlertStore(str(tmp_path / "test_alerts.db"))


_WEBHOOK = "http://example.com/hook"


# ---------------------------------------------------------------------------
# validate_webhook_url
# ---------------------------------------------------------------------------


def test_validate_webhook_url_http_ok() -> None:
    validate_webhook_url("http://example.com/hook")


def test_validate_webhook_url_https_ok() -> None:
    validate_webhook_url("https://example.com/hook?token=abc")


def test_validate_webhook_url_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_webhook_url("")


def test_validate_webhook_url_rejects_non_http() -> None:
    with pytest.raises(ValueError, match="http"):
        validate_webhook_url("ftp://example.com/hook")


def test_validate_webhook_url_rejects_too_long() -> None:
    with pytest.raises(ValueError, match="length"):
        validate_webhook_url("http://example.com/" + "a" * 2048)


# ---------------------------------------------------------------------------
# _evaluate_op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,op,threshold,expected",
    [
        (0.3, "<", 0.4, True),
        (0.4, "<", 0.4, False),
        (0.4, "<=", 0.4, True),
        (0.5, ">", 0.4, True),
        (0.4, ">=", 0.4, True),
        (0.4, "==", 0.4, True),
        (0.3, "==", 0.4, False),
    ],
)
def test_evaluate_op(value: float, op: str, threshold: float, expected: bool) -> None:
    assert _evaluate_op(value, op, threshold) == expected


def test_evaluate_op_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported op"):
        _evaluate_op(0.5, "!=", 0.4)


# ---------------------------------------------------------------------------
# Rule CRUD
# ---------------------------------------------------------------------------


def test_create_rule_ok(store: AlertStore) -> None:
    rule = store.create_rule("low-conf", "confidence", "<", 0.4, _WEBHOOK)
    assert rule.rule_id.startswith("rule_")
    assert rule.name == "low-conf"
    assert rule.field == "confidence"
    assert rule.op == "<"
    assert rule.threshold == pytest.approx(0.4)
    assert rule.enabled is True


def test_create_rule_rejects_unknown_field(store: AlertStore) -> None:
    with pytest.raises(ValueError, match="field"):
        store.create_rule("bad", "latency_ms", "<", 100.0, _WEBHOOK)


def test_create_rule_rejects_unknown_op(store: AlertStore) -> None:
    with pytest.raises(ValueError, match="op"):
        store.create_rule("bad", "confidence", "!=", 0.4, _WEBHOOK)


def test_create_rule_rejects_bad_webhook(store: AlertStore) -> None:
    with pytest.raises(ValueError, match="http"):
        store.create_rule("bad", "confidence", "<", 0.4, "ftp://example.com")


def test_list_rules_empty(store: AlertStore) -> None:
    assert store.list_rules() == []


def test_list_rules_after_create(store: AlertStore) -> None:
    store.create_rule("r1", "confidence", "<", 0.4, _WEBHOOK)
    store.create_rule("r2", "fidelity_score", "<", 0.5, _WEBHOOK)
    rules = store.list_rules()
    assert len(rules) == 2
    assert {r.name for r in rules} == {"r1", "r2"}


def test_delete_rule_ok(store: AlertStore) -> None:
    rule = store.create_rule("to-delete", "confidence", ">", 0.9, _WEBHOOK)
    assert store.delete_rule(rule.rule_id) is True
    assert store.list_rules() == []


def test_delete_rule_missing(store: AlertStore) -> None:
    assert store.delete_rule("rule_doesnotexist") is False


def test_set_rule_enabled(store: AlertStore) -> None:
    rule = store.create_rule("toggle", "confidence", "<", 0.4, _WEBHOOK)
    store.set_rule_enabled(rule.rule_id, enabled=False)
    rules = store.list_rules(enabled_only=True)
    assert rules == []
    all_rules = store.list_rules()
    assert len(all_rules) == 1
    assert all_rules[0].enabled is False


def test_rule_limit_enforced(store: AlertStore) -> None:
    from miru.alerts import MAX_RULES

    for i in range(MAX_RULES):
        store.create_rule(f"rule-{i}", "confidence", "<", 0.4, _WEBHOOK)
    with pytest.raises(ValueError, match="limit"):
        store.create_rule("one-too-many", "confidence", "<", 0.4, _WEBHOOK)


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------


def test_evaluate_fires_on_matching_rule(store: AlertStore) -> None:
    store.create_rule("low-conf", "confidence", "<", 0.4, _WEBHOOK)
    fired = store.evaluate("analysis_abc", {"confidence": 0.3})
    assert len(fired) == 1
    assert fired[0].field == "confidence"
    assert fired[0].fired_value == pytest.approx(0.3)


def test_evaluate_no_fire_when_condition_false(store: AlertStore) -> None:
    store.create_rule("high-conf", "confidence", ">", 0.9, _WEBHOOK)
    fired = store.evaluate("analysis_abc", {"confidence": 0.5})
    assert fired == []


def test_evaluate_skips_absent_field(store: AlertStore) -> None:
    store.create_rule("fid-rule", "fidelity_score", "<", 0.5, _WEBHOOK)
    fired = store.evaluate("analysis_abc", {"confidence": 0.9})
    assert fired == []


def test_evaluate_fidelity_from_nested_dict(store: AlertStore) -> None:
    store.create_rule("fid-low", "fidelity_score", "<", 0.5, _WEBHOOK)
    result = {
        "confidence": 0.8,
        "fidelity": {"fidelity_score": 0.3, "low_fidelity": True},
    }
    fired = store.evaluate("analysis_xyz", result)
    assert len(fired) == 1
    assert fired[0].field == "fidelity_score"


def test_evaluate_records_in_history(store: AlertStore) -> None:
    store.create_rule("low-conf", "confidence", "<", 0.4, _WEBHOOK)
    store.evaluate("analysis_abc", {"confidence": 0.2})
    history = store.list_alerts()
    assert len(history) == 1
    assert history[0].analysis_id == "analysis_abc"


def test_evaluate_multiple_rules(store: AlertStore) -> None:
    store.create_rule("low-conf", "confidence", "<", 0.4, _WEBHOOK)
    store.create_rule("very-low-conf", "confidence", "<", 0.2, _WEBHOOK)
    fired = store.evaluate("x", {"confidence": 0.1})
    assert len(fired) == 2


def test_evaluate_disabled_rule_not_fired(store: AlertStore) -> None:
    rule = store.create_rule("low-conf", "confidence", "<", 0.4, _WEBHOOK)
    store.set_rule_enabled(rule.rule_id, enabled=False)
    fired = store.evaluate("x", {"confidence": 0.1})
    assert fired == []


def test_list_alerts_limit(store: AlertStore) -> None:
    store.create_rule("low-conf", "confidence", "<", 0.9, _WEBHOOK)
    for i in range(5):
        store.evaluate(f"analysis_{i}", {"confidence": 0.1})
    history = store.list_alerts(limit=3)
    assert len(history) == 3


def test_mark_delivered(store: AlertStore) -> None:
    store.create_rule("low-conf", "confidence", "<", 0.4, _WEBHOOK)
    fired = store.evaluate("analysis_abc", {"confidence": 0.2})
    store.mark_delivered(fired[0].alert_id, delivered=True)
    history = store.list_alerts()
    assert history[0].delivered is True


# ---------------------------------------------------------------------------
# _new_id determinism
# ---------------------------------------------------------------------------


def test_new_id_deterministic() -> None:
    assert _new_id("rule", "a", "b") == _new_id("rule", "a", "b")


def test_new_id_different_inputs() -> None:
    assert _new_id("rule", "a") != _new_id("rule", "b")
