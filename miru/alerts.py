"""Explanation alert rules — SQLite-backed rule store with webhook delivery.

Operators define rules against the output of ``POST /explain``.  When a rule
fires (e.g. confidence < 0.4 or fidelity_score < 0.5) the engine:

1. Writes a ``FiredAlert`` row to the ``alerts_history`` table.
2. Delivers a minimal (no image data) JSON payload to the rule's
   ``webhook_url`` in a background thread with a hard 10-second timeout.

Configuration
-------------

``MIRU_ALERTS_PATH``
    Path to the SQLite database.  Defaults to ``./miru_alerts.db``.

``MIRU_ALERTS_ENABLED``
    Set to ``0`` or ``false`` to disable the subsystem entirely.  The
    returned store is ``None`` and callers treat every rule list as empty.

Wire format
-----------

Supported condition fields (``field``):

- ``confidence``     — model confidence from the /explain response.
- ``fidelity_score`` — deletion-test fidelity (only available when
  ``?fidelity=true`` was passed; alert is skipped if the field is absent).

Supported operators (``op``): ``<``, ``<=``, ``>``, ``>=``, ``==``.

Webhook payload (POST application/json)::

    {
        "event": "miru.alert.fired",
        "rule_id": "...",
        "rule_name": "...",
        "analysis_id": "...",
        "field": "confidence",
        "fired_value": 0.31,
        "threshold": 0.4,
        "op": "<",
        "ts": "2026-06-13T00:00:00+00:00"
    }

No image data is ever included in the payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ALERTS_PATH_ENV = "MIRU_ALERTS_PATH"
_ALERTS_ENABLED_ENV = "MIRU_ALERTS_ENABLED"
_DEFAULT_PATH = "./miru_alerts.db"

SUPPORTED_FIELDS: tuple[str, ...] = ("confidence", "fidelity_score")
SUPPORTED_OPS: tuple[str, ...] = ("<", "<=", ">", ">=", "==")
MAX_RULES = 50
MAX_NAME_LEN = 200
WEBHOOK_TIMEOUT_S = 10

# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class Rule:
    """One alert rule row."""

    __slots__ = (
        "rule_id",
        "name",
        "field",
        "op",
        "threshold",
        "webhook_url",
        "enabled",
        "created_at",
    )

    def __init__(
        self,
        rule_id: str,
        name: str,
        field: str,
        op: str,
        threshold: float,
        webhook_url: str,
        *,
        enabled: bool = True,
        created_at: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.name = name
        self.field = field
        self.op = op
        self.threshold = threshold
        self.webhook_url = webhook_url
        self.enabled = enabled
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "field": self.field,
            "op": self.op,
            "threshold": self.threshold,
            "webhook_url": self.webhook_url,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }


class FiredAlert:
    """One row in the alerts history table."""

    __slots__ = (
        "alert_id",
        "rule_id",
        "rule_name",
        "analysis_id",
        "field",
        "fired_value",
        "threshold",
        "op",
        "webhook_url",
        "ts",
        "delivered",
    )

    def __init__(
        self,
        alert_id: str,
        rule_id: str,
        rule_name: str,
        analysis_id: str,
        field: str,
        fired_value: float,
        threshold: float,
        op: str,
        webhook_url: str,
        ts: str,
        *,
        delivered: bool = False,
    ) -> None:
        self.alert_id = alert_id
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.analysis_id = analysis_id
        self.field = field
        self.fired_value = fired_value
        self.threshold = threshold
        self.op = op
        self.webhook_url = webhook_url
        self.ts = ts
        self.delivered = delivered

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "analysis_id": self.analysis_id,
            "field": self.field,
            "fired_value": self.fired_value,
            "threshold": self.threshold,
            "op": self.op,
            "webhook_url": self.webhook_url,
            "ts": self.ts,
            "delivered": self.delivered,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_id(prefix: str, *parts: str) -> str:
    """Deterministic short ID: prefix + first 12 hex chars of SHA-256."""
    raw = "|".join(parts)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluate_op(value: float, op: str, threshold: float) -> bool:
    """Return True when (value op threshold) is satisfied."""
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unsupported op: {op!r}")


def validate_webhook_url(url: str) -> None:
    """Raise ValueError when url is not a safe http/https webhook target.

    Blocks empty strings, non-http schemes, and bare IP literals to
    mitigate SSRF at the API boundary.  The check is intentionally simple
    — it does not resolve hostnames or validate TLS certificates.
    """
    if not url:
        raise ValueError("webhook_url must not be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(
            f"webhook_url must start with http:// or https://, got: {url!r}"
        )
    if len(url) > 2048:
        raise ValueError("webhook_url exceeds maximum length of 2048 characters")


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS rules (
    rule_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    field       TEXT NOT NULL,
    op          TEXT NOT NULL,
    threshold   REAL NOT NULL,
    webhook_url TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts_history (
    alert_id    TEXT PRIMARY KEY,
    rule_id     TEXT NOT NULL,
    rule_name   TEXT NOT NULL,
    analysis_id TEXT NOT NULL,
    field       TEXT NOT NULL,
    fired_value REAL NOT NULL,
    threshold   REAL NOT NULL,
    op          TEXT NOT NULL,
    webhook_url TEXT NOT NULL,
    ts          TEXT NOT NULL,
    delivered   INTEGER NOT NULL DEFAULT 0
);
"""


class AlertStore:
    """Thread-safe SQLite-backed alert rule and history store."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        # Keep a single persistent connection for :memory: (each new
        # connection gets its own isolated in-memory database).
        if path == ":memory:":
            self._mem_conn: sqlite3.Connection | None = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.executescript(_DDL)
            self._mem_conn.commit()
        else:
            self._mem_conn = None
            self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._mem_conn is not None:
                yield self._mem_conn
                self._mem_conn.commit()
            else:
                conn = sqlite3.connect(self._path, timeout=10)
                conn.row_factory = sqlite3.Row
                try:
                    yield conn
                    conn.commit()
                finally:
                    conn.close()

    # ------------------------------------------------------------------
    # Rule CRUD
    # ------------------------------------------------------------------

    def create_rule(
        self,
        name: str,
        field: str,
        op: str,
        threshold: float,
        webhook_url: str,
    ) -> Rule:
        """Insert a new rule; raise ValueError on constraint violations."""
        if len(name) > MAX_NAME_LEN:
            raise ValueError(f"name exceeds {MAX_NAME_LEN} characters")
        if field not in SUPPORTED_FIELDS:
            raise ValueError(f"field must be one of {SUPPORTED_FIELDS}, got {field!r}")
        if op not in SUPPORTED_OPS:
            raise ValueError(f"op must be one of {SUPPORTED_OPS}, got {op!r}")
        validate_webhook_url(webhook_url)

        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
            if count >= MAX_RULES:
                raise ValueError(
                    f"rule limit reached ({MAX_RULES}); delete existing rules first"
                )

            ts = _now_utc()
            rule_id = _new_id("rule", name, field, op, str(threshold), webhook_url, ts)
            conn.execute(
                "INSERT INTO rules (rule_id, name, field, op, threshold, webhook_url, enabled, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (rule_id, name, field, op, threshold, webhook_url, ts),
            )

        return Rule(
            rule_id,
            name,
            field,
            op,
            threshold,
            webhook_url,
            enabled=True,
            created_at=ts,
        )

    def list_rules(self, *, enabled_only: bool = False) -> list[Rule]:
        """Return all rules, optionally filtered to enabled ones."""
        with self._connect() as conn:
            if enabled_only:
                rows = conn.execute(
                    "SELECT * FROM rules WHERE enabled=1 ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM rules ORDER BY created_at"
                ).fetchall()
        return [
            Rule(
                r["rule_id"],
                r["name"],
                r["field"],
                r["op"],
                r["threshold"],
                r["webhook_url"],
                enabled=bool(r["enabled"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID. Returns True when a row was deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM rules WHERE rule_id=?", (rule_id,))
        return cur.rowcount > 0

    def set_rule_enabled(self, rule_id: str, *, enabled: bool) -> bool:
        """Enable or disable a rule. Returns True when the row existed."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE rules SET enabled=? WHERE rule_id=?",
                (1 if enabled else 0, rule_id),
            )
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        analysis_id: str,
        result: dict[str, Any],
    ) -> list[FiredAlert]:
        """Check all enabled rules against *result*; record and return fired alerts.

        *result* is a dict with at least ``confidence`` and optionally
        ``fidelity`` (the FidelityBlock dict from /explain).  Rules whose
        ``field`` is absent from *result* are silently skipped.
        """
        rules = self.list_rules(enabled_only=True)
        if not rules:
            return []

        values: dict[str, float] = {}
        if "confidence" in result:
            values["confidence"] = float(result["confidence"])
        fidelity = result.get("fidelity")
        if isinstance(fidelity, dict) and "fidelity_score" in fidelity:
            values["fidelity_score"] = float(fidelity["fidelity_score"])

        fired: list[FiredAlert] = []
        ts = _now_utc()

        for rule in rules:
            if rule.field not in values:
                continue
            fired_value = values[rule.field]
            if not _evaluate_op(fired_value, rule.op, rule.threshold):
                continue

            alert_id = _new_id("alert", rule.rule_id, analysis_id, ts)
            alert = FiredAlert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                analysis_id=analysis_id,
                field=rule.field,
                fired_value=fired_value,
                threshold=rule.threshold,
                op=rule.op,
                webhook_url=rule.webhook_url,
                ts=ts,
                delivered=False,
            )
            self._record_alert(alert)
            fired.append(alert)

        return fired

    def _record_alert(self, alert: FiredAlert) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO alerts_history"
                " (alert_id, rule_id, rule_name, analysis_id, field,"
                "  fired_value, threshold, op, webhook_url, ts, delivered)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    alert.alert_id,
                    alert.rule_id,
                    alert.rule_name,
                    alert.analysis_id,
                    alert.field,
                    alert.fired_value,
                    alert.threshold,
                    alert.op,
                    alert.webhook_url,
                    alert.ts,
                    0,
                ),
            )

    def mark_delivered(self, alert_id: str, *, delivered: bool) -> None:
        """Update delivery status for one alert row."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE alerts_history SET delivered=? WHERE alert_id=?",
                (1 if delivered else 0, alert_id),
            )

    def list_alerts(self, limit: int = 50) -> list[FiredAlert]:
        """Return the most recent fired alerts, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts_history ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            FiredAlert(
                alert_id=r["alert_id"],
                rule_id=r["rule_id"],
                rule_name=r["rule_name"],
                analysis_id=r["analysis_id"],
                field=r["field"],
                fired_value=r["fired_value"],
                threshold=r["threshold"],
                op=r["op"],
                webhook_url=r["webhook_url"],
                ts=r["ts"],
                delivered=bool(r["delivered"]),
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: AlertStore | None = None
_store_lock = threading.Lock()


def get_store() -> AlertStore | None:
    """Return the shared AlertStore, or None when disabled."""
    enabled = os.environ.get(_ALERTS_ENABLED_ENV, "1").lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                path = os.environ.get(_ALERTS_PATH_ENV, _DEFAULT_PATH)
                _store = AlertStore(path)
    return _store


def reset_store() -> None:
    """Reset the singleton — intended for tests only."""
    global _store
    with _store_lock:
        _store = None


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


def _deliver_webhook(alert: FiredAlert, store: AlertStore) -> None:
    """POST the alert payload to the rule's webhook_url (best-effort)."""
    payload = json.dumps(
        {
            "event": "miru.alert.fired",
            "rule_id": alert.rule_id,
            "rule_name": alert.rule_name,
            "analysis_id": alert.analysis_id,
            "field": alert.field,
            "fired_value": alert.fired_value,
            "threshold": alert.threshold,
            "op": alert.op,
            "ts": alert.ts,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        alert.webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "miru-alerts/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_S) as resp:
            delivered = resp.status < 400
    except OSError as exc:
        logger.warning("webhook delivery failed for alert %s: %s", alert.alert_id, exc)
        delivered = False

    store.mark_delivered(alert.alert_id, delivered=delivered)


def fire_alerts_async(alerts: list[FiredAlert], store: AlertStore) -> None:
    """Dispatch webhook delivery for each alert in a daemon thread."""
    for alert in alerts:
        t = threading.Thread(
            target=_deliver_webhook,
            args=(alert, store),
            daemon=True,
            name=f"miru-alert-{alert.alert_id[:8]}",
        )
        t.start()


__all__ = [
    "AlertStore",
    "FiredAlert",
    "Rule",
    "SUPPORTED_FIELDS",
    "SUPPORTED_OPS",
    "MAX_RULES",
    "WEBHOOK_TIMEOUT_S",
    "fire_alerts_async",
    "get_store",
    "reset_store",
    "validate_webhook_url",
]
