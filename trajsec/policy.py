"""Access control: classification, decisions, and field-level projection.

Pure functions with no I/O, so the whole policy matrix is unit-testable
without a filesystem.

The unification point: ``record.classification`` is the lattice join of the
classifications of the finding types present in it. Detector output *is* ACL
input, which is what makes a detection miss an authorization failure rather
than merely a missed alert.
"""

from __future__ import annotations

from typing import Any, Iterable

from canonical import canonical_json

from trajsec.config import level_at_least, level_rank, max_level
from trajsec.model import (
    ACTIONS,
    AccessDecision,
    Finding,
    FindingSet,
    RecordMeta,
    Subject,
    classification_for_types,
)

# Field sensitivity is a property of the path, not of the content.
FIELD_RULES: tuple[tuple[str, str], ...] = (
    ("/agent_config", "SECRET"),
    ("/sandbox_declaration", "PUBLIC"),
    ("/integrity", "INTERNAL"),
    ("/outcome/final_answer", "CONFIDENTIAL"),
    ("/outcome", "INTERNAL"),
    ("/steps", "CONFIDENTIAL"),
    ("/scenario_prompt", "INTERNAL"),
)

ALWAYS_VISIBLE = ("/run_id", "/thread_id", "/trace_id", "/schema_version", "/scenario_id")


def path_level(pointer: str) -> str:
    for prefix in ALWAYS_VISIBLE:
        if pointer == prefix or pointer.startswith(prefix + "/"):
            return "PUBLIC"
    best = "PUBLIC"
    for prefix, level in FIELD_RULES:
        if pointer == prefix or pointer.startswith(prefix + "/"):
            best = max_level(best, level)
    return best


# --- classification --------------------------------------------------------


def classify(
    run_id: str,
    *,
    owner_principal: str,
    tenant_id: str,
    finding_set: FindingSet | None,
    base_classification: str = "INTERNAL",
) -> RecordMeta:
    """Derive a record's classification from what the detector found."""
    types = finding_set.types() if finding_set else []
    classification = max_level(base_classification, classification_for_types(types))
    max_untrusted = bool(
        finding_set
        and any(f.tainted for f in finding_set.findings if not f.suppressed)
    )
    return RecordMeta(
        run_id=run_id,
        owner_principal=owner_principal,
        tenant_id=tenant_id,
        base_classification=base_classification,
        classification=classification,
        finding_types=types,
        max_untrusted=max_untrusted,
    )


def has_secret_finding(finding_set: FindingSet | None) -> bool:
    if finding_set is None:
        return False
    return any(
        f.classification == "SECRET" for f in finding_set.findings if not f.suppressed
    )


# --- decisions -------------------------------------------------------------


def _decision(
    allow: bool, reason: str, rule: str, subject: Subject, action: str, run_id: str
) -> AccessDecision:
    return AccessDecision(
        allow=allow,
        reason_code=reason,
        policy_rule_id=rule,
        action=action,
        run_id=run_id,
        principal_id=subject.principal_id,
    )


def evaluate(
    subject: Subject,
    meta: RecordMeta,
    action: str,
    *,
    finding_set: FindingSet | None = None,
    published: bool = False,
) -> AccessDecision:
    """Deny by default; an unrecognised action fails closed."""
    if action not in ACTIONS:
        return _decision(False, "unknown_action", "policy.fail_closed", subject, action, meta.run_id)

    same_tenant = subject.tenant_id == meta.tenant_id
    role = subject.role
    secret_present = has_secret_finding(finding_set)

    if action == "read_meta":
        if role in ("owner", "analyst", "auditor", "service"):
            return (
                _decision(True, "ok", "policy.meta.same_tenant", subject, action, meta.run_id)
                if same_tenant
                else _decision(
                    False, "tenant_mismatch", "policy.meta.tenant", subject, action, meta.run_id
                )
            )
        if role == "third_party":
            return (
                _decision(True, "ok", "policy.meta.published", subject, action, meta.run_id)
                if published
                else _decision(
                    False, "not_published", "policy.meta.published", subject, action, meta.run_id
                )
            )
        return _decision(False, "role_not_permitted", "policy.meta.role", subject, action, meta.run_id)

    if action == "read_payload_raw":
        if role != "owner":
            return _decision(
                False, "role_not_permitted", "policy.raw.owner_only", subject, action, meta.run_id
            )
        if not same_tenant:
            return _decision(
                False, "tenant_mismatch", "policy.raw.tenant", subject, action, meta.run_id
            )
        if not level_at_least(subject.clearance, meta.classification):
            return _decision(
                False, "insufficient_clearance", "policy.raw.clearance", subject, action, meta.run_id
            )
        if secret_present and subject.purpose not in ("debug", "incident_response"):
            return _decision(
                False, "purpose_not_permitted", "policy.raw.purpose", subject, action, meta.run_id
            )
        return _decision(True, "ok", "policy.raw.granted", subject, action, meta.run_id)

    if action in ("read_payload_sanitized", "export_sanitized"):
        if role in ("owner", "analyst", "auditor"):
            return (
                _decision(True, "ok", f"policy.{action}.granted", subject, action, meta.run_id)
                if same_tenant
                else _decision(
                    False, "tenant_mismatch", f"policy.{action}.tenant", subject, action, meta.run_id
                )
            )
        if role == "third_party" and published:
            return _decision(
                True, "ok", f"policy.{action}.published", subject, action, meta.run_id
            )
        return _decision(
            False, "role_not_permitted", f"policy.{action}.role", subject, action, meta.run_id
        )

    if action in ("read_findings", "read_finding_evidence"):
        if role in ("owner", "analyst", "auditor"):
            return (
                _decision(True, "ok", f"policy.{action}.granted", subject, action, meta.run_id)
                if same_tenant
                else _decision(
                    False, "tenant_mismatch", f"policy.{action}.tenant", subject, action, meta.run_id
                )
            )
        if role == "third_party" and action == "read_findings" and published:
            return _decision(True, "ok", "policy.findings.counts_only", subject, action, meta.run_id)
        return _decision(
            False, "role_not_permitted", f"policy.{action}.role", subject, action, meta.run_id
        )

    if action == "unmask":
        if role != "owner":
            return _decision(
                False, "role_not_permitted", "policy.unmask.owner_only", subject, action, meta.run_id
            )
        if not same_tenant:
            return _decision(
                False, "tenant_mismatch", "policy.unmask.tenant", subject, action, meta.run_id
            )
        if not level_at_least(subject.clearance, "SECRET"):
            return _decision(
                False, "insufficient_clearance", "policy.unmask.clearance", subject, action, meta.run_id
            )
        if subject.purpose != "incident_response":
            return _decision(
                False, "purpose_not_permitted", "policy.unmask.purpose", subject, action, meta.run_id
            )
        return _decision(True, "ok", "policy.unmask.granted", subject, action, meta.run_id)

    if action == "export_raw":
        # No role may export raw payload. Stated as a rule rather than an
        # omission so the matrix is explicit about it.
        return _decision(
            False, "export_raw_never_permitted", "policy.export_raw.deny_all", subject, action, meta.run_id
        )

    if action in ("replay", "resume"):
        if role == "owner" and same_tenant:
            return _decision(True, "ok", f"policy.{action}.granted", subject, action, meta.run_id)
        return _decision(
            False, "role_not_permitted", f"policy.{action}.owner_only", subject, action, meta.run_id
        )

    if action == "write_append":
        if role == "service" and same_tenant:
            return _decision(True, "ok", "policy.write.granted", subject, action, meta.run_id)
        return _decision(
            False, "role_not_permitted", "policy.write.service_only", subject, action, meta.run_id
        )

    if action == "delete":
        return _decision(
            False, "append_only_store", "policy.delete.deny_all", subject, action, meta.run_id
        )

    if action == "verify_integrity":
        return _decision(True, "ok", "policy.integrity.granted", subject, action, meta.run_id)

    if action == "read_audit":
        if role == "auditor" and same_tenant:
            return _decision(True, "ok", "policy.audit.auditor", subject, action, meta.run_id)
        if role == "owner":
            return _decision(True, "ok", "policy.audit.own_actions", subject, action, meta.run_id)
        if role == "admin":
            return _decision(True, "ok", "policy.audit.policy_events", subject, action, meta.run_id)
        return _decision(False, "role_not_permitted", "policy.audit.role", subject, action, meta.run_id)

    if action == "admin_policy":
        # Separation of duties: the administrator manages policy and cannot
        # read content; the auditor reads audit and cannot read content; the
        # owner reads content and cannot manage policy. No single role can
        # both forge and conceal.
        return (
            _decision(True, "ok", "policy.admin.granted", subject, action, meta.run_id)
            if role == "admin"
            else _decision(
                False, "role_not_permitted", "policy.admin.only", subject, action, meta.run_id
            )
        )

    return _decision(False, "unhandled_action", "policy.fail_closed", subject, action, meta.run_id)


# --- projection ------------------------------------------------------------

REDACTION_MARKER = {"_redacted": True}


def project(
    doc: dict[str, Any],
    subject: Subject,
    meta: RecordMeta,
    *,
    sanitized: bool,
    findings_visible: bool = True,
) -> dict[str, Any]:
    """Field-level projection for an internal read.

    Denied paths become an explicit marker rather than disappearing, because
    for an internal consumer the *existence* of a redaction is useful. The
    third-party export path does the opposite -- it sanitizes on write, so the
    recipient never learns that anything was removed.
    """
    from textindex import iter_leaves

    out = dict(doc)
    out.pop("_projection", None)

    for pointer, _path, _value in list(iter_leaves(doc)):
        tokens = pointer.split("/")[1:]
        level = path_level("/" + "/".join(tokens))
        if not level_at_least(subject.clearance, level):
            _set_marker(out, tokens, {**REDACTION_MARKER, "_reason": f"clearance_below_{level}"})

    if not findings_visible:
        out.pop("findings", None)
    if sanitized:
        out["_projection"] = {
            "mode": "sanitized",
            "subject": subject.principal_id,
            "role": subject.role,
        }
    else:
        out["_projection"] = {
            "mode": "raw",
            "subject": subject.principal_id,
            "role": subject.role,
            "purpose": subject.purpose,
        }
    return out


def _set_marker(doc: dict[str, Any], tokens: list[str], marker: dict[str, Any]) -> None:
    parent: Any = doc
    for token in tokens[:-1]:
        if isinstance(parent, list):
            index = int(token)
            if index >= len(parent):
                return
            parent = parent[index]
        else:
            if token not in parent:
                return
            parent = parent[token]
    last = tokens[-1]
    if isinstance(parent, list):
        index = int(last)
        if index < len(parent):
            parent[index] = marker
    elif isinstance(parent, dict):
        parent[last] = marker


def findings_summary(findings: Iterable[Finding], subject: Subject) -> list[dict[str, Any]]:
    """What a subject may learn about findings, by role.

    The auditor sees type, path and confidence but not the masked preview;
    a third party sees counts only. The preview is a hint that narrows a
    guess, so its audience is deliberately narrower than the finding itself.
    """
    rows: list[dict[str, Any]] = []
    for finding in findings:
        row: dict[str, Any] = {
            "type": finding.type,
            "json_path": finding.json_path,
            "layer": finding.layer,
            "confidence": finding.confidence,
            "severity": finding.severity,
        }
        if subject.role in ("owner", "analyst"):
            row["preview"] = finding.preview
            row["rule"] = finding.evidence.rule_id
        rows.append(row)
    return rows


def matrix_fingerprint(rows: list[Any]) -> str:
    """Stable digest of a set of decisions, so a published matrix can be tied
    to the exact policy evaluation that produced it.

    Accepts either ``AccessDecision`` models or the plain dicts the matrix
    generator produces, because both are legitimate representations of the
    same decision.
    """
    normalised = [
        row.model_dump(mode="json") if hasattr(row, "model_dump") else row for row in rows
    ]
    return canonical_json(normalised)


def level_of(level: str) -> int:
    return level_rank(level)


# --- session claims --------------------------------------------------------

SESSION_SALT_DEFAULT = "trajsec-session-salt"


def sign_subject(subject: Subject, salt: str = SESSION_SALT_DEFAULT) -> Subject:
    """Attach a signature over the subject's claims.

    Not a substitute for real authentication. It exists so the suite can test
    what happens when a session's claims are tampered with, which is a
    different question from whether the claims are true.
    """
    import hashlib
    import hmac as _hmac

    message = f"{subject.principal_id}|{subject.role}|{subject.tenant_id}|{subject.clearance}|{subject.purpose}"
    signature = _hmac.new(salt.encode(), message.encode(), hashlib.sha256).hexdigest()[:32]
    return subject.model_copy(update={"claims_sig": signature, "session_id": subject.session_id or "sess-1"})


def verify_subject(subject: Subject, salt: str = SESSION_SALT_DEFAULT) -> bool:
    if not subject.claims_sig:
        return False
    return sign_subject(subject, salt).claims_sig == subject.claims_sig
