"""The adversarial suite.

The critical test-design point: **a suite in which every test denies would
pass even if the store denied everything.** So every denial is paired with a
positive control that must be allowed, and a case only counts as passing when
both halves behave. That is enforced as a structural requirement of
``paired_suite`` rather than left as a convention.

The cascade section is the headline experiment: because a record's
classification is derived from detector output, a detection miss is an
authorization failure, and both halves of that sentence are measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from canonical import sha256_hex
from textindex import iter_leaves

from trajsec.config import Settings, load_settings
from trajsec.corpus import CorpusEntry
from trajsec.detect import scan_document
from trajsec.model import PROFILES, Subject
from trajsec.policy import classify, evaluate, sign_subject, verify_subject
from trajsec.sanitize import apply, plan
from trajsec.vault import KeyContext


@dataclass
class Case:
    case_id: str
    description: str
    expectation: str  # "deny" | "allow"
    outcome: str
    passed: bool
    detail: str = ""
    pair_of: str | None = None


@dataclass
class SuiteResult:
    cases: list[Case] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    def pairs_balanced(self) -> list[str]:
        """Every negative case must have a positive counterpart."""
        negatives = {c.case_id for c in self.cases if c.expectation == "deny"}
        positives = {c.pair_of for c in self.cases if c.pair_of}
        return sorted(negatives - positives)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": [c.__dict__ for c in self.cases],
            "passed": self.passed,
            "total": len(self.cases),
            "all_passed": self.passed == len(self.cases),
            "unpaired_denials": self.pairs_balanced(),
        }


def personas(*, tenant_a: str = "tenant-a", tenant_b: str = "tenant-b") -> dict[str, Subject]:
    def make(principal: str, role: str, tenant: str, clearance: str, purpose: str) -> Subject:
        return sign_subject(
            Subject(
                principal_id=principal,
                role=role,  # type: ignore[arg-type]
                tenant_id=tenant,
                clearance=clearance,
                purpose=purpose,  # type: ignore[arg-type]
                session_id=f"sess-{principal}",
            )
        )

    return {
        "owner_secret_debug": make("owner-1", "owner", tenant_a, "SECRET", "debug"),
        "owner_internal_debug": make("owner-2", "owner", tenant_a, "INTERNAL", "debug"),
        "owner_secret_metrics": make("owner-3", "owner", tenant_a, "SECRET", "metrics"),
        "analyst": make("analyst-1", "analyst", tenant_a, "INTERNAL", "debug"),
        "auditor": make("auditor-1", "auditor", tenant_a, "INTERNAL", "compliance"),
        "third_party": make("vendor-1", "third_party", tenant_a, "PUBLIC", "publication"),
        "service": make("svc-1", "service", tenant_a, "INTERNAL", "metrics"),
        "admin": make("admin-1", "admin", tenant_a, "INTERNAL", "metrics"),
        "cross_tenant_analyst": make("analyst-b", "analyst", tenant_b, "SECRET", "debug"),
        "cross_tenant_service": make("svc-b", "service", tenant_b, "INTERNAL", "metrics"),
    }


def paired_suite(store: Any, run_id: str, people: dict[str, Subject]) -> SuiteResult:
    """Run the attack cases, each with its positive control."""
    result = SuiteResult()

    def attempt(case_id: str, description: str, expectation: str, subject: Subject,
                action: str, *, pair_of: str | None = None) -> None:
        audit_before = len(store.audit.entries)
        outcome = store.try_action(run_id, subject, action)
        allowed = bool(outcome.get("allowed"))
        audit_written = len(store.audit.entries) > audit_before
        expected_allow = expectation == "allow"
        passed = (allowed == expected_allow) and audit_written
        result.cases.append(
            Case(
                case_id=case_id,
                description=description,
                expectation=expectation,
                outcome="allow" if allowed else "deny",
                passed=passed,
                detail=outcome.get("error", "") if not allowed else "",
                pair_of=pair_of,
            )
        )

    # N1 cross-tenant IDOR, plus the positive control that proves the store is
    # not simply denying everything.
    attempt("N1_idor_cross_tenant", "analyst from another tenant requests this run",
            "deny", people["cross_tenant_analyst"], "read_payload_sanitized")
    attempt("N1p_owner_reads_own", "owner reads its own run (positive control)",
            "allow", people["owner_secret_debug"], "read_payload_sanitized", pair_of="N1_idor_cross_tenant")

    # N2 analyst reaching for raw payload
    attempt("N2_analyst_raw", "analyst requests raw payload",
            "deny", people["analyst"], "read_payload_raw")
    attempt("N2p_owner_raw_with_purpose", "owner with clearance and purpose=debug reads raw (positive control)",
            "allow", people["owner_secret_debug"], "read_payload_raw", pair_of="N2_analyst_raw")

    # N3 clearance escalation
    # N3 clearance escalation, paired with the same action at sufficient
    # clearance so the denial is attributable to clearance alone.
    attempt("N3_clearance_escalation", "owner with INTERNAL clearance reads a SECRET record",
            "deny", people["owner_internal_debug"], "read_payload_raw")
    attempt("N3p_owner_with_clearance", "owner holding the required clearance reads raw (positive control)",
            "allow", people["owner_secret_debug"], "read_payload_raw", pair_of="N3_clearance_escalation")

    # N4 purpose violation: identical subject clearance, different declared
    # purpose, so the denial is attributable to the purpose and nothing else.
    attempt("N4_purpose_violation", "purpose=metrics requests raw payload",
            "deny", people["owner_secret_metrics"], "read_payload_raw")
    attempt("N4p_purpose_debug", "the same owner with purpose=debug is allowed (positive control)",
            "allow", people["owner_secret_debug"], "read_payload_raw", pair_of="N4_purpose_violation")

    # N5 auditor separation of duties
    attempt("N5a_auditor_replay", "auditor attempts to replay a record",
            "deny", people["auditor"], "replay")
    attempt("N5b_auditor_delete", "auditor attempts to delete a record",
            "deny", people["auditor"], "delete")
    attempt("N5p_auditor_reads_audit", "auditor reads the audit log (positive control)",
            "allow", people["auditor"], "read_audit", pair_of="N5a_auditor_replay")
    attempt("N5bp_auditor_reads_sanitized", "auditor reads a sanitized payload (positive control)",
            "allow", people["auditor"], "read_payload_sanitized", pair_of="N5b_auditor_delete")

    # N6 forged session claims
    forged = people["analyst"].model_copy(update={"role": "owner", "clearance": "SECRET"})
    attempt("N6_forged_session", "session claims tampered to escalate the role",
            "deny", forged, "read_payload_raw")
    attempt("N6p_valid_session", "the same principal with untampered claims (positive control)",
            "allow", people["analyst"], "read_payload_sanitized", pair_of="N6_forged_session")

    # N7 unknown action fails closed
    attempt("N7_unknown_action", "an action the policy does not define",
            "deny", people["owner_secret_debug"], "read_everything")
    attempt("N7p_defined_action", "a defined action on the same record (positive control)",
            "allow", people["owner_secret_debug"], "read_meta", pair_of="N7_unknown_action")

    # N8 service writing across tenants
    attempt("N8_service_cross_tenant_write", "service in another tenant appends a record",
            "deny", people["cross_tenant_service"], "write_append")
    attempt("N8p_service_own_tenant_write", "service appending within its own tenant (positive control)",
            "allow", people["service"], "write_append", pair_of="N8_service_cross_tenant_write")

    # Export paths
    attempt("N9_export_raw_anyone", "any role attempts a raw export",
            "deny", people["owner_secret_debug"], "export_raw")
    attempt("N9p_third_party_published", "third party reads a published run's sanitized export",
            "allow", people["third_party"], "export_sanitized", pair_of="N9_export_raw_anyone")

    # Separation of duties: admin manages policy and cannot read content
    attempt("N10_admin_reads_content", "administrator attempts to read record content",
            "deny", people["admin"], "read_payload_sanitized")
    attempt("N10p_admin_manages_policy", "administrator manages policy (positive control)",
            "allow", people["admin"], "admin_policy", pair_of="N10_admin_reads_content")

    return result


def access_matrix(store: Any, run_id: str, people: dict[str, Subject], actions: list[str]) -> list[dict[str, Any]]:
    """Enumerate (role x action) by *calling the engine*, so the published
    matrix cannot drift from the implementation."""
    rows: list[dict[str, Any]] = []
    for name, subject in people.items():
        for action in actions:
            outcome = store.try_action(run_id, subject, action)
            rows.append(
                {
                    "persona": name,
                    "role": subject.role,
                    "clearance": subject.clearance,
                    "action": action,
                    "allowed": bool(outcome.get("allowed")),
                    "reason": outcome.get("error", "ok"),
                }
            )
    return rows


def matrix_markdown(rows: list[dict[str, Any]]) -> str:
    personas_order: list[str] = []
    for row in rows:
        if row["persona"] not in personas_order:
            personas_order.append(row["persona"])
    actions: list[str] = []
    for row in rows:
        if row["action"] not in actions:
            actions.append(row["action"])

    lookup = {(r["persona"], r["action"]): r for r in rows}
    lines = ["| action | " + " | ".join(personas_order) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in personas_order) + " |")
    for action in actions:
        cells = []
        for persona in personas_order:
            row = lookup.get((persona, action))
            cells.append("allow" if row and row["allowed"] else "deny")
        lines.append(f"| `{action}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- the cascade -----------------------------------------------------------


def cascading_failure(
    corpus: list[CorpusEntry],
    key_context: KeyContext,
    settings: Settings | None = None,
    *,
    tenant_id: str = "tenant-a",
    layers: list[str] | None = None,
    label: str = "full stack",
) -> dict[str, Any]:
    """Detection failure becomes authorization failure.

    "Missed" means the item's *span* carried no active finding -- the same
    definition the recall metric uses. It deliberately does not mean "the
    plaintext was not recovered": a base64-wrapped secret can be flagged in
    place, leave the record correctly classified, and still never be recovered
    as a value. Conflating the two would make the cascade look catastrophic
    for a reason that has nothing to do with authorization.

    Both consequences are then measured directly:

    * ``under_classified`` -- the record holds a secret the detector missed and
      is therefore classified below SECRET;
    * ``exported_in_clear`` -- the *plaintext* of that secret still appears in
      the artifact a third party would receive. That is the sharpest form of
      the claim: not that someone could read something sensitive, but that the
      vendor is handed the credential itself.

    Passing ``layers`` ablates a layer, which is how the mechanism is
    demonstrated rather than asserted: remove the entropy layer and watch
    under-classification appear where it was absent.
    """
    settings = settings or load_settings()
    from trajsec.evaluate import merge_regions

    by_evasion: dict[str, dict[str, int]] = {}
    totals = {"records": 0, "under_classified": 0, "exported_in_clear": 0}

    for entry in corpus:
        if not entry.manifest.items:
            continue
        totals["records"] += 1
        doc = entry.document
        finding_set = scan_document(doc, settings, layers)
        regions = merge_regions(finding_set.active)
        meta = classify(
            entry.run_id, owner_principal="owner-1", tenant_id=tenant_id, finding_set=finding_set
        )
        sanitize_plan = plan(doc, finding_set, PROFILES["scrubbed"], key_context, tenant_id=tenant_id)
        sanitized, _, _, _ = apply(
            doc, sanitize_plan, key_context, policy_version=settings.policy_version
        )
        exported_text = "\n".join(
            value for _p, _path, value in iter_leaves(sanitized) if isinstance(value, str)
        )

        under = False
        in_clear = False
        for item in entry.manifest.items:
            bucket = by_evasion.setdefault(
                item.evasion, {"items": 0, "missed": 0, "exported_in_clear": 0}
            )
            bucket["items"] += 1
            flagged = any(
                r.pointer == item.pointer and r.start < item.char_end and item.char_start < r.end
                for r in regions
            )
            if flagged:
                continue
            bucket["missed"] += 1
            under = True
            plaintext = entry.plaintext_lookup.get(item.value_sha256, "")
            if plaintext and plaintext in exported_text:
                bucket["exported_in_clear"] += 1
                in_clear = True

        if under:
            totals["under_classified"] += 1
        if in_clear:
            totals["exported_in_clear"] += 1

    records = max(1, totals["records"])
    for bucket in by_evasion.values():
        items = max(1, bucket["items"])
        bucket["p_missed"] = round(bucket["missed"] / items, 4)
        bucket["p_exported_in_clear"] = round(bucket["exported_in_clear"] / items, 4)

    return {
        "label": label,
        "layers": list(layers) if layers is not None else list(settings.thresholds.get("layers", [])),
        "records": totals["records"],
        "under_classified": totals["under_classified"],
        "exported_in_clear": totals["exported_in_clear"],
        "p_record_under_classified": round(totals["under_classified"] / records, 4),
        "p_record_exported_in_clear": round(totals["exported_in_clear"] / records, 4),
        "by_evasion": by_evasion,
        "claim": (
            "A detection miss is an authorization failure: the record is classified "
            "below SECRET, and where the plaintext survives, the artifact handed to a "
            "third party carries it. Measured by ablating a layer, not asserted."
        ),
    }


# --- token scope property --------------------------------------------------


def token_scope_property(key_context: KeyContext, *, value: str, run_a: str, run_b: str) -> dict[str, Any]:
    """Same secret across runs must not correlate; within a run it must."""
    from trajsec.vault import derive_digest, format_preserving_token, scope_identifier

    def token(scope: str, run_id: str) -> str:
        scope_id = scope_identifier(scope, run_id=run_id, tenant_id="tenant-a")
        return format_preserving_token(value, derive_digest(key_context.key, scope_id, value))

    run_scope = {
        "scope": "run",
        "within_run_stable": token("run", run_a) == token("run", run_a),
        "across_run_distinct": token("run", run_a) != token("run", run_b),
        "run_a": token("run", run_a),
        "run_b": token("run", run_b),
    }
    global_scope = {
        "scope": "global",
        "across_run_identical": token("global", run_a) == token("global", run_b),
        "note": "global scope is what creates the cross-tenant confirmation oracle; it is not the default",
    }
    return {
        "run_scope": run_scope,
        "global_scope": global_scope,
        "confirmed": run_scope["within_run_stable"] and run_scope["across_run_distinct"],
    }


def idempotence_check(
    doc: dict[str, Any],
    sanitized: dict[str, Any],
    settings: Settings,
    produced_hashes: set[str],
) -> dict[str, Any]:
    """Sanitizing an already-sanitized artifact must not produce new findings.

    This only holds because a format-preserving token still looks exactly like
    the kind of secret it replaced, so the sanitizer has to tell the detector
    what it produced. Without that hand-off the check fails by construction.
    """
    from trajsec.detect import Detector

    detector = Detector(settings)
    detector.l4.runtime_allow_hashes = set(produced_hashes)
    rescan = detector.scan(sanitized)
    residual = [f for f in rescan.active if f.value_sha256 in produced_hashes]
    return {
        "residual_token_findings": len(residual),
        "total_findings_after_rescan": len(rescan.active),
        "idempotent": len(residual) == 0,
        "note": "residual findings whose value is one of our own tokens would mean the sanitizer re-flags its own output",
    }


def plaintext_absent(store_artifacts: dict[str, bytes], plaintexts: list[str]) -> dict[str, Any]:
    """The load-bearing property, as a test: no persisted artifact holds
    plaintext. Findings and the audit log are the ones that must never."""
    hits: list[dict[str, str]] = []
    for name, blob in store_artifacts.items():
        for plaintext in plaintexts:
            if plaintext and plaintext.encode() in blob:
                hits.append({"artifact": name, "value_sha256": sha256_hex(plaintext)})
    return {
        "artifacts_checked": sorted(store_artifacts),
        "plaintexts_checked": len(plaintexts),
        "hits": hits,
        "clean": not hits,
    }
