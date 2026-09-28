"""The trajectory store: the only module that touches record bytes on disk.

Two publish paths, and the difference matters:

* for an **internal** consumer, access is gated **on read** and a denied field
  is projected as an explicit marker, because for an insider the existence of
  a redaction is useful information;
* for a **third party**, sanitization happens **on write** -- the recipient
  gets a pre-tokenized artifact and never learns that anything was removed,
  because even the count of redactions is metadata.

The store also re-verifies Experiment 1's hash chain on ingest, which is what
connects the two halves of the project: a tampered trajectory is refused at
the door rather than analysed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.schema import Trajectory, verify_chain

from trajsec.audit import AuditLog
from trajsec.config import Settings, load_settings
from trajsec.detect import scan_document
from trajsec.model import FindingSet, Profile, PROFILES, RecordMeta, Subject
from trajsec.policy import classify, evaluate, project, verify_subject
from trajsec.sanitize import apply, plan
from trajsec.vault import KeyContext


class IngestError(RuntimeError):
    pass


@dataclass
class StoredRecord:
    run_id: str
    raw: dict[str, Any]
    sanitized: dict[str, Any]
    meta: RecordMeta
    finding_set: FindingSet
    replacement_map: dict[str, Any] = field(default_factory=dict)
    produced_token_hashes: set[str] = field(default_factory=set)
    tenant_id: str = "tenant-a"
    published: bool = False


class ProtectedStore:
    def __init__(
        self,
        root: Path,
        key_context: KeyContext,
        audit: AuditLog,
        settings: Settings | None = None,
        *,
        tenant_id: str = "tenant-a",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_context = key_context
        self.audit = audit
        self.settings = settings or load_settings()
        self.tenant_id = tenant_id
        self.records: dict[str, StoredRecord] = {}

    # --- ingest ------------------------------------------------------------

    def ingest(
        self,
        doc: dict[str, Any],
        *,
        owner_principal: str = "owner-1",
        profile: str = "redacted",
        published: bool = False,
    ) -> StoredRecord:
        """Verify, scan, classify, sanitize, store.

        Ingest refuses a trajectory whose Experiment 1 chain does not verify.
        Analysing a tampered artifact would silently corrupt every downstream
        number, so the failure is loud.
        """
        run_id = str(doc.get("run_id", "unknown"))
        try:
            trajectory = Trajectory.model_validate(doc)
        except Exception as exc:  # schema drift is an error, not a warning
            raise IngestError(f"{run_id}: does not match the trajectory schema: {exc}") from exc

        ok, bad_index = verify_chain(trajectory.steps, trajectory.integrity.prev_run_hash)
        if not ok:
            raise IngestError(
                f"{run_id}: trajectory hash chain fails at step {bad_index}; refusing to ingest"
            )

        finding_set = scan_document(doc, self.settings)
        meta = classify(
            run_id,
            owner_principal=owner_principal,
            tenant_id=self.tenant_id,
            finding_set=finding_set,
        )
        sanitize_plan = plan(doc, finding_set, PROFILES[profile], self.key_context, tenant_id=self.tenant_id)

        from trajsec.vault import Vault

        vault = (
            Vault(self.root / "vault.db")
            if PROFILES[profile].vault
            else None
        )
        sanitized, report, replacement_map, produced = apply(
            doc,
            sanitize_plan,
            self.key_context,
            policy_version=self.settings.policy_version,
            vault=vault,
        )
        if vault is not None:
            vault.close()

        for decision in sanitize_plan.decisions:
            if decision.action in ("tokenized", "placeholder", "suppressed"):
                self.audit.record_sanitize(
                    decision.audit_fields(), run_id, self.settings.policy_version
                )

        record = StoredRecord(
            run_id=run_id,
            raw=doc,
            sanitized=sanitized,
            meta=meta,
            finding_set=finding_set,
            replacement_map=replacement_map.model_dump(mode="json"),
            produced_token_hashes=produced,
            tenant_id=self.tenant_id,
            published=published,
        )
        self.records[run_id] = record

        self._write(run_id, "raw", doc)
        self._write(run_id, "sanitized", sanitized)
        self._write(run_id, "findings", finding_set.model_dump(mode="json"))
        self._write(run_id, "replacement_map", record.replacement_map)
        return record

    def _write(self, run_id: str, kind: str, payload: Any) -> None:
        target = self.root / run_id / f"{kind}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- read paths --------------------------------------------------------

    def _authorize(self, subject: Subject, record: StoredRecord, action: str) -> None:
        """Verify the session, evaluate the policy, audit the outcome, and raise
        on denial.

        The audit entry is written on denials too. A log that only records
        successful reads cannot answer the question an auditor actually asks,
        which is who tried to reach what and was refused.
        """
        if not verify_subject(subject):
            self.audit.append(
                action=action,
                decision="deny",
                reason_code="invalid_session",
                policy_version=self.settings.policy_version,
                principal_id=subject.principal_id,
                role=subject.role,
                session_id=subject.session_id,
                run_id=record.run_id,
            )
            raise PermissionError("invalid_session (policy.session.signature)")

        decision = evaluate(
            subject,
            record.meta,
            action,
            finding_set=record.finding_set,
            published=record.published,
        )
        self.audit.record_decision(subject, decision, self.settings.policy_version)
        if not decision.allow:
            raise PermissionError(f"{decision.reason_code} ({decision.policy_rule_id})")

    def get_raw(self, run_id: str, subject: Subject) -> dict[str, Any]:
        """Record-level denial raises; field-level denial projects."""
        record = self._require(run_id)
        self._authorize(subject, record, "read_payload_raw")
        return project(record.raw, subject, record.meta, sanitized=False)

    def get_sanitized(self, run_id: str, subject: Subject) -> dict[str, Any]:
        record = self._require(run_id)
        self._authorize(subject, record, "read_payload_sanitized")
        return project(record.sanitized, subject, record.meta, sanitized=True)

    def export_for_third_party(self, run_id: str, subject: Subject) -> dict[str, Any]:
        """Sanitize on write: the recipient sees no redaction markers at all.

        Even the existence and count of redactions is metadata, so the
        untrusted path must not expose it -- which is why this returns the
        stored sanitized artifact rather than a projection.
        """
        record = self._require(run_id)
        self._authorize(subject, record, "export_sanitized")
        return record.sanitized

    def findings_for(self, run_id: str, subject: Subject) -> list[dict[str, Any]]:
        from trajsec.policy import findings_summary

        record = self._require(run_id)
        self._authorize(subject, record, "read_findings")
        return findings_summary(
            [f for f in record.finding_set.findings if not f.suppressed], subject
        )

    def read_meta(self, run_id: str, subject: Subject) -> dict[str, Any]:
        record = self._require(run_id)
        self._authorize(subject, record, "read_meta")
        return record.meta.model_dump(mode="json")

    def try_action(self, run_id: str, subject: Subject, action: str) -> dict[str, Any]:
        """Uniform entry point used by the adversarial suite and the matrix."""
        record = self._require(run_id)
        try:
            self._authorize(subject, record, action)
            return {"allowed": True}
        except PermissionError as exc:
            return {"allowed": False, "error": str(exc)}

    def _require(self, run_id: str) -> StoredRecord:
        record = self.records.get(run_id)
        if record is None:
            raise KeyError(f"no such run: {run_id}")
        return record

    def publish(self, run_id: str) -> None:
        self._require(run_id).published = True

    def run_ids(self) -> list[str]:
        return sorted(self.records)
