"""Tamper-evident audit log.

Every access decision and every sanitizer decision becomes an entry. Entries
carry no plaintext, ever -- the log has to be safe to hand to an auditor
without becoming a second copy of what it is auditing.

The claim is deliberately bounded. A hash chain detects modification,
deletion, reordering, truncation and replay by a principal who can write
records but cannot write the anchor. It does **not** detect a forged append,
and it does not detect an attacker who rewrites the chain from genesis and
also rewrites the anchor. The tamper suite measures both sides rather than
asserting a property that cannot hold on one machine.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from canonical import canonical_json, chain_hash, sha256_hex

from trajsec.model import AccessDecision, AuditEntry


@dataclass
class AuditLog:
    path: Path
    store_salt: str
    tenant: str = "tenant-a"
    entries: list[AuditEntry] = field(default_factory=list)
    anchor_path: Path | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.anchor_path is None:
            self.anchor_path = self.path.with_suffix(".anchor")
        if self.path.is_file():
            self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        self.entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.entries.append(AuditEntry.model_validate_json(line))

    def _genesis(self) -> str:
        return sha256_hex(self.store_salt)

    def _prev_hash(self) -> str:
        return self.entries[-1].entry_hash if self.entries else self._genesis()

    # --- appending ---------------------------------------------------------

    def append(
        self,
        *,
        action: str,
        decision: str,
        reason_code: str,
        policy_version: str,
        principal_id: str = "system",
        role: str = "service",
        session_id: str = "",
        run_id: str = "",
        path: str | None = None,
        findings_ref: str | None = None,
        token_id: str | None = None,
        key_scope: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            seq=len(self.entries),
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tenant=self.tenant,
            actor_principal=principal_id,
            actor_role=role,
            actor_session=session_id,
            action=action,
            object_run_id=run_id,
            object_path=path,
            decision=decision,
            reason_code=reason_code,
            policy_version=policy_version,
            findings_ref=findings_ref,
            token_id=token_id,
            key_scope=key_scope,
            prev_hash=self._prev_hash(),
        )
        entry.entry_hash = self._hash_entry(entry)
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        return entry

    def record_decision(self, subject: Any, decision: AccessDecision, policy_version: str) -> AuditEntry:
        return self.append(
            action=decision.action,
            decision="allow" if decision.allow else "deny",
            reason_code=decision.reason_code,
            policy_version=policy_version,
            principal_id=subject.principal_id,
            role=subject.role,
            session_id=subject.session_id,
            run_id=decision.run_id,
        )

    def record_sanitize(self, decision_fields: dict[str, Any], run_id: str, policy_version: str) -> AuditEntry:
        return self.append(
            action="sanitize",
            decision=decision_fields.get("decision", "unknown"),
            reason_code="sanitizer",
            policy_version=policy_version,
            run_id=run_id,
            path=decision_fields.get("json_path"),
            findings_ref=decision_fields.get("finding_id"),
            token_id=decision_fields.get("token_id"),
            key_scope=decision_fields.get("layer"),
        )

    # --- verification ------------------------------------------------------

    @staticmethod
    def _hash_entry(entry: AuditEntry) -> str:
        data = entry.model_dump(mode="json")
        data.pop("entry_hash", None)
        return sha256_hex(canonical_json(data))

    def verify(self) -> tuple[bool, int | None, str]:
        """Recompute the chain. Returns ``(ok, first_bad_seq, reason)``."""
        previous = self._genesis()
        for index, entry in enumerate(self.entries):
            if entry.seq != index:
                return False, index, f"sequence gap: expected {index}, found {entry.seq}"
            if entry.prev_hash != previous:
                return False, index, "prev_hash does not match the preceding entry"
            if self._hash_entry(entry) != entry.entry_hash:
                return False, index, "entry hash does not match its contents"
            previous = entry.entry_hash

        if self.anchor_path and Path(self.anchor_path).is_file():
            anchored = json.loads(Path(self.anchor_path).read_text(encoding="utf-8"))
            if anchored.get("count") != len(self.entries):
                return False, len(self.entries), "entry count differs from the anchor"
            if anchored.get("merkle_root") != self.merkle_root():
                return False, len(self.entries), "merkle root differs from the anchor"
        return True, None, "ok"

    def merkle_root(self) -> str:
        if not self.entries:
            return sha256_hex("empty")
        level = [entry.entry_hash for entry in self.entries]
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [
                sha256_hex(level[i] + level[i + 1]) for i in range(0, len(level), 2)
            ]
        return level[0]

    def anchor(self) -> Path:
        """Write the root somewhere the record writer is not expected to reach.

        The honest framing: this detects tampering by a principal who can
        write the chain but not the anchor. It is not a defence against an
        attacker who has both.
        """
        assert self.anchor_path is not None
        Path(self.anchor_path).write_text(
            json.dumps(
                {
                    "count": len(self.entries),
                    "merkle_root": self.merkle_root(),
                    "anchored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "note": "detects tampering by a writer without anchor access; not a defence against one with it",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return Path(self.anchor_path)

    def read_for(self, subject: Any) -> list[dict[str, Any]]:
        """The auditor reads everything; the owner reads its own actions only."""
        if subject.role == "auditor":
            return [entry.model_dump(mode="json") for entry in self.entries]
        if subject.role == "owner":
            return [
                entry.model_dump(mode="json")
                for entry in self.entries
                if entry.actor_principal == subject.principal_id
            ]
        return []


# --- tamper suite ----------------------------------------------------------


def _write_entries(path: Path, entries: Iterable[AuditEntry]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")


def tamper_suite(
    seed_entries: list[AuditEntry], workdir: Path, *, store_salt: str
) -> dict[str, Any]:
    """Apply each tamper operation and record whether the chain noticed.

    Reported as a detection rate rather than a claim of tamper-proofing. The
    forged-append case is included precisely because it is *not* detected, and
    the report says so.

    ``store_salt`` must be the salt of the log the entries came from. Using a
    different one makes the very first link fail to verify, which looks like a
    detection but actually tests nothing.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    def run(name: str, mutate: Any, expect_detected: bool, note: str = "") -> None:
        path = workdir / f"tamper_{name}.jsonl"
        _write_entries(path, seed_entries)
        anchor_path = workdir / f"tamper_{name}.anchor"
        log = AuditLog(path=path, store_salt=store_salt, anchor_path=anchor_path)
        log.anchor()
        mutate(path, log)
        reloaded = AuditLog(path=path, store_salt=store_salt, anchor_path=anchor_path)
        ok, bad_seq, reason = reloaded.verify()
        results.append(
            {
                "operation": name,
                "detected": not ok,
                "expect_detected": expect_detected,
                "first_bad_seq": bad_seq,
                "reason": reason,
                "note": note,
            }
        )

    def modify(path: Path, _log: AuditLog) -> None:
        entries = [AuditEntry.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries[1].decision = "allow"
        _write_entries(path, entries)

    def delete(path: Path, _log: AuditLog) -> None:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        del lines[1]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def reorder(path: Path, _log: AuditLog) -> None:
        entries = [AuditEntry.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries[1], entries[2] = entries[2], entries[1]
        _write_entries(path, entries)

    def truncate(path: Path, _log: AuditLog) -> None:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    def duplicate(path: Path, _log: AuditLog) -> None:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines.insert(1, lines[1])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def forge_append(path: Path, log: AuditLog) -> None:
        """Append a well-formed entry the chain cannot distinguish from a real one."""
        forged = seed_entries[-1].model_copy(deep=True)
        forged.seq = len(seed_entries)
        forged.prev_hash = seed_entries[-1].entry_hash
        forged.action = "read_payload_raw"
        forged.decision = "allow"
        forged.entry_hash = AuditLog._hash_entry(forged)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(forged.model_dump_json() + "\n")
        # Re-anchor, as an attacker with anchor access would.
        reloaded = AuditLog(path=path, store_salt=store_salt, anchor_path=log.anchor_path)
        reloaded.anchor()

    run("MODIFY_FIELD", modify, True)
    run("DELETE_ENTRY", delete, True)
    run("REORDER", reorder, True)
    run("TRUNCATE_TAIL", truncate, True)
    run("REPLAY_DUPLICATE", duplicate, True)
    run(
        "FORGE_APPEND",
        forge_append,
        False,
        "a well-formed append is indistinguishable from a real one by chain alone; "
        "only an anchor the attacker cannot rewrite would catch it",
    )

    detected = sum(1 for r in results if r["detected"])
    return {
        "operations": results,
        "detected": detected,
        "total": len(results),
        "detection_rate": round(detected / len(results), 4) if results else 0.0,
        "claim": (
            "Detects tampering by a principal that can write records but not the "
            "anchor. Does not detect a forged append, and does not detect an "
            "attacker who rewrites the chain from genesis and the anchor together."
        ),
    }
