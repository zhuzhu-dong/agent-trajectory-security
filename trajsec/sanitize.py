"""Sanitization.

Four principles, each of which is enforced here rather than asserted:

* **Minimum span.** Replace the smallest span that removes the secret and
  preserve the enclosing structure. A connection string becomes
  ``postgresql://appuser:<token>@db.internal:5432/prod``: the host, port, user
  and database are useful and are not secret, so destroying them is a
  measurable utility loss for no security gain.
* **Format preservation.** Tokens keep the length and character classes of
  what they replace, so downstream parsers keep working and every offset stays
  put.
* **No plaintext persistence.** Decisions reference a finding id, a path and a
  token. The plaintext exists only for the duration of the splice.
* **Low-entropy values are not tokenized by value.** A token derived from a
  six-character password is brute-forceable by anyone holding the key, so such
  values get a typed counter placeholder instead and correlation is given up
  on purpose.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from canonical import sha256_hex
from textindex import resolve_pointer

from trajsec.model import (
    Finding,
    FindingSet,
    Profile,
    PROFILES,
    Replacement,
    ReplacementMap,
    canonicalize_secret,
)
from trajsec.vault import (
    KeyContext,
    Vault,
    derive_digest,
    format_preserving_token,
    is_low_entropy,
    placeholder_token,
    scope_identifier,
)


@dataclass
class Decision:
    finding_id: str
    pointer: str
    char_start: int
    char_end: int
    type: str
    layer: str
    action: str  # tokenized | placeholder | parent_covered | suppressed
    token_id: str | None = None
    token_value: str | None = None
    reason: str = ""

    def audit_fields(self) -> dict[str, Any]:
        """What may be written to the audit log. Never the plaintext."""
        return {
            "finding_id": self.finding_id,
            "json_path": self.pointer,
            "type": self.type,
            "layer": self.layer,
            "decision": self.action,
            "token_id": self.token_id,
        }


@dataclass
class SanitizePlan:
    run_id: str
    profile: Profile
    tenant_id: str
    transform_id: str
    decisions: list[Decision] = field(default_factory=list)

    @property
    def tokenized(self) -> list[Decision]:
        return [d for d in self.decisions if d.action in ("tokenized", "placeholder")]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.action] = counts.get(decision.action, 0) + 1
        return counts


@dataclass
class SanitizeReport:
    run_id: str
    profile: str
    transform_id: str
    policy_version: str
    decisions: int
    tokens: int
    bytes_changed: int
    bytes_total: int
    suppression_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def destruction_footprint(self) -> float:
        """Fraction of the artifact rewritten. The T7 over-sanitization metric."""
        return round(self.bytes_changed / self.bytes_total, 6) if self.bytes_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "destruction_footprint": self.destruction_footprint}


def plan(
    doc: dict[str, Any],
    finding_set: FindingSet,
    profile: Profile | str,
    key_context: KeyContext,
    *,
    tenant_id: str = "tenant-a",
) -> SanitizePlan:
    """Decide, per finding, what should happen. Writes nothing and returns
    nothing secret."""
    if isinstance(profile, str):
        profile = PROFILES[profile]

    run_id = str(doc.get("run_id", "unknown"))
    scope_id = scope_identifier(profile.token_scope, run_id=run_id, tenant_id=tenant_id)
    transform_id = uuid.uuid4().hex[:16]

    # A finding that contains another finding is replaced only through its
    # child: that is what minimum-span means in practice.
    parents = {
        f.evidence.parent_finding_id
        for f in finding_set.active
        if f.evidence.parent_finding_id
    }

    # Suppressed findings are recorded so the audit log shows a decision was
    # made, not that the span was overlooked.
    decisions: list[Decision] = [
        Decision(
            finding_id=f.finding_id,
            pointer=f.locator.pointer,
            char_start=f.locator.char_start,
            char_end=f.locator.char_end,
            type=f.type,
            layer=f.layer,
            action="suppressed",
            reason=f.suppressed_by or "allowlisted",
        )
        for f in finding_set.findings
        if f.suppressed
    ]

    counters: dict[str, int] = {}
    for finding in finding_set.active:
        if finding.finding_id in parents:
            decisions.append(
                Decision(
                    finding_id=finding.finding_id,
                    pointer=finding.locator.pointer,
                    char_start=finding.locator.char_start,
                    char_end=finding.locator.char_end,
                    type=finding.type,
                    layer=finding.layer,
                    action="parent_covered",
                    reason="a more specific child finding replaces the minimal span",
                )
            )
            continue

        plain_value = _value_at(doc, finding)
        if not plain_value:
            continue

        if _placeholder_required(plain_value, profile):
            counters[finding.type] = counters.get(finding.type, 0) + 1
            replacement = placeholder_token(finding.type, counters[finding.type])
            action = "placeholder"
            token_id = f"{finding.type}:{counters[finding.type]}"
        else:
            nonce = uuid.uuid4().hex if profile.token_scope == "none" else ""
            digest = derive_digest(key_context.key, scope_id, plain_value, nonce=nonce)
            if profile.token_form == "placeholder":
                counters[finding.type] = counters.get(finding.type, 0) + 1
                replacement = placeholder_token(finding.type, counters[finding.type])
                action = "placeholder"
                token_id = f"{finding.type}:{counters[finding.type]}"
            else:
                replacement = format_preserving_token(plain_value, digest)
                action = "tokenized"
                token_id = f"{finding.type}:{sha256_hex(digest)[:12]}"

        decisions.append(
            Decision(
                finding_id=finding.finding_id,
                pointer=finding.locator.pointer,
                char_start=finding.locator.char_start,
                char_end=finding.locator.char_end,
                type=finding.type,
                layer=finding.layer,
                action=action,
                token_id=token_id,
                token_value=replacement,
                reason="token is derived from the canonicalised value"
                if action == "tokenized"
                else "value is low-entropy, so the token is not derived from it",
            )
        )

    return SanitizePlan(
        run_id=run_id,
        profile=profile,
        tenant_id=tenant_id,
        transform_id=transform_id,
        decisions=decisions,
    )


def _placeholder_required(value: str, profile: Profile) -> bool:
    if profile.token_form == "placeholder":
        return True
    return is_low_entropy(value)


def _value_at(doc: dict[str, Any], finding: Finding) -> str:
    try:
        node = resolve_pointer(doc, finding.locator.pointer)
    except (KeyError, IndexError, ValueError):
        return ""
    if not isinstance(node, str):
        return ""
    return node[finding.locator.char_start : finding.locator.char_end]


def apply(
    doc: dict[str, Any],
    sanitize_plan: SanitizePlan,
    key_context: KeyContext,
    *,
    policy_version: str,
    vault: Vault | None = None,
) -> tuple[dict[str, Any], SanitizeReport, ReplacementMap, set[str]]:
    """Splice replacements in. Returns the sanitized document and its evidence.

    The fourth return value is the set of hashes of the tokens that were
    produced, so a later scan can recognise its own output. Without that the
    idempotence check would fail by construction: a format-preserving token
    still looks exactly like the kind of secret it replaced.
    """
    sanitized = copy.deepcopy(doc)
    replacements: list[Replacement] = []
    produced_hashes: set[str] = set()
    bytes_changed = 0

    by_pointer: dict[str, list[Decision]] = {}
    for decision in sanitize_plan.tokenized:
        by_pointer.setdefault(decision.pointer, []).append(decision)

    for pointer, decisions in by_pointer.items():
        try:
            original = resolve_pointer(sanitized, pointer)
        except (KeyError, IndexError, ValueError):
            continue
        if not isinstance(original, str):
            continue

        # Descending order so earlier offsets stay valid as we splice.
        text = original
        for decision in sorted(decisions, key=lambda d: d.char_start, reverse=True):
            start, end = decision.char_start, decision.char_end
            if not (0 <= start <= end <= len(text)):
                continue
            replacement = decision.token_value or ""
            bytes_changed += end - start
            text = text[:start] + replacement + text[end:]

            if vault is not None and decision.action == "tokenized":
                vault.put(decision.token_id or "", sanitize_plan.run_id, decision.type,
                          original[start:end])

            replacements.append(
                Replacement(
                    finding_id=decision.finding_id,
                    pointer=pointer,
                    token_id=decision.token_id or "",
                    original_span=(start, end),
                    new_span=(start, start + len(replacement)),
                    type=decision.type,
                    original_length=end - start,
                    new_length=len(replacement),
                )
            )
            produced_hashes.add(sha256_hex(canonicalize_secret(replacement)))

        _write_back(sanitized, pointer, text)

    # --- integrity bookkeeping --------------------------------------------
    #
    # Tokenizing necessarily invalidates the hash chain over the artifact's
    # bytes, so the sanitized copy is "tampered with" by construction. The
    # original chain hash is kept as immutable evidence of what was recorded,
    # and a fresh chain is computed over the sanitized steps so the copy is
    # internally consistent too.
    integrity = sanitized.setdefault("integrity", {})
    integrity["original_chain_hash"] = (doc.get("integrity") or {}).get("chain_hash")
    integrity["transform_id"] = sanitize_plan.transform_id
    integrity["policy_version"] = policy_version
    integrity["sanitized_chain_hash"] = _rechain(sanitized)

    total_bytes = len(str(doc).encode("utf-8"))
    report = SanitizeReport(
        run_id=sanitize_plan.run_id,
        profile=sanitize_plan.profile.name,
        transform_id=sanitize_plan.transform_id,
        policy_version=policy_version,
        decisions=len(sanitize_plan.decisions),
        tokens=len(replacements),
        bytes_changed=bytes_changed,
        bytes_total=total_bytes,
        suppression_reasons=_tally_reasons(sanitize_plan),
    )
    replacement_map = ReplacementMap(
        run_id=sanitize_plan.run_id,
        profile=sanitize_plan.profile.name,
        replacements=replacements,
        mode="redacted" if sanitize_plan.profile.vault else "scrubbed",
    )
    return sanitized, report, replacement_map, produced_hashes


def _write_back(doc: dict[str, Any], pointer: str, value: str) -> None:
    tokens = pointer.split("/")[1:]
    parent: Any = doc
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    last = tokens[-1]
    if isinstance(parent, list):
        parent[int(last)] = value
    else:
        parent[last] = value


def _rechain(sanitized: dict[str, Any]) -> str:
    """Recompute the step hash chain over the sanitized content."""
    from agentlab.schema import Step, build_chain

    steps = [Step.model_validate(step) for step in sanitized.get("steps", [])]
    return build_chain(steps, None)


def _tally_reasons(sanitize_plan: SanitizePlan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in sanitize_plan.decisions:
        if decision.action == "suppressed":
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
    return counts


def unmask(
    token_ids: list[str],
    vault: Vault,
    *,
    subject_role: str,
    purpose: str,
) -> dict[str, str]:
    """Reverse a token. Privileged, purpose-gated, and audited by the caller.

    Kept separate from every other path on purpose: an ordinary read must never
    be able to reach the plaintext.
    """
    if subject_role != "owner":
        raise PermissionError("only the record owner role may unmask, and only for its own tenant")
    if purpose != "incident_response":
        raise PermissionError("unmask requires purpose=incident_response")
    out: dict[str, str] = {}
    for token_id in token_ids:
        value = vault.get(token_id)
        if value is not None:
            out[token_id] = value
    return out
