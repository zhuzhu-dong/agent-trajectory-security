"""Types for Experiment 2.

The sensitive-data taxonomy lives here because it is shared by the detector
(what to look for), the sanitizer (how to replace it) and the policy engine
(how to classify a record). Keeping one definition means a type cannot be
detected as one thing and classified as another.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trajsec.config import LEVELS, max_level

Severity = Literal["Critical", "High", "Medium", "Low"]
DetectorLayer = Literal["L1", "L2", "L3", "L4"]


class StrictModel(BaseModel):
    """Reject unknown fields, so a typo cannot silently drop evidence."""

    model_config = ConfigDict(extra="forbid")


# --- taxonomy --------------------------------------------------------------


class SensitiveType(StrictModel):
    id: str
    severity: Severity
    classification: str
    description: str
    layers: list[DetectorLayer] = Field(default_factory=list)


TAXONOMY: dict[str, SensitiveType] = {
    t.id: t
    for t in (
        SensitiveType(
            id="SECRET_API_KEY",
            severity="Critical",
            classification="SECRET",
            description="Provider API key with a recognisable prefix (sk-, ghp_, xoxb-, ...).",
            layers=["L1", "L2"],
        ),
        SensitiveType(
            id="SECRET_CLOUD_CREDENTIAL",
            severity="Critical",
            classification="SECRET",
            description="Cloud access key id / secret pair, e.g. AWS AKIA + secret.",
            layers=["L1", "L2"],
        ),
        SensitiveType(
            id="CRYPTO_MATERIAL",
            severity="Critical",
            classification="SECRET",
            description="Private key blocks and SSH key material.",
            layers=["L1"],
        ),
        SensitiveType(
            id="SECRET_JWT",
            severity="High",
            classification="SECRET",
            description="JSON Web Token: three base64url segments with a decodable header.",
            layers=["L1"],
        ),
        SensitiveType(
            id="SECRET_BEARER",
            severity="High",
            classification="SECRET",
            description="Authorization: Bearer <token>.",
            layers=["L1", "L2"],
        ),
        SensitiveType(
            id="SECRET_DSN_PASSWORD",
            severity="High",
            classification="SECRET",
            description="Password component of a connection string.",
            layers=["L2"],
        ),
        SensitiveType(
            id="SECRET_DSN",
            severity="High",
            classification="CONFIDENTIAL",
            description="Connection string; the enclosing structure around a DSN password.",
            layers=["L1", "L2"],
        ),
        SensitiveType(
            id="SECRET_GENERIC_ASSIGNMENT",
            severity="High",
            classification="CONFIDENTIAL",
            description="KEY=VALUE whose key name marks it as credential-bearing.",
            layers=["L2", "L3"],
        ),
        SensitiveType(
            id="SECRET_ENTROPIC_TOKEN",
            severity="Medium",
            classification="CONFIDENTIAL",
            description="High-entropy token of unknown format; a guess, not a certainty.",
            layers=["L3"],
        ),
        SensitiveType(
            id="PII_PATH_USERNAME",
            severity="Low",
            classification="INTERNAL",
            description="Absolute path that discloses the account name.",
            layers=["L2"],
        ),
        SensitiveType(
            id="PII_HOSTNAME_MACHINEID",
            severity="Low",
            classification="INTERNAL",
            description="Hostname or machine identifier.",
            layers=["L2"],
        ),
        SensitiveType(
            id="PII_EMAIL",
            severity="Medium",
            classification="CONFIDENTIAL",
            description="Email address in tool output.",
            layers=["L1"],
        ),
        SensitiveType(
            id="INTERNAL_TOPOLOGY",
            severity="Low",
            classification="INTERNAL",
            description="Internal hostnames, private IP ranges.",
            layers=["L1", "L2"],
        ),
        SensitiveType(
            id="ABUSE_INJECTION_MARKER",
            severity="High",
            classification="CONFIDENTIAL",
            description="Text inside untrusted content that addresses an automated agent.",
            layers=["L2"],
        ),
    )
}

# Types that escalate when they appear in untrusted content. This is the
# taint rule: a credential the agent fetched because a file told it to is a
# different event from one the user asked for.
TAINT_ESCALATED_TYPES = frozenset(
    {"SECRET_API_KEY", "SECRET_CLOUD_CREDENTIAL", "CRYPTO_MATERIAL", "SECRET_JWT"}
)


def classification_for_types(type_ids: list[str]) -> str:
    """Lattice join of the classifications of the given finding types."""
    levels = ["PUBLIC"]
    for type_id in type_ids:
        entry = TAXONOMY.get(type_id)
        if entry is not None:
            levels.append(entry.classification)
    return max_level(*levels)


def severity_for(type_id: str) -> str:
    entry = TAXONOMY.get(type_id)
    return entry.severity if entry else "Low"


def canonicalize_secret(value: str) -> str:
    """Normalise a matched value before hashing or tokenizing it.

    Without this, ``KEY=abc`` and ``KEY="abc"`` produce different tokens, so
    the same secret stops correlating with itself and the correlation metrics
    silently read as failures. Also normalises line endings and Unicode
    composition, both of which differ between a file read and a shell dump.
    """
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "`"):
        text = text[1:-1]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


# --- findings --------------------------------------------------------------


class Locator(StrictModel):
    """A span in the decoded logical string at a JSON Pointer."""

    pointer: str
    char_start: int
    char_end: int
    line: int = 0
    col: int = 0


class Evidence(StrictModel):
    rule_id: str
    rule_version: str = "1"
    entropy: float | None = None
    entropy_normalised: float | None = None
    alphabet_class: str | None = None
    key_name: str | None = None
    decoded_from: str | None = None
    parent_finding_id: str | None = None
    taint: bool = False


class Finding(StrictModel):
    """A detected span.

    Load-bearing property: this record never contains the matched plaintext.
    Not in the store, not in the audit log, not in the report. ``value_sha256``
    gives correlation and ``preview`` gives a human something to triage with,
    and neither is reversible into the secret.
    """

    finding_id: str
    run_id: str
    trace_id: str = ""
    step_index: int = -1
    json_path: str = ""
    locator: Locator
    type: str
    layer: DetectorLayer
    confidence: float
    severity: str
    classification: str
    value_sha256: str
    value_length: int = 0
    preview: str | None = None
    evidence: Evidence
    tainted: bool = False
    suppressed_by: str | None = None
    ts_created: str = ""

    @property
    def suppressed(self) -> bool:
        return self.suppressed_by is not None

    @property
    def active(self) -> bool:
        return not self.suppressed


class FindingSet(StrictModel):
    run_id: str
    findings: list[Finding] = Field(default_factory=list)
    layers_run: list[str] = Field(default_factory=list)
    rules_fingerprint: str = ""
    thresholds_version: str = ""

    @property
    def active(self) -> list[Finding]:
        return [f for f in self.findings if f.active]

    @property
    def suppressed(self) -> list[Finding]:
        return [f for f in self.findings if f.suppressed]

    def types(self) -> list[str]:
        return sorted({f.type for f in self.active})

    def classification(self) -> str:
        return classification_for_types(self.types())


# --- access control --------------------------------------------------------

Role = Literal["owner", "analyst", "auditor", "third_party", "service", "admin"]
Purpose = Literal["debug", "incident_response", "metrics", "compliance", "publication"]
Action = Literal[
    "read_meta",
    "read_payload_raw",
    "read_payload_sanitized",
    "read_findings",
    "read_finding_evidence",
    "unmask",
    "export_sanitized",
    "export_raw",
    "replay",
    "resume",
    "write_append",
    "delete",
    "verify_integrity",
    "read_audit",
    "admin_policy",
]

ACTIONS: tuple[str, ...] = (
    "read_meta",
    "read_payload_raw",
    "read_payload_sanitized",
    "read_findings",
    "read_finding_evidence",
    "unmask",
    "export_sanitized",
    "export_raw",
    "replay",
    "resume",
    "write_append",
    "delete",
    "verify_integrity",
    "read_audit",
    "admin_policy",
)


class Subject(StrictModel):
    principal_id: str
    role: Role
    tenant_id: str
    clearance: str = "INTERNAL"
    purpose: Purpose = "debug"
    session_id: str = ""
    claims_sig: str = ""


class RecordMeta(StrictModel):
    run_id: str
    owner_principal: str
    tenant_id: str
    base_classification: str = "INTERNAL"
    classification: str = "INTERNAL"
    finding_types: list[str] = Field(default_factory=list)
    max_untrusted: bool = False


class AccessDecision(StrictModel):
    allow: bool
    reason_code: str
    policy_rule_id: str
    action: str
    run_id: str
    principal_id: str
    projected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# --- sanitization ----------------------------------------------------------

TokenScope = Literal["global", "tenant", "run", "none"]
TokenForm = Literal["format_preserving", "placeholder"]


class Profile(StrictModel):
    """A sanitization posture. Note this is a *policy* distinction.

    On one machine both profiles are reversible by anyone holding the key.
    ``redacted`` is reversible because a vault exists and is access
    controlled; ``scrubbed`` is one-way because no vault is ever written.
    That is policy separation, not a cryptographic guarantee, and the report
    says so.
    """

    name: str
    token_scope: TokenScope
    token_form: TokenForm
    length_preserving: bool
    vault: bool
    egress_allowed: bool
    include_preview: bool


PROFILES: dict[str, Profile] = {
    "redacted": Profile(
        name="redacted",
        token_scope="tenant",
        token_form="format_preserving",
        length_preserving=True,
        vault=True,
        egress_allowed=False,
        include_preview=True,
    ),
    "scrubbed": Profile(
        name="scrubbed",
        token_scope="run",
        token_form="format_preserving",
        length_preserving=True,
        vault=False,
        egress_allowed=True,
        include_preview=False,
    ),
    "internal_only": Profile(
        name="internal_only",
        token_scope="run",
        token_form="placeholder",
        length_preserving=False,
        vault=False,
        egress_allowed=False,
        include_preview=False,
    ),
}


class Replacement(StrictModel):
    finding_id: str
    pointer: str
    token_id: str
    original_span: tuple[int, int]
    new_span: tuple[int, int]
    type: str
    original_length: int
    new_length: int


class ReplacementMap(StrictModel):
    run_id: str
    profile: str
    replacements: list[Replacement] = Field(default_factory=list)
    mode: str = "redacted"

    @property
    def identical_length(self) -> bool:
        return all(r.original_length == r.new_length for r in self.replacements)


# --- audit -----------------------------------------------------------------


class AuditEntry(StrictModel):
    seq: int
    ts: str
    tenant: str
    actor_principal: str
    actor_role: str
    actor_session: str = ""
    action: str
    object_run_id: str = ""
    object_path: str | None = None
    decision: str
    reason_code: str
    policy_version: str
    findings_ref: str | None = None
    token_id: str | None = None
    key_scope: str | None = None
    prev_hash: str
    entry_hash: str = ""


# --- ground truth ----------------------------------------------------------


class GroundTruthItem(StrictModel):
    run_id: str
    pointer: str
    char_start: int
    char_end: int
    type: str
    value_sha256: str
    tier: str
    evasion: str = "plain"
    injected: bool = True
    note: str = ""


class Manifest(StrictModel):
    """Ground truth. It names where every secret sits, so it is itself
    sensitive and lives under the same access control as raw trajectories;
    only its hash is published."""

    seed: int
    schema_version: str
    items: list[GroundTruthItem] = Field(default_factory=list)
    negative_controls: list[dict[str, Any]] = Field(default_factory=list)
    calibration_runs: list[str] = Field(default_factory=list)
    test_runs: list[str] = Field(default_factory=list)


@dataclass
class CorpusEntry:
    run_id: str
    document: dict[str, Any]
    manifest: Manifest
    split: str
    # value_sha256 -> the injected plaintext, in memory only. It exists so the
    # cascading-failure test can ask whether a value survived sanitization at
    # all. It is never serialized: the manifest stores hashes for that reason.
    plaintext_lookup: dict[str, str] = field(default_factory=dict)
