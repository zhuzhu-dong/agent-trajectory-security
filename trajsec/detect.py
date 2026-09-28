"""The layered detector.

Layer order is ``L4-deny -> L2 -> L1 -> L3 -> L4-allow``. Suppression *marks*
a finding rather than deleting it, so ablation can re-enable any layer and
measure the difference.

Three design points carry most of the weight:

* **L2 exists for values no content rule can find.** The account name in this
  environment is a single character. It appears in every path in every
  trajectory and no format regex or entropy score will ever flag it. Only a
  structural rule can, and that is the concrete case for context-awareness.
* **L3 normalises entropy by alphabet class.** Raw Shannon entropy is not
  comparable across token shapes: 20 characters of hex cap out around 4 bits,
  base64 around 6, and a long English sentence reaches ~4.2. Any fixed raw
  threshold therefore misses hex secrets while flooding on prose.
* **L4 is a false-positive control, not a nicety.** It must suppress the
  trajectory's own hash-chain fields -- 64 hex characters, entropically
  indistinguishable from a key, and not a key.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import string
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from canonical import sha256_hex
from textindex import TextIndex, index_document, iter_leaves

from trajsec.config import Settings, load_settings
from trajsec.model import (
    TAXONOMY,
    TAINT_ESCALATED_TYPES,
    Evidence,
    Finding,
    FindingSet,
    Locator,
    canonicalize_secret,
    severity_for,
)

# --- rules loading ---------------------------------------------------------


def _rules_path(name: str):
    from trajsec.config import RULES_DIR

    return RULES_DIR / name


def load_json_rule(name: str) -> dict[str, Any]:
    path = _rules_path(name)
    return json.loads(path.read_text(encoding="utf-8"))


def rules_fingerprint() -> str:
    """Content hash of every rule file, so a result can name its rule version."""
    from trajsec.config import RULES_DIR

    parts = []
    for path in sorted(RULES_DIR.glob("*.json")):
        parts.append(f"{path.name}:{sha256_hex(path.read_bytes())}")
    return sha256_hex("|".join(parts))


# --- structural validators -------------------------------------------------


def _validator_jwt_header(value: str) -> bool:
    """A JWT's header segment must base64url-decode to JSON carrying ``alg``."""
    header = value.split(".", 1)[0]
    padding = "=" * (-len(header) % 4)
    try:
        decoded = base64.urlsafe_b64decode(header + padding)
        parsed = json.loads(decoded)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False
    return isinstance(parsed, dict) and "alg" in parsed


def _validator_dsn_has_password(value: str) -> bool:
    """A connection string only matters if it actually carries a password."""
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return False
    return parts.password is not None and len(parts.password) > 0


def _validator_pem_pair(value: str) -> bool:
    begins = re.findall(r"-----BEGIN ([A-Z ]*PRIVATE KEY)-----", value)
    ends = re.findall(r"-----END ([A-Z ]*PRIVATE KEY)-----", value)
    return bool(begins) and sorted(begins) == sorted(ends)


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "jwt_header": _validator_jwt_header,
    "dsn_has_password": _validator_dsn_has_password,
    "pem_pair": _validator_pem_pair,
}


# --- text scoring ----------------------------------------------------------

_ALPHABET_CLASSES: tuple[tuple[str, str], ...] = (
    ("hex", string.hexdigits),
    ("digits", string.digits),
    ("lower", string.ascii_lowercase),
    ("upper", string.ascii_uppercase),
    ("alnum", string.ascii_letters + string.digits),
    ("base64", string.ascii_letters + string.digits + "+/=_-"),
)


def alphabet_class(token: str) -> str | None:
    """The narrowest known alphabet that contains every character."""
    for name, alphabet in _ALPHABET_CLASSES:
        if set(token) <= set(alphabet):
            return name
    return None


def shannon_entropy(text: str) -> float:
    """Bits of entropy per character."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def normalised_entropy(text: str) -> tuple[float, float, str | None]:
    """Return ``(raw_bits_per_char, normalised_0_to_1, alphabet_class)``.

    Normalisation divides by the maximum achievable entropy for the token's
    alphabet, which is what makes the score comparable between a hex string
    and a sentence.
    """
    raw = shannon_entropy(text)
    cls = alphabet_class(text)
    if cls is None:
        return raw, 0.0, None
    alphabet_size = dict(_ALPHABET_CLASSES)[cls]
    ceiling = math.log2(len(set(alphabet_size)))
    if ceiling <= 0:
        return raw, 0.0, cls
    return raw, min(raw / ceiling, 1.0), cls


_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9+/=_.~:-]+")

# `=` is base64 padding and only ever trails. Allowing it mid-token would glue
# `DEEPSEEK_API_KEY=sk-abc...` into one token and score the assignment line as
# a single high-entropy blob, which is both wrong and noisy.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+/_.~:-]{12,}={0,2}")


def tokenize(text: str) -> Iterable[tuple[str, int]]:
    """Yield ``(token, offset)`` pairs. Scored per token, never per line."""
    for match in re.finditer(_TOKEN_PATTERN, text):
        yield match.group(0).rstrip("="), match.start()


def looks_like_word(token: str) -> bool:
    """Decide whether a token is an identifier or prose rather than a secret.

    The first version of this used a vowel-ratio test on the raw token and
    rejected roughly half of all random alphanumeric secrets, because a random
    string with few digits lands in the same vowel band as a word. The fix is
    to require that the token be *almost entirely letters* first: identifiers
    like ``getUserAccountBalanceHistoryForTenant`` and
    ``billing_sync_worker_queue_production`` are all letters once separators
    are removed, while a random secret essentially always carries digits.

    Entropy alone cannot make this call, which is why it is a separate check
    rather than a threshold.
    """
    letters = [c for c in token if c.isalpha()]
    core = token.replace("_", "").replace("-", "")
    if not core:
        return True
    letter_fraction = len(letters) / len(core)
    if letter_fraction < 0.9:
        return False  # carries digits or symbols: not word-shaped
    vowels = sum(1 for c in letters if c.lower() in "aeiou")
    ratio = vowels / len(letters)
    if 0.15 <= ratio <= 0.65:
        return True
    # Long runs of one character, or very low diversity, are not secrets either.
    return len(set(token)) / len(token) < 0.35


def make_preview(value: str) -> str:
    """A masked, non-reversible hint for human triage."""
    text = canonicalize_secret(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-3:]}"


# --- intermediate finding --------------------------------------------------


@dataclass
class Partial:
    pointer: str
    json_path: str
    start: int
    end: int
    type: str
    layer: str
    confidence: float
    rule_id: str
    value: str
    key_name: str | None = None
    entropy: float | None = None
    entropy_normalised: float | None = None
    alphabet: str | None = None
    decoded_from: str | None = None
    tainted: bool = False


@dataclass
class ScanContext:
    run_id: str
    trace_id: str
    settings: Settings
    provenance: dict[str, tuple[int, bool]] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    document: dict[str, Any] = field(default_factory=dict)
    indexes: dict[str, TextIndex] = field(default_factory=dict)

    def is_untrusted(self, pointer: str) -> bool:
        return self.provenance.get(pointer, (-1, False))[1]

    def step_index(self, pointer: str) -> int:
        return self.provenance.get(pointer, (-1, False))[0]

    def confidence_for(self, layer: str) -> float:
        return float(self.settings.thresholds["confidence"].get(layer, 0.5))

    def text_at(self, pointer: str) -> str:
        """The decoded logical string at a pointer. Empty if not a string leaf."""
        index = self.indexes.get(pointer)
        if index is not None:
            return index.text
        try:
            from textindex import resolve_pointer

            value = resolve_pointer(self.document, pointer)
        except (KeyError, IndexError, ValueError):
            return ""
        return value if isinstance(value, str) else ""


def build_provenance(doc: dict[str, Any]) -> dict[str, tuple[int, bool]]:
    """Map every string pointer to ``(step_index, untrusted)``."""
    out: dict[str, tuple[int, bool]] = {}
    steps = doc.get("steps") or []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        untrusted = bool((step.get("provenance") or {}).get("untrusted", False))
        base = f"/steps/{index}"
        out[base] = (index, untrusted)
        for pointer, _path, value in iter_leaves(step, base_pointer=base, base_path="$"):
            if isinstance(value, str):
                out[pointer] = (index, untrusted)
    # Top-level configuration is trusted by origin: it was not fetched.
    for pointer, _path, value in iter_leaves(doc, base_pointer="", base_path="$"):
        if isinstance(value, str) and pointer not in out:
            out[pointer] = (-1, False)
    return out


# --- layers ----------------------------------------------------------------


class L1FormatLayer:
    """Anchored patterns with structural validation. High precision, low recall."""

    id = "L1"

    def __init__(self, rules: dict[str, Any]) -> None:
        self.version = str(rules.get("version", "1"))
        self.compiled: list[tuple[dict[str, Any], re.Pattern[str]]] = [
            (rule, re.compile(rule["pattern"])) for rule in rules["rules"]
        ]

    def scan_text(self, text: str, pointer: str, json_path: str, ctx: ScanContext) -> list[Partial]:
        found: list[Partial] = []
        for rule, pattern in self.compiled:
            validator = VALIDATORS.get(rule.get("validator") or "")
            group = int(rule.get("group", 0))
            for match in pattern.finditer(text):
                try:
                    value = match.group(group)
                except IndexError:
                    continue
                if not value:
                    continue
                if validator is not None and not validator(value):
                    continue
                start, end = match.span(group)
                found.append(
                    Partial(
                        pointer=pointer,
                        json_path=json_path,
                        start=start,
                        end=end,
                        type=rule["type"],
                        layer=self.id,
                        confidence=ctx.confidence_for(self.id),
                        rule_id=f"{rule['id']}@{self.version}",
                        value=value,
                    )
                )
        return found


class L2ContextLayer:
    """Path-, key- and structure-aware rules.

    Runs the sub-parsers that turn "entropy on a blob" into "assignment
    detection", which is where most of the precision comes from.
    """

    id = "L2"

    def __init__(self, rules: dict[str, Any]) -> None:
        self.version = str(rules.get("version", "1"))
        self.sensitive_key = re.compile(rules["sensitive_key_names"])
        self.key_allow = re.compile(rules["sensitive_key_allowlist"])
        self.context_keys = {k.lower() for k in rules.get("context_key_allowlist", [])}
        self.path_policies = rules.get("path_policies", [])
        self.injection = [re.compile(p) for p in rules.get("injection_markers", [])]
        self.env_keys = set(rules.get("environment_dump_keys", []))
        self.dsn_re = re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|amqps|mssql|smtp|ftps?|ldaps?)://[^\s\"'<>\\)\]]+"
        )

    # -- path policy --------------------------------------------------------

    def path_policy(self, pointer: str) -> dict[str, Any] | None:
        for policy in self.path_policies:
            prefix = policy["pointer_prefix"]
            if pointer == prefix or pointer.startswith(prefix + "/"):
                return policy
        return None

    # -- key-name rule ------------------------------------------------------

    def key_name_findings(
        self, key: str | None, text: str, pointer: str, json_path: str, ctx: ScanContext
    ) -> list[Partial]:
        if not key or not self.sensitive_key.search(key) or self.key_allow.search(key):
            return []
        stripped = text.strip()
        if not stripped or stripped in {"", "null", "none"}:
            return []
        return [
            Partial(
                pointer=pointer,
                json_path=json_path,
                start=0,
                end=len(text),
                type="SECRET_GENERIC_ASSIGNMENT",
                layer=self.id,
                confidence=ctx.confidence_for(self.id),
                rule_id=f"l2.keyname.{key.lower()}@{self.version}",
                value=text,
                key_name=key,
            )
        ]

    # -- sub-parsers --------------------------------------------------------

    def assignment_findings(
        self, text: str, pointer: str, json_path: str, ctx: ScanContext
    ) -> list[Partial]:
        """Parse dotenv / `set` shaped text as assignments, not as a blob.

        One finding per credential-bearing variable, with the offset of the
        value inside the containing string.

        The key may begin mid-line, not only at the start: a credential inside
        a sentence or a log line is still a credential, and anchoring to ``^``
        silently missed every one of those. The key name must still match a
        sensitive pattern, which is what keeps prose out.
        """
        out: list[Partial] = []
        for match in re.finditer(
            r"(?m)(?:^|(?<=[\s,;:(\[]))(?:export[ \t]+)?"
            r"([A-Za-z_][A-Za-z0-9_]{1,64})[ \t]*=[ \t]*(\S[^\r\n]*)",
            text,
        ):
            key, raw_value = match.group(1), match.group(2).rstrip()
            if not self.sensitive_key.search(key) or self.key_allow.search(key):
                continue
            if not raw_value:
                continue
            value = canonicalize_secret(raw_value)
            if len(value) < 4:
                continue
            value_offset = match.start(2)
            out.append(
                Partial(
                    pointer=pointer,
                    json_path=json_path,
                    start=value_offset,
                    end=value_offset + len(raw_value),
                    type="SECRET_GENERIC_ASSIGNMENT",
                    layer=self.id,
                    confidence=ctx.confidence_for(self.id),
                    rule_id=f"l2.assignment.{key.lower()}@{self.version}",
                    value=raw_value,
                    key_name=key,
                )
            )
        return out

    def dsn_password_findings(
        self, text: str, pointer: str, json_path: str, ctx: ScanContext
    ) -> list[Partial]:
        """The password inside a connection string, as its own span.

        Emitted separately from the enclosing DSN so the sanitizer can replace
        the minimum span and leave host, port and database intact.
        """
        out: list[Partial] = []
        for match in self.dsn_re.finditer(text):
            dsn = match.group(0)
            try:
                parts = urllib.parse.urlsplit(dsn)
            except ValueError:
                continue
            if not parts.password:
                continue
            scheme_end = dsn.find("://") + 3
            userinfo_end = dsn.find("@", scheme_end)
            if userinfo_end < 0:
                continue
            colon = dsn.find(":", scheme_end, userinfo_end)
            if colon < 0:
                continue
            # urlsplit may percent-decode; locate the raw form instead.
            raw_password = dsn[colon + 1 : userinfo_end]
            if not raw_password:
                continue
            out.append(
                Partial(
                    pointer=pointer,
                    json_path=json_path,
                    start=match.start() + colon + 1,
                    end=match.start() + userinfo_end,
                    type="SECRET_DSN_PASSWORD",
                    layer=self.id,
                    confidence=ctx.confidence_for(self.id),
                    rule_id=f"l2.dsn_password@{self.version}",
                    value=raw_password,
                    key_name="password",
                )
            )
        return out

    def injection_findings(
        self, text: str, pointer: str, json_path: str, ctx: ScanContext
    ) -> list[Partial]:
        """Text in untrusted content that addresses an automated agent.

        This is a security event rather than a secret, and it is what makes
        the prompt-injection threat measurable instead of asserted.
        """
        if not ctx.is_untrusted(pointer):
            return []
        out: list[Partial] = []
        for pattern in self.injection:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                out.append(
                    Partial(
                        pointer=pointer,
                        json_path=json_path,
                        start=match.start(),
                        end=match.end(),
                        type="ABUSE_INJECTION_MARKER",
                        layer=self.id,
                        confidence=0.9,
                        rule_id=f"l2.injection@{self.version}",
                        value=match.group(0),
                        key_name=None,
                    )
                )
                _ = (start, end)
        return out

    # -- re-encoding --------------------------------------------------------

    def decoded_findings(
        self, text: str, pointer: str, json_path: str, ctx: ScanContext, depth: int = 0
    ) -> list[Partial]:
        """Unwrap base64 / hex / percent layers and re-scan the plaintext.

        Bounded in depth and size. Without this there is no answer to "what
        about a base64-wrapped secret?", which a reviewer will ask.
        """
        thresholds = ctx.settings.thresholds
        max_depth = int(thresholds.get("reencode_max_depth", 2))
        max_bytes = int(thresholds.get("reencode_max_bytes", 4096))
        if depth >= max_depth:
            return []

        out: list[Partial] = []
        candidates: list[tuple[str, str, int, int]] = []  # (encoding, decoded, start, end)

        for match in re.finditer(r"\b[A-Za-z0-9+/]{20,}={0,2}", text):
            token = match.group(0)
            if len(token) > max_bytes:
                continue
            try:
                decoded = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
                decoded_text = decoded.decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                continue
            if decoded_text.isprintable():
                candidates.append(("base64", decoded_text, match.start(), match.end()))

        for match in re.finditer(r"\b(?:[0-9a-fA-F]{2}){10,}\b", text):
            token = match.group(0)
            try:
                decoded_text = bytes.fromhex(token).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if decoded_text.isprintable():
                candidates.append(("hex", decoded_text, match.start(), match.end()))

        for encoding, decoded_text, start, end in candidates:
            for inner in self._scan_decoded(decoded_text, ctx, depth):
                out.append(
                    Partial(
                        pointer=pointer,
                        json_path=json_path,
                        # Reported at the *encoded* span: that is what the
                        # artifact contains and what a sanitizer must replace.
                        start=start,
                        end=end,
                        type=inner.type,
                        layer=self.id,
                        confidence=max(0.4, inner.confidence - 0.2),
                        rule_id=f"l2.recoded.{encoding}->{inner.rule_id}",
                        value=inner.value,
                        decoded_from=encoding,
                    )
                )
        return out

    def _scan_decoded(self, text: str, ctx: ScanContext, depth: int) -> list[Partial]:
        out = self.assignment_findings(text, "", "$", ctx)
        out += self.dsn_password_findings(text, "", "$", ctx)
        return out


class L3EntropyLayer:
    """Entropy scoring on normalised, class-gated, word-suppressed tokens."""

    id = "L3"

    def __init__(self, thresholds: dict[str, Any]) -> None:
        self.version = str(thresholds.get("version", "1"))
        self.min_length = int(thresholds.get("entropy_min_token_length", 20))
        self.threshold = float(thresholds.get("entropy_normalised_threshold", 0.75))
        self.min_classes = int(thresholds.get("entropy_min_classes", 2))

    def scan_text(self, text: str, pointer: str, json_path: str, ctx: ScanContext) -> list[Partial]:
        out: list[Partial] = []
        for token, offset in tokenize(text):
            if len(token) < self.min_length:
                continue
            raw, normalised, cls = normalised_entropy(token)
            if cls is None:
                continue
            # Require a mix of character classes. A single-class token (all
            # lowercase, say) is far more likely to be a word or an id than a
            # credential.
            classes = set()
            for char in token:
                if char.isdigit():
                    classes.add("digit")
                elif char.islower():
                    classes.add("lower")
                elif char.isupper():
                    classes.add("upper")
                else:
                    classes.add("symbol")
            if len(classes) < self.min_classes:
                continue
            if normalised < self.threshold:
                continue
            if looks_like_word(token):
                continue
            out.append(
                Partial(
                    pointer=pointer,
                    json_path=json_path,
                    start=offset,
                    end=offset + len(token),
                    type="SECRET_ENTROPIC_TOKEN",
                    layer=self.id,
                    confidence=ctx.confidence_for(self.id),
                    rule_id=f"l3.entropy.{cls}@{self.version}",
                    value=token,
                    entropy=raw,
                    entropy_normalised=normalised,
                    alphabet=cls,
                )
            )
        return out


class L4PolicyLayer:
    """Allowlist suppression and known-value denylist."""

    id = "L4"

    def __init__(self, rules: dict[str, Any]) -> None:
        self.version = str(rules.get("version", "1"))
        suppress = rules.get("suppress", {})
        self.pointer_prefixes = suppress.get("pointer_prefixes", [])
        self.pointer_suffixes = suppress.get("pointer_suffixes", [])
        self.value_literals = {v.lower() for v in suppress.get("value_literals", [])}
        hex_rule = suppress.get("hex_token_rule", {}) or {}
        self.hex_enabled = bool(hex_rule.get("enabled", False))
        self.hex_pattern = re.compile(hex_rule.get("pattern", r"\b[0-9a-fA-F]{7,64}\b"))
        self.hex_window = int(hex_rule.get("context_window_chars", 48))
        self.token_pattern = re.compile(suppress.get("sanitizer_token_pattern", r"(?!x)x"))
        self.placeholder_pattern = re.compile(suppress.get("sanitizer_placeholder_pattern", r"(?!x)x"))
        self.deny_hashes = set(rules.get("deny", {}).get("value_hashes", []))
        self._context_keys: set[str] = set()
        # Hashes of tokens this sanitizer produced. A format-preserving token
        # still looks exactly like the secret it replaced, so the detector has
        # to be told what to ignore or the idempotence check fails by
        # construction rather than by finding a real bug.
        self.runtime_allow_hashes: set[str] = set()

    def set_context_keys(self, keys: Iterable[str]) -> None:
        self._context_keys = {k.lower() for k in keys}

    def deny_findings(self, pointer: str, json_path: str, value: str, ctx: ScanContext) -> list[Partial]:
        """Force-flag values known to be real. Empty unless a denylist is loaded."""
        if not self.deny_hashes:
            return []
        digest = sha256_hex(canonicalize_secret(value))
        if digest not in self.deny_hashes:
            return []
        return [
            Partial(
                pointer=pointer,
                json_path=json_path,
                start=0,
                end=len(value),
                type="SECRET_API_KEY",
                layer=self.id,
                confidence=ctx.confidence_for(self.id),
                rule_id=f"l4.denylist@{self.version}",
                value=value,
            )
        ]

    def suppression_reason(self, finding: Partial, text: str) -> str | None:
        pointer = finding.pointer
        for prefix in self.pointer_prefixes:
            if pointer == prefix or pointer.startswith(prefix + "/"):
                return f"l4.pointer_prefix:{prefix}"
        for suffix in self.pointer_suffixes:
            if pointer.endswith(suffix):
                return f"l4.pointer_suffix:{suffix}"

        # Key-name suppression. This is what catches the trajectory's own
        # per-step chain_hash and trace_id fields: 64 hex characters, which no
        # entropy score can distinguish from a key, sitting under a key that
        # says exactly what they are.
        #
        # The limitation is real and worth stating: an attacker who controls
        # field names can hide a secret under an allowlisted name. L4 is a
        # false-positive control, not an adversary-proof filter.
        if "/" in pointer:
            key = pointer.rsplit("/", 1)[-1].lower()
            if key in self._context_keys:
                return f"l4.key_name:{key}"

        stripped = finding.value.strip()
        if stripped.lower() in self.value_literals:
            return "l4.value_literal"

        if self.runtime_allow_hashes:
            digest = sha256_hex(canonicalize_secret(stripped))
            if digest in self.runtime_allow_hashes:
                return "l4.produced_token"

        # The sanitizer's own output must never be re-flagged, or the
        # idempotence test fails.
        if self.token_pattern.fullmatch(stripped) or self.placeholder_pattern.fullmatch(stripped):
            return "l4.sanitizer_token"

        if self.hex_enabled and self._is_allowlisted_hex(finding, text):
            return "l4.hex_context"
        return None

    def _is_allowlisted_hex(self, finding: Partial, text: str) -> bool:
        """Suppress hex only when the *immediately preceding* key says what it is.

        An earlier version looked for an allowlisted word anywhere in a
        48-character window. That silently suppressed genuine hex secrets
        whenever an unrelated allowlisted word happened to sit nearby, and the
        corpus caught it: a 32-hex `AUTH_TOKEN` value was being dropped because
        a chain hash ended a few characters earlier on the line.

        So the rule now requires the value to be *introduced* by the allowlisted
        name -- `build_id: <hex>` -- which is what "this hex is an identifier"
        actually looks like. Substring matching on the key name keeps
        `content_sha256` covered without opening the window back up.
        """
        if not self.hex_pattern.fullmatch(finding.value.strip()):
            return False
        prefix = text[max(0, finding.start - 64) : finding.start]
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*$", prefix)
        if not match:
            return False
        key = match.group(1).lower()
        return any(context_key in key for context_key in self._context_keys)


# --- orchestration ---------------------------------------------------------


class Detector:
    """Runs the enabled layers over one trajectory document."""

    ALL_LAYERS = ("L4_deny", "L2", "L1", "L3", "L4_allow")

    def __init__(self, settings: Settings | None = None, layers: Iterable[str] | None = None) -> None:
        self.settings = settings or load_settings()
        self.l1 = L1FormatLayer(load_json_rule("l1_patterns.json"))
        self.l2_rules = load_json_rule("l2_field_map.json")
        self.l2 = L2ContextLayer(self.l2_rules)
        self.l3 = L3EntropyLayer(self.settings.thresholds)
        self.l4 = L4PolicyLayer(load_json_rule("l4_allowlist.json"))
        self.l4.set_context_keys(self.l2_rules.get("context_key_allowlist", []))
        self.layers = tuple(layers) if layers is not None else tuple(
            self.settings.thresholds.get("layers", self.ALL_LAYERS)
        )
        self.rules_fp = rules_fingerprint()

    def scan(self, doc: dict[str, Any]) -> FindingSet:
        run_id = str(doc.get("run_id", "unknown"))
        trace_id = str(doc.get("trace_id", ""))
        ctx = ScanContext(
            run_id=run_id,
            trace_id=trace_id,
            settings=self.settings,
            provenance=build_provenance(doc),
            document=doc,
            indexes=index_document(doc),
        )

        partials: list[Partial] = []
        for pointer, path_expr, value in iter_leaves(doc):
            if not isinstance(value, str) or not value:
                continue
            partials.extend(self._scan_leaf(doc, pointer, path_expr, value, ctx))

        findings = self._materialise(partials, ctx)
        findings = self._apply_l4(findings, ctx)
        findings = self._link_parents(findings)
        findings = self._dedupe(findings)

        return FindingSet(
            run_id=run_id,
            findings=findings,
            layers_run=list(self.layers),
            rules_fingerprint=self.rules_fp,
            thresholds_version=self.settings.thresholds_version,
        )

    # -- leaf scanning ------------------------------------------------------

    def _key_for(self, pointer: str) -> str | None:
        if "/" not in pointer:
            return None
        return pointer.rsplit("/", 1)[-1]

    def _scan_leaf(
        self,
        doc: dict[str, Any],
        pointer: str,
        path_expr: str,
        value: str,
        ctx: ScanContext,
    ) -> list[Partial]:
        out: list[Partial] = []
        policy = self.l2.path_policy(pointer)
        scrutinise = policy is None or bool(policy.get("scrutinise", True))

        if "L4_deny" in self.layers:
            out.extend(self.l4.deny_findings(pointer, path_expr, value, ctx))

        if "L2" in self.layers and scrutinise:
            key = self._key_for(pointer)
            out.extend(self.l2.key_name_findings(key, value, pointer, path_expr, ctx))
            out.extend(self.l2.assignment_findings(value, pointer, path_expr, ctx))
            out.extend(self.l2.dsn_password_findings(value, pointer, path_expr, ctx))
            out.extend(self.l2.injection_findings(value, pointer, path_expr, ctx))
            out.extend(self.l2.decoded_findings(value, pointer, path_expr, ctx))

        if "L1" in self.layers and scrutinise:
            out.extend(self.l1.scan_text(value, pointer, path_expr, ctx))

        if "L3" in self.layers and scrutinise:
            out.extend(self.l3.scan_text(value, pointer, path_expr, ctx))

        return out

    # -- materialisation ----------------------------------------------------

    def _materialise(self, partials: list[Partial], ctx: ScanContext) -> list[Finding]:
        findings: list[Finding] = []
        for partial in partials:
            text = ctx.text_at(partial.pointer)
            if not text:
                continue
            index = ctx.indexes.get(partial.pointer) or TextIndex(text)
            end = min(partial.end, len(text))
            start = max(0, min(partial.start, end))
            location = index.locate(start, end)

            tainted = partial.tainted or (
                ctx.is_untrusted(partial.pointer) and partial.type in TAINT_ESCALATED_TYPES
            )
            confidence = partial.confidence
            severity = severity_for(partial.type)
            if tainted:
                confidence = min(1.0, confidence + 0.05)
                if severity in ("High", "Critical"):
                    severity = "Critical"

            digest = sha256_hex(canonicalize_secret(partial.value))
            finding_id = sha256_hex(
                f"{ctx.run_id}|{partial.pointer}|{start}|{end}|{partial.type}"
            )[:16]

            findings.append(
                Finding(
                    finding_id=finding_id,
                    run_id=ctx.run_id,
                    trace_id=ctx.trace_id,
                    step_index=ctx.step_index(partial.pointer),
                    json_path=partial.json_path,
                    locator=Locator(
                        pointer=partial.pointer,
                        char_start=location.char_start,
                        char_end=location.char_end,
                        line=location.line,
                        col=location.col,
                    ),
                    type=partial.type,
                    layer=partial.layer,  # type: ignore[arg-type]
                    confidence=round(confidence, 4),
                    severity=severity,
                    classification=TAXONOMY.get(
                        partial.type,
                        TAXONOMY["SECRET_ENTROPIC_TOKEN"],
                    ).classification,
                    value_sha256=digest,
                    value_length=len(partial.value),
                    preview=make_preview(partial.value),
                    evidence=Evidence(
                        rule_id=partial.rule_id,
                        entropy=partial.entropy,
                        entropy_normalised=partial.entropy_normalised,
                        alphabet_class=partial.alphabet,
                        key_name=partial.key_name,
                        decoded_from=partial.decoded_from,
                        taint=tainted,
                    ),
                    tainted=tainted,
                )
            )
        return findings

    # -- post-processing ----------------------------------------------------

    def _apply_l4(self, findings: list[Finding], ctx: ScanContext) -> list[Finding]:
        if "L4_allow" not in self.layers:
            return findings
        # The context-bounded hex rule needs the containing text, so the whole
        # leaf is fetched rather than just the matched span.
        for finding in findings:
            text = ctx.text_at(finding.locator.pointer)
            partial = Partial(
                pointer=finding.locator.pointer,
                json_path=finding.json_path,
                start=finding.locator.char_start,
                end=finding.locator.char_end,
                type=finding.type,
                layer=finding.layer,
                confidence=finding.confidence,
                rule_id=finding.evidence.rule_id,
                value=text[finding.locator.char_start : finding.locator.char_end],
            )
            _ = ctx
            reason = self.l4.suppression_reason(partial, text)
            if reason:
                finding.suppressed_by = reason
        return findings

    @staticmethod
    def _link_parents(findings: list[Finding]) -> list[Finding]:
        """Attach a DSN's password finding to the DSN finding that contains it."""
        containers = [
            f
            for f in findings
            if f.type == "SECRET_DSN" and not f.suppressed
        ]
        for finding in findings:
            if finding.type != "SECRET_DSN_PASSWORD":
                continue
            for container in containers:
                if (
                    container.locator.pointer == finding.locator.pointer
                    and container.locator.char_start <= finding.locator.char_start
                    and container.locator.char_end >= finding.locator.char_end
                ):
                    finding.evidence.parent_finding_id = container.finding_id
                    break
        return findings

    @staticmethod
    def _dedupe(findings: list[Finding]) -> list[Finding]:
        """One finding per (pointer, span, type); keep the most confident."""
        best: dict[tuple[str, int, int, str], Finding] = {}
        for finding in findings:
            key = (
                finding.locator.pointer,
                finding.locator.char_start,
                finding.locator.char_end,
                finding.type,
            )
            current = best.get(key)
            if current is None or finding.confidence > current.confidence:
                best[key] = finding
        return sorted(
            best.values(),
            key=lambda f: (f.locator.pointer, f.locator.char_start, f.layer),
        )


def scan_document(doc: dict[str, Any], settings: Settings | None = None,
                  layers: Iterable[str] | None = None) -> FindingSet:
    """Scan a trajectory document.

    ``doc`` is the artifact as stored: the detector reads leaf text from the
    document itself, so offsets always refer to the same decoded strings the
    sanitizer will later rewrite.
    """
    return Detector(settings, layers).scan(doc)
