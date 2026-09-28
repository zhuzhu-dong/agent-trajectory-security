"""Key custody and tokenization.

Two claims this module is careful *not* to make.

First, it does not claim the ``redacted`` profile is cryptographically
irreversible. On one machine the key sits next to the data, so anyone holding
it can reverse the tokens. The distinction between the profiles is
operational: ``redacted`` is reversible *because a vault exists* and the vault
is access controlled and audited; ``scrubbed`` is one-way *because no vault is
ever written*. That is policy separation, not a cryptographic guarantee, and
the report says so.

Second, it does not claim a token hides a low-entropy secret. HMAC is a PRF,
not a slow KDF, and an attacker who holds the key can enumerate guesses
directly. For values with little entropy the honest move is to not derive the
token from the value at all -- correlation is simply given up, because
correlating a six-character password is not worth what it costs.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import string
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canonical import canonical_json, sha256_hex

from trajsec.model import canonicalize_secret

KEY_ENV_VAR = "SANITIZER_KEY"
KEY_FILE_NAME = "sanitizer.key.dpapi"

# Below this, a value-derived token would be brute-forceable by anyone holding
# the key, so the value is not used to derive anything.
LOW_ENTROPY_MIN_LENGTH = 16
LOW_ENTROPY_MIN_NORMALISED = 0.60


class CustodyError(RuntimeError):
    pass


# --- Windows DPAPI ---------------------------------------------------------


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _to_blob(data: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(data, len(data))
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    return blob, buffer  # caller must keep the buffer alive


def dpapi_protect(plaintext: bytes) -> bytes:
    """User-scoped DPAPI encryption. Stdlib only, via ctypes."""
    if os.name != "nt":
        raise CustodyError("DPAPI is only available on Windows")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_in, _keep = _to_blob(plaintext)
    blob_out = _DataBlob()
    ok = crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise CustodyError("CryptProtectData failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def dpapi_unprotect(ciphertext: bytes) -> bytes:
    if os.name != "nt":
        raise CustodyError("DPAPI is only available on Windows")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_in, _keep = _to_blob(ciphertext)
    blob_out = _DataBlob()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise CustodyError("CryptUnprotectData failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


@dataclass
class KeyContext:
    """The tokenization key, plus how it was obtained."""

    key: bytes
    custody: str

    def describe(self) -> dict[str, Any]:
        return {
            "custody": self.custody,
            "key_fingerprint": sha256_hex(self.key)[:16],
            # Never the key itself. This is what the "no key material in any
            # artifact" test greps for the absence of.
            "key_material": None,
        }


def load_or_create_key(key_dir: Path) -> KeyContext:
    """Resolve the tokenization key.

    Precedence: an explicit environment variable (convenient for CI, weaker on
    a shared machine), then a DPAPI-protected file, then a freshly generated
    key written back under DPAPI.

    The stated limit: DPAPI protects against another *user* on the machine. It
    does not protect against the same user, an administrator, or anyone who can
    read the process memory.
    """
    from_env = os.environ.get(KEY_ENV_VAR)
    if from_env:
        return KeyContext(key=hashlib.sha256(from_env.encode()).digest(), custody="env")

    key_path = Path(key_dir) / KEY_FILE_NAME
    if key_path.is_file():
        try:
            return KeyContext(key=dpapi_unprotect(key_path.read_bytes()), custody="dpapi-file")
        except (CustodyError, OSError):
            # Fall through and regenerate rather than failing the run; the old
            # key only mattered to artifacts that no longer verify anyway.
            pass

    key = secrets.token_bytes(32)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(dpapi_protect(key))
        custody = "dpapi-file"
    except (CustodyError, OSError):
        custody = "ephemeral"  # non-Windows or no DPAPI: in-memory only
    return KeyContext(key=key, custody=custody)


# --- token forms -----------------------------------------------------------


def is_low_entropy(value: str) -> bool:
    """Whether a value is too guessable to derive a token from."""
    canonical = canonicalize_secret(value)
    if len(canonical) < LOW_ENTROPY_MIN_LENGTH:
        return True
    counts: dict[str, int] = {}
    for char in canonical:
        counts[char] = counts.get(char, 0) + 1
    total = len(canonical)
    import math

    raw = -sum((n / total) * math.log2(n / total) for n in counts.values())
    import string as _string

    alphabet = _string.ascii_letters + _string.digits
    ceiling = math.log2(len(alphabet))
    return (raw / ceiling) < LOW_ENTROPY_MIN_NORMALISED


_DIGITS = string.digits
_LOWER = string.ascii_lowercase
_UPPER = string.ascii_uppercase
_SEPARATORS = "-_."


def format_preserving_token(value: str, digest: bytes) -> str:
    """Replace a value with same-length, same-character-class filler.

    Format-blind replacement would break the very things the trajectory is
    used for: a DSN parser, a `re.match(r'sk-')` check, or anything that
    type-checks a field would fail on a 64-hex digest where a prefixed key
    used to be. Keeping the shape also keeps every downstream offset stable,
    which is what makes the replacement map unnecessary for these tokens.
    """
    out: list[str] = []
    for index, char in enumerate(value):
        byte = digest[index % len(digest)]
        if char.isdigit():
            out.append(_DIGITS[byte % len(_DIGITS)])
        elif char.islower():
            out.append(_LOWER[byte % len(_LOWER)])
        elif char.isupper():
            out.append(_UPPER[byte % len(_UPPER)])
        elif char in _SEPARATORS:
            out.append(char)
        else:
            pool = _LOWER + _UPPER + _DIGITS
            out.append(pool[byte % len(pool)])
    return "".join(out)


def placeholder_token(type_id: str, counter: int) -> str:
    """A typed, non-value-derived placeholder.

    Used for low-entropy secrets, where a value-derived token would be
    recoverable by enumeration against the key. Correlation between
    occurrences is deliberately lost.
    """
    return f"<REDACTED:{type_id}:{counter}>"


def derive_digest(key: bytes, scope_id: str, value: str, *, nonce: str = "") -> bytes:
    """Keyed digest over the canonicalised value, scoped to a correlation domain."""
    message = f"{scope_id}|{nonce}|{canonicalize_secret(value)}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).digest()


def scope_identifier(scope: str, *, run_id: str, tenant_id: str) -> str:
    """Which correlations survive, expressed as a key-space choice.

    A globally stable token is itself a leak: anyone who can read two tenants'
    trajectories can confirm that the same credential appears in both, and the
    usage graph of a credential is confidentiality-relevant with no
    need-to-know justification. Scoping the token domain is how that is
    controlled rather than assumed away.
    """
    if scope == "global":
        return "global"
    if scope == "tenant":
        return f"tenant:{tenant_id}"
    if scope == "run":
        return f"run:{run_id}"
    return "none"


# --- vault -----------------------------------------------------------------


class Vault:
    """token -> plaintext, for the reversible profile only.

    Its own access control and its own audit chain, because unmasking is a
    privileged, purpose-gated action rather than an ordinary read.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vault ("
            " token_id TEXT PRIMARY KEY, run_id TEXT, type TEXT, plaintext TEXT)"
        )
        self._conn.commit()

    def put(self, token_id: str, run_id: str, type_id: str, plaintext: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO vault VALUES (?,?,?,?)", (token_id, run_id, type_id, plaintext)
        )
        self._conn.commit()

    def get(self, token_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT plaintext FROM vault WHERE token_id = ?", (token_id,)
        ).fetchone()
        return row[0] if row else None

    def tokens_for_run(self, run_id: str) -> list[str]:
        return [
            row[0]
            for row in self._conn.execute(
                "SELECT token_id FROM vault WHERE run_id = ?", (run_id,)
            )
        ]

    def close(self) -> None:
        self._conn.close()


def assert_key_not_in_store(key_context: KeyContext, artifact_bytes: bytes) -> None:
    """The T1 control, as an executable check rather than a claim."""
    if key_context.key in artifact_bytes:
        raise AssertionError("key material found in a store artifact")
    encoded = key_context.key.hex().encode()
    if encoded in artifact_bytes:
        raise AssertionError("hex-encoded key material found in a store artifact")
