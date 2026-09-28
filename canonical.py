"""Canonical serialization and hashing.

The single serializer both experiments depend on. Hash chains, audit entries
and artifact fingerprints all route through :func:`canonical_json`, so two
components can never disagree about what "the same bytes" means.

Two producers must be handled: plain JSON-compatible data, and pydantic
models (the trajectory schema). Anything else is a programming error and
raises rather than being coerced with ``repr()`` -- a silent ``repr()``
fallback would make hashes depend on object identity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "sha256_hex", "chain_hash"]


def _normalise(obj: Any) -> Any:
    """Convert pydantic models to JSON-compatible data, recursively."""
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(
        f"value of type {type(obj).__name__} is not canonically serializable; "
        "convert it explicitly rather than relying on a repr() fallback"
    )


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 kept.

    ``ensure_ascii=False`` matters: escaping non-ASCII would make the hash
    depend on an encoding choice rather than on the content.
    """
    return json.dumps(
        _normalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def chain_hash(previous: str, payload: Any) -> str:
    """Hash one link: ``sha256(previous || canonical_json(payload))``.

    ``previous`` is mixed in as its hex text, so the link commits to its
    position in the chain and not only to its own contents.
    """
    digest = hashlib.sha256()
    digest.update(previous.encode("utf-8"))
    digest.update(canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()
