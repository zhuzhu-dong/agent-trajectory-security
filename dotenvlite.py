"""Minimal .env reader -- stdlib only.

We deliberately do not use python-dotenv (installed version is 0.21.0 from 2022;
its parsing semantics differ from current expectations). This is the whole
feature set we need: KEY=VALUE lines, # comments, optional quotes, no
interpolation. Kept intentionally small so its behaviour is auditable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def parse_env_text(text: str) -> Dict[str, str]:
    """Parse dotenv-shaped text into a dict. Later keys win."""
    out: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env_file(path: os.PathLike | str, *, override: bool = False) -> Dict[str, str]:
    """Read a .env file into os.environ.

    Existing environment variables win unless ``override`` is True, so an
    operator can always override a stale file by exporting the variable.
    Returns only the keys this file actually contributed.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    parsed = parse_env_text(p.read_text(encoding="utf-8", errors="replace"))
    applied: Dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def get_secret(name: str) -> Optional[str]:
    """Fetch a secret from the environment, treating blanks as absent."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
