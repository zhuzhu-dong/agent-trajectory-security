"""Experiment 1: real LangGraph agent with full trajectory capture."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable (dotenvlite.py lives there) regardless of
# whether we are run as a module or a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
