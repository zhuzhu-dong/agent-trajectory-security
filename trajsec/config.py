"""Settings, paths and the frozen thresholds.

Every tunable lives here or in ``rules/``. In particular the detector's
numeric thresholds are *calibrated on a held-out split and then frozen* --
they are read from ``rules/thresholds.json``, not hardcoded in the detection
code, because a magic constant in the source is indistinguishable from a
number tuned on the test set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RULES_DIR = PACKAGE_DIR / "rules"
TRAJ_RUNS_DIR = PROJECT_ROOT / "reports" / "metrics" / "runs"
ARTIFACTS_DIR = PROJECT_ROOT / "reports" / "artifacts"
PROTECTED_DIR = ARTIFACTS_DIR / "protected"
THRESHOLDS_PATH = RULES_DIR / "thresholds.json"

# Seeds for corpus construction. Five, so injection variance is reported as a
# spread rather than a single draw.
SEEDS: tuple[int, ...] = (1337, 2024, 7, 99, 4242)

# Trajectory-level calibration/test split. Findings inside one trajectory are
# correlated, so the split must be by trajectory or the test set leaks.
CALIBRATION_FRACTION = 0.5

POLICY_VERSION = "trajsec-policy-1.0"
SCHEMA_VERSION = "1.0"

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "version": "uncalibrated-default",
    "frozen": False,
    # Minimum token length before entropy is scored at all.
    "entropy_min_token_length": 20,
    # Normalised entropy H / log2(|alphabet|), in [0, 1]. Raw Shannon entropy
    # is useless as a threshold: 20 chars of hex cap out at 4 bits while a
    # long English sentence reaches ~4.2, so a fixed raw cut-off both misses
    # hex secrets and floods on prose.
    "entropy_normalised_threshold": 0.75,
    # Token must use at least this many character classes.
    "entropy_min_classes": 2,
    # Maximum decode depth when unwrapping base64/hex/percent layers.
    "reencode_max_depth": 2,
    "reencode_max_bytes": 4096,
    # Confidence assigned per layer, before taint escalation.
    "confidence": {"L1": 0.95, "L2": 0.8, "L3": 0.5, "L4": 0.99},
}


@dataclass(frozen=True)
class Settings:
    runs_dir: Path = TRAJ_RUNS_DIR
    artifacts_dir: Path = ARTIFACTS_DIR
    protected_dir: Path = PROTECTED_DIR
    rules_dir: Path = RULES_DIR
    seeds: tuple[int, ...] = SEEDS
    calibration_fraction: float = CALIBRATION_FRACTION
    policy_version: str = POLICY_VERSION
    thresholds: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @property
    def thresholds_version(self) -> str:
        return str(self.thresholds.get("version", "unknown"))

    def enabled_layers(self) -> list[str]:
        return list(self.thresholds.get("layers", ["L4_deny", "L2", "L1", "L3", "L4_allow"]))


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    """Read frozen thresholds, falling back to the documented default."""
    target = Path(path or THRESHOLDS_PATH)
    if not target.is_file():
        return dict(DEFAULT_THRESHOLDS)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update(loaded)
    return merged


def save_thresholds(thresholds: dict[str, Any], path: Path | None = None) -> Path:
    """Freeze thresholds to disk. Called once, on the calibration split."""
    target = Path(path or THRESHOLDS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_settings(**overrides: Any) -> Settings:
    thresholds = overrides.pop("thresholds", None) or load_thresholds()
    return Settings(thresholds=thresholds, **overrides)


# --- sensitivity levels ----------------------------------------------------

LEVELS: tuple[str, ...] = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET")
_LEVEL_RANK = {level: index for index, level in enumerate(LEVELS)}


def level_rank(level: str) -> int:
    return _LEVEL_RANK.get(level, 0)


def max_level(*levels: str) -> str:
    """Lattice join: the highest classification among the inputs."""
    best = LEVELS[0]
    for level in levels:
        if level_rank(level) > level_rank(best):
            best = level
    return best


def level_at_least(level: str, required: str) -> bool:
    return level_rank(level) >= level_rank(required)
