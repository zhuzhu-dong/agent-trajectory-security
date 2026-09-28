"""Settings and paths for the agent experiment.

Single source of truth for everything tunable. Nothing here reads a secret
into a log or an artifact: the API key is available to the model client, and
only a truncated SHA-256 *fingerprint* of it is ever persisted.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from dotenvlite import get_secret, load_env_file

# --- Paths -----------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR / "workspace"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = REPORTS_DIR / "metrics" / "runs"
CHECKPOINT_DB = REPORTS_DIR / "metrics" / "checkpoints.sqlite"

ENV_FILE = PROJECT_ROOT / ".env"


# --- Model defaults --------------------------------------------------------

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_REASONING_EFFORT = "none"

# Published rates, USD per million tokens, read from the provider's docs on
# 2026-09-28. DeepSeek's docs use both "deepseek-flash" and "deepseek-v4-flash"
# for the same tier; the model listing endpoint uses the former, which is what
# the API accepts. These are recorded so cost figures in the report are
# reproducible rather than remembered, and they are trivially overridable.
RATE_CARDS: dict[str, dict[str, float | str]] = {
    "deepseek-flash": {
        "input_cache_miss": 0.14,
        "input_cache_hit": 0.028,
        "output": 0.28,
        "currency": "USD",
    },
    "deepseek-v4-flash": {
        "input_cache_miss": 0.14,
        "input_cache_hit": 0.028,
        "output": 0.28,
        "currency": "USD",
    },
    "deepseek-v4-pro": {
        "input_cache_miss": 1.74,
        "input_cache_hit": 0.145,
        "output": 3.48,
        "currency": "USD",
    },
}


def rate_card_for(model: str):
    """Build a stats.RateCard for a model id, falling back to unpriced."""
    from .stats import RateCard

    rates = RATE_CARDS.get(model)
    if not rates:
        return RateCard()
    return RateCard(
        input_cache_hit=float(rates["input_cache_hit"]),
        input_cache_miss=float(rates["input_cache_miss"]),
        output=float(rates["output"]),
        currency=str(rates["currency"]),
    )


class ConfigError(RuntimeError):
    """Raised when the environment is not usable for a live run."""


@dataclass(frozen=True)
class Settings:
    """Resolved run settings."""

    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    reasoning_effort: str
    request_timeout: float
    max_retries: int
    sandbox_root: Path
    runs_dir: Path

    @property
    def api_key_fingerprint(self) -> str:
        """Truncated sha256 of the key. Safe to persist; the key itself is not.

        Recorded in every trajectory so a run can be tied to a credential
        without the credential being recoverable from the artifact.
        """
        digest = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
        return f"sha256:{digest[:16]}"

    def framework_versions(self) -> dict[str, str]:
        """Versions of the pieces whose behaviour the trajectory depends on."""
        import importlib

        versions: dict[str, str] = {}
        for mod_name, label in (
            ("langgraph", "langgraph"),
            ("langchain", "langchain"),
            ("langchain_core", "langchain-core"),
            ("langchain_openai", "langchain-openai"),
            ("openai", "openai"),
            ("pydantic", "pydantic"),
        ):
            try:
                mod = importlib.import_module(mod_name)
                versions[label] = str(getattr(mod, "__version__", "unknown"))
            except Exception:  # pragma: no cover - defensive
                versions[label] = "unavailable"
        versions["python"] = ".".join(str(p) for p in __import__("sys").version_info[:3])
        return versions


def load_settings(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    sandbox_root: Path | None = None,
    runs_dir: Path | None = None,
    require_key: bool = True,
) -> Settings:
    """Resolve settings from .env + process environment.

    Precedence: explicit argument > process environment > .env file > default.
    """
    load_env_file(ENV_FILE)

    api_key = get_secret("DEEPSEEK_API_KEY")
    if require_key and (api_key is None or api_key.startswith("sk-REPLACE")):
        raise ConfigError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it "
            f"in (looked in {ENV_FILE} and the process environment)."
        )

    return Settings(
        model=model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL,
        base_url=(os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        api_key=api_key or "",
        temperature=temperature,
        max_tokens=max_tokens,
        # DeepSeek now defaults thinking mode on. A tool-calling loop wants
        # deterministic, cheap turns, so we ask for non-thinking by default.
        reasoning_effort=os.environ.get("DEEPSEEK_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT,
        request_timeout=float(os.environ.get("DEEPSEEK_TIMEOUT_S") or 120.0),
        max_retries=int(os.environ.get("DEEPSEEK_MAX_RETRIES") or 2),
        sandbox_root=Path(sandbox_root or WORKSPACE_DIR).resolve(),
        runs_dir=Path(runs_dir or RUNS_DIR).resolve(),
    )
