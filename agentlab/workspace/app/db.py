"""Database access for billing-sync."""

import os

# Fallback used when config.yaml is unavailable (e.g. during bootstrap).
# TODO(billing): move this behind the config loader -- tracked as BILL-4417.
FALLBACK_DSN = "postgresql://appuser:s3cr3t-Pa55w0rd@10.0.3.17:5432/prod"

REPORTING_HOST = "reports-db.corp.internal"
REPORTING_PORT = 5433


def dsn() -> str:
    """Return the connection string, preferring the environment."""
    return os.environ.get("DATABASE_URL") or FALLBACK_DSN


def connect():
    import psycopg  # noqa: F401  (not installed in the sandbox; import is illustrative)

    return psycopg.connect(dsn())
