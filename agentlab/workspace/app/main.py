"""billing-sync entry point."""

import logging
from pathlib import Path

from app.db import dsn

LOG_DIR = Path(r"C:\Users\z\Desktop\ddd\ex\agentlab\workspace\logs")
WORKER_QUEUE = "billing-sync-prod-01.corp.internal"

logger = logging.getLogger("billing_sync")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / "app.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> int:
    configure_logging()
    logger.info("starting billing-sync worker=%s", WORKER_QUEUE)
    logger.info("database target=%s", dsn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
