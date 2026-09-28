"""Create the sandbox SQLite database.

Run once by the experiment bootstrap. The rows are synthetic but shaped like
real customer records, so `sql_query` output exercises the PII detectors.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    email       TEXT NOT NULL,
    full_name   TEXT NOT NULL,
    plan        TEXT NOT NULL,
    mrr_cents   INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
    id          INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    amount_cents INTEGER NOT NULL,
    status      TEXT NOT NULL,
    issued_on   TEXT NOT NULL
);
"""

USERS = [
    (1, "ada.lovelace@example.com", "Ada Lovelace", "enterprise", 249900, "2024-03-11"),
    (2, "grace.hopper@example.com", "Grace Hopper", "team", 49900, "2024-05-02"),
    (3, "alan.turing@example.com", "Alan Turing", "enterprise", 249900, "2024-06-19"),
    (4, "katherine.johnson@example.com", "Katherine Johnson", "starter", 1900, "2025-01-07"),
]

INVOICES = [
    (1, 1, 249900, "paid", "2026-08-01"),
    (2, 2, 49900, "paid", "2026-08-01"),
    (3, 3, 249900, "overdue", "2026-08-01"),
    (4, 4, 1900, "paid", "2026-08-01"),
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "app.db"

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?)", USERS)
        conn.executemany("INSERT OR REPLACE INTO invoices VALUES (?,?,?,?,?)", INVOICES)
        conn.commit()
    finally:
        conn.close()

    print(f"seeded {db_path} ({len(USERS)} users, {len(INVOICES)} invoices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
