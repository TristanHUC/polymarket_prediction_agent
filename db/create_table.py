"""Create the RL agent traces table in PostgreSQL.

Usage (from the project root RL_realtime_strategies/):
    uv run python db/create_table.py
    uv run python -m db.create_table

The connection is configured via .env or environment variables
(see RL_realtime_strategies/.env).
"""

import sys
from pathlib import Path

# Make the project root importable regardless of how this script is launched
# (python db/create_table.py, python -m db.create_table, or as a module
# of the RL_realtime_strategies package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.db_connection import execute, fetch_all

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TABLE_NAME = "agent_traces"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (

    -- 1. Primary Keys & Metadata
    trace_id        UUID PRIMARY KEY,
    condition_id    TEXT NOT NULL,
    model_version   TEXT NOT NULL DEFAULT 'base_llama3',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 2. Market Context (The Prompt State)
    question        TEXT,
    description     TEXT,
    end_date_iso    TIMESTAMPTZ,
    outcomes        JSONB,

    -- 3. The Agent's Execution (The Action)
    agent_full_trace        JSONB,
    -- GRPO: token-level probabilities of the *reference* model over the
    -- assistant tokens of agent_full_trace. Stored as JSON text, one entry
    -- per assistant token (label != -100 in mask_assistant_tokens):
    --     [{{"token_id": 1234, "prob": 0.9876}}, ...]
    ref_probs               TEXT,
    -- GRPO: token-level log probabilities of the *old* policy model
    -- (previous adapter). Same JSON layout as ref_probs, but each entry
    -- carries "log_prob" instead of "prob":
    --     [{{"token_id": 1234, "log_prob": -0.0124}}, ...]
    old_log_probs           TEXT,
    predicted_token_id      TEXT,
    predicted_action        TEXT CHECK (predicted_action IN ('BUY YES', 'BUY NO', 'PASS')),
    confidence_score        NUMERIC,
    bet_amount_usd          NUMERIC,

    -- 4. The Financials (Crucial for the Reward Signal)
    entry_price             NUMERIC,
    entry_liquidity         NUMERIC,

    -- 5. The Evaluator / Reward (Updated 24h Later)
    eval_status             TEXT NOT NULL DEFAULT 'PENDING'
                                CHECK (eval_status IN ('PENDING', 'COMPLETED')),
    price_24h_later         NUMERIC,
    pnl_percent             NUMERIC,
    is_chosen               BOOLEAN
);

-- Indexes for the most common query patterns
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_condition_id
    ON {TABLE_NAME} (condition_id);

CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_eval_status
    ON {TABLE_NAME} (eval_status)
    WHERE eval_status = 'PENDING';

CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_model_version
    ON {TABLE_NAME} (model_version);
"""

# Idempotent migrations for columns added after the initial release.
# `ADD COLUMN IF NOT EXISTS` is safe to run on existing tables: nothing is
# dropped and existing rows keep their data. The new columns stay NULL until
# backfilled by complete_agent_trace_table_with_log_probs.py.
MIGRATION_STATEMENTS: list[str] = [
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS ref_probs TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS old_log_probs TEXT",
]


def table_exists() -> bool:
    """Check if the table already exists."""
    rows = fetch_all(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s",
        (TABLE_NAME,),
    )
    return len(rows) > 0


def create_table(drop_first: bool = False) -> None:
    """Create (or re-create) the agent_traces table.

    Parameters
    ----------
    drop_first:
        If True, drops the table first (use with caution — data loss!).
    """
    if drop_first:
        execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
        print(f"Dropped existing table '{TABLE_NAME}'.")

    # psycopg2 only runs one statement per call, so execute the CREATE
    # TABLE / CREATE INDEX statements one by one, then apply the idempotent
    # ALTER TABLE migrations (safe on existing tables — no data loss).
    for statement in (s.strip() for s in CREATE_TABLE_SQL.split(";") if s.strip()):
        execute(statement)
    for statement in MIGRATION_STATEMENTS:
        execute(statement)
    print(f"Table '{TABLE_NAME}' is ready.")


def describe_table() -> None:
    """Print the table's column info for verification."""
    rows = fetch_all(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (TABLE_NAME,),
    )
    print(f"\nColumns of '{TABLE_NAME}':")
    print(f"{'Column':30s} {'Type':20s} {'Nullable':10s} Default")
    print("-" * 80)
    for r in rows:
        default = r["column_default"] or ""
        print(f"{r['column_name']:30s} {r['data_type']:20s} {r['is_nullable']:10s} {default}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    drop = "--drop" in sys.argv
    if drop:
        print("⚠️  WARNING: --drop flag detected! Existing data will be lost.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    if table_exists() and not drop:
        print(f"Table '{TABLE_NAME}' already exists (use --drop to recreate).")
        print("Applying idempotent migrations (ADD COLUMN IF NOT EXISTS) ...")

    create_table(drop_first=drop)
    describe_table()
