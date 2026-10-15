"""Evaluate pending agent traces 24+ hours after they were created.

For each trace with ``eval_status = 'PENDING'`` and ``created_at`` older than
24 hours, fetches the current midpoint price of the chosen token, computes
``pnl_percent``, and marks the row as ``COMPLETED``.

After all evaluations, groups traces by ``condition_id`` and sets ``is_chosen``:
``TRUE`` for the best-performing strategy, ``FALSE`` for the worst.

Usage:
    uv run python -m RL_realtime_strategies.pipeline.evaluate_24h_later
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Make the project root importable regardless of how this script is launched
# (python pipeline/evaluate_24h_later.py, python -m ..., or as a package module).
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.robust_http import install as _install_robust_http

_install_robust_http()  # bypass ISP DNS-hijacking of *.polymarket.com

from db.db_connection import execute, fetch_all

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

_CLOB_HOST: str = "https://clob.polymarket.com"
_USER_AGENT: str = "polybrrr-rl/0.1"
_REQUEST_TIMEOUT: int = 30
_MAX_RETRIES: int = 3
_RETRY_SLEEP: float = 1.0

# Agent portfolio (must match the fill_agent_trace_table config)
_PORTFOLIO_USD: int = int(os.getenv("AGENT_PORTFOLIO_USD", "1000"))

# Polymarket taker fee for binary markets (2%).  The CLOB API's per-market fee
# schedule takes precedence when available, but this is a safe fallback.
_DEFAULT_TAKER_FEE: float = 0.02

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_with_retry(url: str, params: dict | None = None) -> dict[str, Any]:
    """GET with retries, returning parsed JSON or an error dict."""
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": _USER_AGENT},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_SLEEP * attempt)
    return {"error": str(last_err)}


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------


def _get_order_book(token_id: str) -> tuple[list[dict], list[dict]]:
    """Return ``(bids, asks)`` for a token, sorted best-first."""
    book = _get_with_retry(f"{_CLOB_HOST}/book", {"token_id": token_id})
    bids: list[dict] = book.get("bids", []) or []
    asks: list[dict] = book.get("asks", []) or []

    # Bids come best-first (highest price → lowest), asks best-first (lowest → highest).
    # For safety, sort explicitly.
    bids.sort(key=lambda level: float(level["price"]), reverse=True)
    asks.sort(key=lambda level: float(level["price"]))
    return bids, asks


def _get_best_bid(token_id: str) -> float | None:
    """Return the current best bid price for a token, or None."""
    bids, _asks = _get_order_book(token_id)
    if not bids:
        return None
    return float(bids[0]["price"])


def _walk_book_for_volume(
    token_id: str,
    target_usd: float,
) -> tuple[float | None, float | None, float, float]:
    """Walk the bid side to fill ``target_usd`` of sell volume.

    Returns ``(best_bid, avg_fill_price, filled_usd, shares_filled)``.
    ``best_bid`` is the top-of-book spot price (None when no bids).
    ``avg_fill_price`` is the volume-weighted average (None when nothing filled).
    """
    bids, _asks = _get_order_book(token_id)
    if not bids:
        return None, None, 0.0, 0.0

    best_bid = float(bids[0]["price"])

    remaining_usd = target_usd
    total_shares = 0.0
    total_proceeds = 0.0

    for level in bids:
        if remaining_usd <= 0:
            break
        price = float(level["price"])
        size = float(level["size"])
        level_usd = price * size
        fill_usd = min(remaining_usd, level_usd)
        fill_shares = fill_usd / price if price > 0 else 0.0
        total_proceeds += fill_usd
        total_shares += fill_shares
        remaining_usd -= fill_usd

    if total_shares == 0:
        return best_bid, None, 0.0, 0.0

    avg_fill = total_proceeds / total_shares
    filled_usd = target_usd - remaining_usd
    return best_bid, avg_fill, filled_usd, total_shares


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_one(trace: dict[str, Any]) -> bool:
    """Evaluate a single pending trace.

    Position size = ``confidence_score * entry_liquidity`` — the agent's
    confidence determines what fraction of available liquidity to deploy.
    Selling is simulated by walking the bid side of the order book; a larger
    position walks deeper and gets a worse average fill, so confidence
    directly impacts realized pnl_percent.
    """
    trace_id = trace["trace_id"]
    action = (trace.get("predicted_action") or "").strip()
    token_id = trace.get("predicted_token_id")

    def _f(val: object) -> float | None:
        return None if val is None else float(val)

    entry_price = _f(trace.get("entry_price"))
    entry_liquidity = _f(trace.get("entry_liquidity"))
    bet_amount_usd = _f(trace.get("bet_amount_usd"))

    log.info("Evaluating %s — %s %s", trace_id, action, token_id or "")

    # PASS / HOLD: no trade → no PnL
    if action == "PASS" or not token_id or entry_price is None:
        execute(
            """UPDATE agent_traces
               SET eval_status = 'COMPLETED', pnl_percent = 0
             WHERE trace_id = %s""",
            (trace_id,),
        )
        log.info("  PASS — marked COMPLETED, pnl=0")
        return True

    # ---- position size: the agent's bet_amount_usd ----
    target_usd = bet_amount_usd if bet_amount_usd is not None else 0.0

    if target_usd <= 0 or entry_price <= 0:
        log.warning(
            "  Skipping %s — target_usd=%.0f entry_price=%.4f",
            trace_id,
            target_usd,
            entry_price,
        )
        return False

    # ---- sell by walking the bid side ----
    best_bid, avg_fill, filled_usd, _shares = _walk_book_for_volume(token_id, target_usd)
    if avg_fill is None:
        log.warning("  Skipping %s — no bids available", trace_id)
        return False

    # PnL relative to entry_liquidity (the available pool), not the bet.
    # This rewards strategies that capture more of the available opportunity.
    # Formula: (gross_return - fee) / entry_liquidity
    liquid = entry_liquidity if entry_liquidity is not None and entry_liquidity > 0 else target_usd
    fee = target_usd * _DEFAULT_TAKER_FEE
    pnl_percent = (filled_usd - target_usd - fee) / liquid

    execute(
        """
        UPDATE agent_traces
        SET eval_status = 'COMPLETED',
            price_24h_later = %s,
            pnl_percent = %s
        WHERE trace_id = %s
        """,
        (best_bid, round(pnl_percent, 6), trace_id),
    )
    log.info(
        "  %s: entry=$%.4f  bid=$%.4f  fill=$%.4f  target=$%.0f  filled=$%.0f  pnl=%.2f%%",
        action,
        entry_price,
        best_bid,
        avg_fill,
        target_usd,
        filled_usd,
        pnl_percent * 100,
    )
    return True


def _set_chosen(condition_id: str) -> int:
    """Set ``is_chosen`` for all COMPLETED traces of a condition_id.

    - ``TRUE``  → highest ``pnl_percent``
    - ``FALSE`` → lowest ``pnl_percent``
    Others are left as NULL.

    Returns the number of traces marked.
    """
    rows = fetch_all(
        """
        SELECT trace_id, pnl_percent
        FROM agent_traces
        WHERE condition_id = %s AND eval_status = 'COMPLETED' AND pnl_percent IS NOT NULL
        ORDER BY pnl_percent DESC
        """,
        (condition_id,),
    )
    if len(rows) < 2:
        return 0

    best = rows[0]
    worst = rows[-1]

    execute(
        "UPDATE agent_traces SET is_chosen = TRUE  WHERE trace_id = %s",
        (best["trace_id"],),
    )
    execute(
        "UPDATE agent_traces SET is_chosen = FALSE WHERE trace_id = %s",
        (worst["trace_id"],),
    )
    log.info(
        "  is_chosen: best=%s (pnl=%.2f%%)  worst=%s (pnl=%.2f%%)",
        best["trace_id"],
        best["pnl_percent"] * 100,
        worst["trace_id"],
        worst["pnl_percent"] * 100,
    )
    return 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 1. Fetch pending traces older than 24h
    pending = fetch_all(
        """
        SELECT *
        FROM agent_traces
        WHERE eval_status = 'PENDING'
          AND created_at < NOW() - INTERVAL '24 hours'
        ORDER BY condition_id, created_at
        """
    )
    log.info("Found %d pending traces (created >24h ago)", len(pending))
    if not pending:
        log.info("Nothing to evaluate — exiting.")
        return

    # 2. Evaluate each trace
    evaluated = 0
    for trace in pending:
        try:
            if _evaluate_one(trace):
                evaluated += 1
        except Exception as exc:
            log.error("  Failed to evaluate %s: %s", trace.get("trace_id", "?"), exc)

    log.info("Evaluated %d/%d traces", evaluated, len(pending))

    # 3. Set is_chosen per condition_id (only for those that just got evaluated)
    condition_ids = sorted({t["condition_id"] for t in pending})
    total_chosen = 0
    for cid in condition_ids:
        total_chosen += _set_chosen(cid)

    log.info("Set is_chosen for %d traces across %d markets", total_chosen, len(condition_ids))
    log.info("Done.")


if __name__ == "__main__":
    main()
