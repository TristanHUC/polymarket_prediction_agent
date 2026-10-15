"""Fetch the top-K most active Polymarket markets and dump to a JSON snapshot.

Uses the CLOB ``/sampling-markets`` endpoint which returns active, open markets.
Sorts by ``rewards.min_size`` (descending) as a proxy for volume/liquidity.

The CLOB API does not expose raw trade-volume on market-list endpoints,
so ``rewards.min_size`` (the minimum order size in USDC that qualifies
for liquidity rewards) is the best available signal — higher values
correlate with deeper, more liquid markets.

Output:
    data/top_k_active_market/top_k_markets_YYYY-MM-DD.json
    → list of {"condition_id": "...", "question": "...", "rewards_min_size": ..., ...}

Usage:
    uv run python -m RL_realtime_strategies.pipeline.retrieve_most_active_markets
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Make the project root importable regardless of how this script is launched.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.robust_http import install as _install_robust_http

_install_robust_http()  # bypass ISP DNS-hijacking of *.polymarket.com

from db.db_connection import fetch_all

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

_TOP_K: int = int(os.getenv("TOP_K_ACTIVE_MARKETS", "30"))

_CLOB_HOST: str = "https://clob.polymarket.com"
_USER_AGENT: str = "polybrrr-rl/0.1"
_REQUEST_TIMEOUT: int = 30  # seconds
_MAX_RETRIES: int = 3
_RETRY_SLEEP: float = 1.0  # seconds base

# The /sampling-markets endpoint can return up to this many markets in a single
# call.  Set high enough to always capture top-K, even after filtering out
# already-used markets.
_SAMPLING_LIMIT: int = 1000

# Exclude markets that end too soon — the LLM pipeline needs breathing room.
_MIN_HOURS_LEFT: int = 3

_OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "top_k_active_market"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_with_retry(url: str, params: dict | None = None) -> dict[str, Any] | None:
    """GET with retries; returns parsed JSON or None on exhaustion."""
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
                log.debug("Retry %d/%d: %s", attempt, _MAX_RETRIES, exc)
                time.sleep(_RETRY_SLEEP * attempt)
    log.warning("GET %s failed after %d retries: %s", url, _MAX_RETRIES, last_err)
    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _rewards_min_size(market: dict[str, Any]) -> float:
    """Extract min_size from the rewards dict, defaulting to 0."""
    try:
        return float((market.get("rewards") or {}).get("min_size", 0))
    except (ValueError, TypeError):
        return 0.0


def _already_used_condition_ids() -> set[str]:
    """Return the set of condition_ids we have already processed."""
    try:
        rows = fetch_all("SELECT DISTINCT condition_id FROM agent_traces")
        return {r["condition_id"] for r in rows}
    except Exception:
        log.warning("Could not query agent_traces — returning empty exclusion set")
        return set()


def _collect_markets(k: int) -> list[dict[str, Any]]:
    """Fetch /sampling-markets, filter, sort by min_size, return top k.

    /sampling-markets already returns active, non-closed markets only.
    We still double-check the filters for safety.
    """
    url = f"{_CLOB_HOST}/sampling-markets"
    log.info("Fetching %s?limit=%d", url, _SAMPLING_LIMIT)

    response = _get_with_retry(url, params={"limit": _SAMPLING_LIMIT})
    if response is None:
        log.error("Failed to fetch sampling-markets")
        return []

    all_markets: list[dict[str, Any]] = response.get("data") or []
    log.info("Received %d markets from /sampling-markets", len(all_markets))

    # Hard filters (belt-and-suspenders — the endpoint already filters some)
    now = datetime.now(UTC)
    viable: list[dict[str, Any]] = []
    for m in all_markets:
        if m.get("closed") is True or m.get("active") is not True:
            continue
        # Exclude markets ending within _MIN_HOURS_LEFT (default 3h)
        end_str = m.get("end_date_iso")
        if end_str:
            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if (end_dt - now).total_seconds() < _MIN_HOURS_LEFT * 3600:
                    continue
            except ValueError:
                pass  # unparseable date → keep the market
        viable.append(m)
    log.info(
        "After filtering (active, not closed, >%dh left): %d markets",
        _MIN_HOURS_LEFT,
        len(viable),
    )

    # Exclude markets we've already processed
    used_cids = _already_used_condition_ids()
    if used_cids:
        fresh = [m for m in viable if m.get("condition_id") not in used_cids]
        log.info(
            "After excluding %d already-used markets: %d fresh markets",
            len(viable) - len(fresh),
            len(fresh),
        )
        viable = fresh

    # Sort by rewards.min_size descending
    viable.sort(key=_rewards_min_size, reverse=True)

    top = viable[:k]
    log.info("Top-%d markets selected", len(top))
    return top


def _extract_fields(top_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the fields we care about for downstream consumers."""
    slim: list[dict[str, Any]] = []
    for m in top_markets:
        tokens = m.get("tokens", []) or []
        rewards = m.get("rewards") or {}
        slim.append(
            {
                "condition_id": m.get("condition_id"),
                "question": m.get("question"),
                "market_slug": m.get("market_slug"),
                "end_date_iso": m.get("end_date_iso"),
                "active": m.get("active"),
                "closed": m.get("closed"),
                "rewards_min_size": rewards.get("min_size"),
                "rewards_max_spread": rewards.get("max_spread"),
                "token_count": len(tokens),
                "tokens": [
                    {"token_id": t.get("token_id"), "outcome": t.get("outcome")} for t in tokens
                ],
            }
        )
    return slim


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _today_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _output_path() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR / f"top_k_markets_{_today_str()}.json"


def save(markets: list[dict[str, Any]]) -> Path:
    """Write the top-k market list to a timestamped JSON file."""
    out = _output_path()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "k": _TOP_K,
        "count": len(markets),
        "markets": markets,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d markets → %s", len(markets), out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Fetching top-%d active markets (volume-sorted)", _TOP_K)
    top = _collect_markets(_TOP_K)
    slim = _extract_fields(top)

    out = save(slim)
    print(f"\nDone — {len(slim)} markets saved to:\n  {out}")


if __name__ == "__main__":
    main()
