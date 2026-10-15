"""Build a historical training snapshot for a past bet date.

This script mirrors ``retrieve_most_active_markets.py`` for live markets, but
targets backtesting. It reuses an existing top-k snapshot as the market
universe (those markets demonstrably existed at snapshot time), sets
``as_of_date`` to the snapshot date, and enriches every market with a 24-hour
price window from the free CLOB ``/prices-history`` endpoint.

Each market needs two CLOB calls at most (one per outcome is unnecessary for
the reward; the YES token is used, falling back to the first token). The
endpoint is free and keyless, but it only returns minute-or-coarser buckets,
and it never returns order-book depth. Depth is therefore carried over from
the snapshot as an approximation and flagged as such.

Output:
    data/top_k_historical_market/top_k_historical_YYYY-MM-DD.json
    Same market fields as the live snapshot, plus per market:
    as_of_date, entry_token_id, entry_price, entry_ts,
    exit_price_24h, exit_ts, resolution (nullable when unknown).

Usage:
    uv run python -m RL_realtime_strategies.pipeline.retrieve_historical_markets 2026-09-17
    uv run python -m RL_realtime_strategies.pipeline.retrieve_historical_markets \
        2026-09-17 top_k_markets_2026-09-17.json

Environment (same .env as the live script, all optional):
    HISTORICAL_AS_OF_DATE  ISO date used when no CLI date is given.
    HISTORICAL_SNAPSHOT_FILE  Snapshot filename inside data/top_k_active_market/.
    TOP_K_ACTIVE_MARKETS  Maximum markets to enrich (default 30).
    HISTORICAL_FIDELITY_MINUTES  CLOB bucket size in minutes (default 60).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.robust_http import install as _install_robust_http

_install_robust_http()

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

_TOP_K: int = int(os.getenv("TOP_K_ACTIVE_MARKETS", "30"))
_FIDELITY_MINUTES: int = int(os.getenv("HISTORICAL_FIDELITY_MINUTES", "60"))
_CLOB_HOST: str = "https://clob.polymarket.com"
_GAMMA_HOST: str = "https://gamma-api.polymarket.com"
_USER_AGENT: str = "polybrrr-rl/0.1"
_REQUEST_TIMEOUT: int = 30
_MAX_RETRIES: int = 3
_RETRY_SLEEP: float = 1.0

_LIVE_SNAPSHOT_DIR: Path = (
    Path(__file__).resolve().parents[1] / "data" / "top_k_active_market"
)
_OUTPUT_DIR: Path = (
    Path(__file__).resolve().parents[1] / "data" / "top_k_historical_market"
)

log = logging.getLogger(__name__)


def _get_with_retry(
    url: str, params: dict[str, Any] | None = None
) -> dict[str, Any] | list[Any] | None:
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
    log.warning("GET %s failed after %d retries: %s", url, _MAX_RETRIES, last_err)
    return None


def _parse_as_of_date(raw: str) -> datetime:
    text = (raw or "").strip()
    if not text:
        msg = "as-of date must be a non-empty YYYY-MM-DD or ISO string"
        raise ValueError(msg)
    if len(text) == 10:
        text += "T00:00:00+00:00"
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_universe(snapshot_arg: str | None) -> tuple[list[dict[str, Any]], str]:
    if snapshot_arg:
        path = _LIVE_SNAPSHOT_DIR / snapshot_arg
        if not path.exists():
            msg = f"snapshot not found: {path}"
            raise FileNotFoundError(msg)
    else:
        env_file = os.getenv("HISTORICAL_SNAPSHOT_FILE", "").strip()
        if env_file:
            path = _LIVE_SNAPSHOT_DIR / env_file
            if not path.exists():
                msg = f"snapshot not found: {path}"
                raise FileNotFoundError(msg)
        else:
            snapshots = sorted(
                _LIVE_SNAPSHOT_DIR.glob("top_k_markets_*.json"),
                key=lambda p: p.name,
                reverse=True,
            )
            if not snapshots:
                msg = f"no live snapshots in {_LIVE_SNAPSHOT_DIR}"
                raise FileNotFoundError(msg)
            path = snapshots[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    markets = payload.get("markets", []) or []
    log.info("Loaded %d markets from %s", len(markets), path.name)
    return markets, path.name


def _select_token(market: dict[str, Any]) -> dict[str, Any] | None:
    tokens = market.get("tokens", []) or []
    if not tokens:
        return None
    for token in tokens:
        if str(token.get("outcome", "")).strip().lower() == "yes":
            return token
    return tokens[0]


def _parse_history_points(payload: Any) -> list[tuple[int, float]]:
    if isinstance(payload, dict):
        for key in ("history", "prices", "data", "points"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            return []
    if not isinstance(payload, list):
        return []
    points: list[tuple[int, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_ts = item.get("t", item.get("timestamp", item.get("ts")))
        raw_price = item.get("p", item.get("price"))
        try:
            if isinstance(raw_ts, str) and not raw_ts.isdigit():
                ts = int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp())
            else:
                ts = int(float(str(raw_ts)))
            price = float(str(raw_price))
        except (TypeError, ValueError):
            continue
        # Normalize millisecond timestamps to seconds.
        if ts > 10_000_000_000:
            ts //= 1000
        points.append((ts, price))
    points.sort(key=lambda point: point[0])
    return points


def _fetch_window(
    token_id: str, start: datetime, end: datetime, fidelity: int
) -> list[tuple[int, float]]:
    payload = _get_with_retry(
        f"{_CLOB_HOST}/prices-history",
        params={
            "market": token_id,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "fidelity": fidelity,
        },
    )
    if payload is None:
        return []
    return _parse_history_points(payload)


def _fetch_resolution(condition_id: str) -> dict[str, Any] | None:
    for url in (
        f"{_GAMMA_HOST}/markets/{condition_id}",
        f"{_GAMMA_HOST}/markets",
    ):
        params = None if url.endswith(condition_id) else {"condition_ids": condition_id}
        payload = _get_with_retry(url, params=params)
        if isinstance(payload, dict) and payload.get("conditionId") == condition_id:
            return payload
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict) and (
                    row.get("conditionId") == condition_id
                    or row.get("condition_id") == condition_id
                ):
                    return row
    return None


def _enrich_market(
    market: dict[str, Any], as_of: datetime, fidelity: int
) -> dict[str, Any] | None:
    end_date_raw = str(market.get("end_date_iso", "") or "")
    try:
        end_date = datetime.fromisoformat(end_date_raw.replace("Z", "+00:00"))
    except ValueError:
        end_date = None
    if end_date is not None and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)
    window_end = as_of + timedelta(hours=24)
    if end_date is not None and end_date < window_end:
        log.info("Skipping %s: resolves before as_of+24h", market.get("question", "?")[:60])
        return None
    token = _select_token(market)
    if token is None or not token.get("token_id"):
        return None
    token_id = str(token["token_id"])
    points = _fetch_window(token_id, as_of, window_end, fidelity)
    if not points and fidelity != 1440:
        # Retry once at daily fidelity; some old markets only serve coarse buckets.
        points = _fetch_window(token_id, as_of, window_end, 1440)
    if len(points) < 2:
        log.info("Skipping %s: only %d price points", market.get("question", "?")[:60], len(points))
        return None
    entry_ts, entry_price = points[0]
    exit_ts, exit_price = points[-1]
    resolution = _fetch_resolution(str(market.get("condition_id", "")))
    enriched = dict(market)
    enriched["as_of_date"] = as_of.isoformat()
    enriched["entry_token_id"] = token_id
    enriched["entry_price"] = entry_price
    enriched["entry_ts"] = entry_ts
    enriched["exit_price_24h"] = exit_price
    enriched["exit_ts"] = exit_ts
    enriched["liquidity_approx"] = True
    if resolution:
        enriched["resolution"] = {
            "closed": resolution.get("closed", resolution.get("closedTime")),
            "winner": resolution.get("winner", resolution.get("resolvedOutcome")),
        }
    else:
        enriched["resolution"] = None
    return enriched


def build_snapshot(as_of_raw: str, snapshot_arg: str | None, top_k: int) -> Path:
    as_of = _parse_as_of_date(as_of_raw)
    universe, snapshot_name = _load_universe(snapshot_arg)
    # Prefer the most liquid markets first, mirroring the live selector.
    def _liquidity_key(market: dict[str, Any]) -> float:
        try:
            return float(market.get("rewards_min_size", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(universe, key=_liquidity_key, reverse=True)
    enriched: list[dict[str, Any]] = []
    skipped = 0
    for market in ordered:
        if len(enriched) >= top_k:
            break
        try:
            row = _enrich_market(market, as_of, _FIDELITY_MINUTES)
        except Exception as exc:
            log.warning("Failed to enrich %s: %s", market.get("condition_id", "?"), exc)
            row = None
        if row is None:
            skipped += 1
            continue
        enriched.append(row)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / f"top_k_historical_{as_of.strftime('%Y-%m-%d')}.json"
    payload = {
        "generated_at": as_of.isoformat(),
        "as_of_date": as_of.isoformat(),
        "source_snapshot": snapshot_name,
        "k": top_k,
        "count": len(enriched),
        "skipped": skipped,
        "fidelity_minutes": _FIDELITY_MINUTES,
        "note": "entry/exit from CLOB prices-history; depth carried from snapshot as approximation",
        "markets": enriched,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d historical markets (%d skipped) to %s", len(enriched), skipped, out)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = list(argv if argv is not None else sys.argv[1:])
    as_of_raw = args[0] if len(args) >= 1 else os.getenv("HISTORICAL_AS_OF_DATE", "")
    snapshot_arg = args[1] if len(args) >= 2 else None
    if not as_of_raw:
        msg = "provide an as-of date: YYYY-MM-DD as argv[1] or HISTORICAL_AS_OF_DATE"
        raise ValueError(msg)
    out = build_snapshot(as_of_raw, snapshot_arg, _TOP_K)
    print(f"\nDone - {out}")


if __name__ == "__main__":
    main()
