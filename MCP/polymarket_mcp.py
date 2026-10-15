"""MCP server exposing Polymarket data and analysis tools.

Run directly:  uv run python -m polybrrr.MCP.polymarket_mcp
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import requests
from mcp.server.fastmcp import FastMCP
from py_clob_client_v2.client import ClobClient

# Make the project root importable regardless of how this server is launched.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.robust_http import install as _install_robust_http

_install_robust_http()  # bypass ISP DNS-hijacking of *.polymarket.com

mcp = FastMCP("polybrrr-mcp")

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_CLOB_HOST = "https://clob.polymarket.com"
_CHAIN_ID = 137
_USER_AGENT = "polybrrr-mcp/0.1"
_REQUEST_TIMEOUT = 30  # seconds
_MAX_RETRIES = 3
_RETRY_SLEEP = 1.0  # seconds base

# Local data roots — use env vars so the server is portable, with sensible defaults.
_PROCESSED_MARKETS = Path(
    os.getenv("POLYBRRR_MARKETS_DIR", "data/processed/market_universe")
)
_PRICE_HISTORY_DIR = Path(
    os.getenv("POLYBRRR_PRICE_HISTORY_DIR", "data/processed/price_history")
)


def _clob_client() -> ClobClient:
    return ClobClient(_CLOB_HOST, chain_id=_CHAIN_ID)


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


def _latest_snapshot_dir(base: Path) -> Path | None:
    """Return the most recent snapshot=* directory under *base*, or None."""
    if not base.exists():
        return None
    snapshots = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("snapshot=")],
        key=lambda p: p.name,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _outcomes_parquet() -> Path | None:
    snap = _latest_snapshot_dir(_PROCESSED_MARKETS)
    if snap is None:
        return None
    return snap / "outcomes.parquet"


def _query_parquet(path: Path, sql: str) -> list[dict[str, Any]]:
    """Run a DuckDB SQL query against a parquet file, return rows as dicts."""
    if not path.exists():
        return []
    try:
        con = duckdb.connect(":memory:")
        result = con.execute(sql).fetchdf()
        con.close()
        return result.to_dict(orient="records")
    except Exception as exc:
        return [{"error": str(exc)}]


def _clob_market_tokens(condition_id: str) -> list[dict[str, Any]] | None:
    """Fetch CLOB market info and return the token array, or None on failure.

    Each token dict has keys like ``t`` (token_id), ``id``, ``p`` (price).
    Also caches tick size, neg-risk, and fee info for downstream use.
    """
    try:
        resp = requests.get(
            f"{_CLOB_HOST}/clob-markets/{condition_id}",
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("t")
    except Exception:
        return None


# ============================================================================
# Tools
# ============================================================================


@mcp.tool()
def get_order_book(condition_id: str) -> str:
    """Get the current order book for a Polymarket market.

    Returns the real AMM price (midpoint), spread, and last trade price
    for each token (Yes/No), plus the top bid/ask levels from the CLOB
    order book.

    Args:
        condition_id: The Polymarket condition ID (e.g. "0xabc123...") for the market.
    """
    try:
        # Resolve condition_id → token IDs via /clob-markets
        tokens = _clob_market_tokens(condition_id)
        if not tokens or not tokens[0]:
            return json.dumps(
                {"error": f"Could not resolve CLOB market info for {condition_id}"},
                indent=2,
            )

        entries: list[dict[str, Any]] = []
        for tok in tokens:
            tid = tok.get("t") or tok.get("token_id")
            if not tid:
                continue

            # Real AMM price
            mp = _get_with_retry(
                f"{_CLOB_HOST}/midpoint", params={"token_id": tid}
            )
            sp = _get_with_retry(
                f"{_CLOB_HOST}/spread", params={"token_id": tid}
            )
            lp = _get_with_retry(
                f"{_CLOB_HOST}/last-trade-price", params={"token_id": tid}
            )

            # Raw order book levels
            book = _get_with_retry(
                f"{_CLOB_HOST}/book", params={"token_id": tid}
            )
            bids = book.get("bids", []) or []
            asks = book.get("asks", []) or []

            midpoint = float(mp.get("mid")) if "mid" in mp else None

            def _top_near(levels: list[dict], ref_price: float, n: int = 5) -> list[dict]:
                if not levels or ref_price is None:
                    return []
                sorted_levels = sorted(levels, key=lambda l: abs(
                    float(l["price"]) - ref_price))
                return [
                    {"price": lvl["price"], "size": lvl["size"]}
                    for lvl in sorted_levels[:n]
                ]

            entries.append({
                "token_id": tid,
                "midpoint": midpoint,
                "spread": float(sp.get("spread")) if "spread" in sp else None,
                "last_trade_price": float(lp.get("price")) if "price" in lp else None,
                "last_trade_side": lp.get("side") if "side" in lp else None,
                "bid_depth_total": sum(float(b["size"]) for b in bids),
                "ask_depth_total": sum(float(a["size"]) for a in asks),
                "levels_near_mid": {
                    "bids": _top_near(bids, midpoint),
                    "asks": _top_near(asks, midpoint),
                },
            })

        return json.dumps(
            {
                "condition_id": condition_id,
                "tokens": entries,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def get_price_history(condition_id: str) -> str:
    """Get the resolved price history for a market from the local processed dataset.

    Queries the Parquet price-history buckets collected by polybrrr. Requires a
    previously-run ``update_dataset`` or ``collect_price_history`` pipeline stage.

    Args:
        condition_id: The Polymarket condition ID (e.g. "0xabc123...").
    """
    try:
        points_dir = _PRICE_HISTORY_DIR
        if not points_dir.exists():
            return json.dumps(
                {"error": f"Price history directory not found: {points_dir}"}, indent=2
            )

        # Find the latest run
        runs: list[Path] = []
        for snap_dir in sorted(points_dir.iterdir(), reverse=True):
            if snap_dir.is_dir() and snap_dir.name.startswith("market_snapshot="):
                for run_dir in sorted(snap_dir.iterdir(), reverse=True):
                    if run_dir.is_dir() and run_dir.name.startswith("run="):
                        runs.append(run_dir)

        if not runs:
            return json.dumps({"error": "No price history runs found"}, indent=2)

        # Search token_index in the latest run
        latest_run = runs[0]
        points_root = latest_run / "points"
        token_index = latest_run / "token_index.parquet"

        if not token_index.exists():
            return json.dumps(
                {"error": f"No token_index in {latest_run}"}, indent=2
            )

        # Look up which token_bucket the token lives in
        rows = _query_parquet(
            token_index,
            f"""
            SELECT * FROM read_parquet('{token_index.as_posix()}')
            WHERE condition_id = '{condition_id}'
            """,
        )
        if not rows:
            return json.dumps(
                {"error": f"Token not found: {condition_id}"}, indent=2
            )

        record = rows[0]
        bucket = record.get("bucket", 0)
        bucket_dir = points_root / f"token_bucket={bucket:03d}"

        if not bucket_dir.exists():
            return json.dumps(
                {"error": f"Bucket dir not found: {bucket_dir}"}, indent=2
            )

        # Read price points
        points_parquet = bucket_dir / "part-0.parquet"
        if not points_parquet.exists():
            return json.dumps({"error": "No price points file found"}, indent=2)

        price_rows = _query_parquet(
            points_parquet,
            f"""
            SELECT ts, price, source
            FROM read_parquet('{points_parquet.as_posix()}')
            WHERE condition_id = '{condition_id}'
            ORDER BY ts
            """,
        )

        return json.dumps(
            {
                "condition_id": condition_id,
                "points": price_rows,
                "count": len(price_rows),
            },
            indent=2,
            default=str,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def read_webpage(url: str) -> str:
    """Read the full text content of a webpage.

    Use this tool when the snippet from websearch is not enough and you need to 
    read the full article to evaluate a market.

    Args:
        url: The full URL of the webpage to read.
    """
    try:
        # r.jina.ai is a free service that converts any URL to clean Markdown for LLMs
        jina_url = f"https://r.jina.ai/{url}"

        headers = {
            "User-Agent": _USER_AGENT,
            # Optional: Accept header ensures we get markdown
            "Accept": "text/markdown"
        }

        resp = requests.get(jina_url, headers=headers, timeout=15)
        resp.raise_for_status()

        content = resp.text

        # TRUNCATION IS CRITICAL
        # An average token is ~4 characters.
        # 16,000 characters is roughly 4,000 tokens.
        # This protects your 30K context window from getting blown up by massive pages.
        max_chars = 16000
        if len(content) > max_chars:
            content = content[:max_chars] + \
                "\n\n...[CONTENT TRUNCATED FOR LENGTH]..."

        return json.dumps({
            "url": url,
            "content": content,
            "truncated": len(resp.text) > max_chars
        }, indent=2)

    except Exception as exc:
        return json.dumps({"error": f"Failed to read webpage: {str(exc)}"}, indent=2)


@mcp.tool()
def websearch(query: str, time_filter: str = None) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: The search query.
        time_filter: Optional. Use 'd' for past day, 'w' for past week, 'm' for past month, 'y' for past year. 
                     Leave null for no time limit.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            # Passes the timelimit to DuckDuckGo
            raw = list(ddgs.text(query, timelimit=time_filter, max_results=8))
            results = [
                {"title": r["title"], "url": r["href"],
                    "snippet": r.get("body", "")}
                for r in raw
            ]
        return json.dumps({"query": query, "time_filter": time_filter, "results": results}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def search_active_markets(query: str) -> str:
    """Search Polymarket markets matching a keyword.

    Searches two sources:
      1. The CLOB /markets endpoint (standalone markets, paginated).
      2. The Polymarket website /api/search page (covers event-based
         sub-markets such as World Cup match markets).

    Punctuation is stripped so "vs." matches "vs".

    Args:
        query: A search keyword or phrase to filter markets.
    """
    try:
        import re as _re

        # Normalize keywords
        raw_keywords = query.lower().split()
        keywords = [
            kw for kw in (_re.sub(r"[^a-z0-9\u00e0-\u00ff]+", "", kw) for kw in raw_keywords)
            if len(kw) >= 3
        ]

        def _score_market(m: dict) -> tuple[float, dict] | None:
            """Return (match_score, record) if market matches keywords, else None."""
            question = (m.get("question") or "").lower()
            slug = (m.get("market_slug") or "").lower()
            tags_text = " ".join(m.get("tags") or []).lower()

            q_clean = _re.sub(r"[^a-z0-9\u00e0-\u00ff\s]+", "", question)
            slug_clean = _re.sub(
                r"[^a-z0-9\u00e0-\u00ff\s]+", " ", slug.replace("-", " "))
            tags_clean = _re.sub(r"[^a-z0-9\u00e0-\u00ff\s]+", " ", tags_text)

            all_words = set(q_clean.split()) | set(
                slug_clean.split()) | set(tags_clean.split())
            matched = [kw for kw in keywords if kw in all_words]
            ratio = len(matched) / len(keywords) if keywords else 0.0

            if ratio >= 0.6 or len(matched) >= 2:
                tokens = m.get("tokens", []) or []
                return (ratio, {
                    "condition_id": m.get("condition_id"),
                    "question": m.get("question"),
                    "market_slug": m.get("market_slug"),
                    "closed": m.get("closed"),
                    "active": m.get("active"),
                    "match_score": round(ratio, 2),
                    "token_count": len(tokens),
                    "tokens": [
                        {"token_id": t.get("token_id"),
                         "outcome": t.get("outcome")}
                        for t in tokens
                    ],
                })
            return None

        seen_cids: set[str] = set()
        matches: list[dict[str, Any]] = []

        # --- Source 1: CLOB /sampling-markets (active / featured markets) ---
        try:
            resp = requests.get(
                f"{_CLOB_HOST}/sampling-markets",
                params={"limit": 200},
                headers={"User-Agent": _USER_AGENT},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                for m in (resp.json().get("data") or []):
                    result = _score_market(m)
                    if result:
                        cid = result[1]["condition_id"] or ""
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            matches.append(result[1])
        except Exception:
            pass

        # --- Source 2: CLOB /markets (full listing, paginated) ---
        try:
            client = _clob_client()
            cursor = ""
            for _ in range(5):
                page = client.get_markets(cursor)
                for m in (page.get("data") or []):
                    result = _score_market(m)
                    if result:
                        cid = result[1]["condition_id"] or ""
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            matches.append(result[1])
                cursor = page.get("next_cursor", "")
                if not cursor or cursor == "LTE=":
                    break
        except Exception:
            pass

        # --- Source 3: Polymarket website search (covers event sub-markets) ---
        # Collect event slugs whose sub-markets need conditionId enrichment
        _event_slugs: set[str] = set()
        try:
            site_resp = requests.get(
                "https://polymarket.com/api/search",
                params={"q": query},
                headers={"User-Agent": _USER_AGENT},
                timeout=15,
            )
            if site_resp.status_code == 200 and "__NEXT_DATA__" in site_resp.text:
                html = site_resp.text
                s = html.find("__NEXT_DATA__")
                s = html.find(">", s) + 1
                e = html.find("</script>", s)
                nd = json.loads(html[s:e].strip())
                ds = nd.get("props", {}).get("pageProps", {}).get(
                    "dehydratedState", {}) or {}
                for q_entry in ds.get("queries") or []:
                    qk = q_entry.get("queryKey", [])
                    if not isinstance(qk, list) or "search" not in str(qk):
                        continue
                    qdata = (q_entry.get("state") or {}).get("data")
                    if not qdata:
                        continue
                    for page in (qdata if isinstance(qdata, list) else qdata.get("pages", [])):
                        if not isinstance(page, dict):
                            continue
                        for item in (page.get("results") or []):
                            if not isinstance(item, dict):
                                continue
                            title = (item.get("title") or "").lower()
                            event_slug = item.get("slug", "")
                            for sub in (item.get("markets") or []):
                                if not isinstance(sub, dict):
                                    continue
                                sq = (sub.get("question") or "").lower()
                                combined_text = f"{sq} {title}"
                                c_words = set(
                                    _re.sub(r"[^a-z0-9\u00e0-\u00ff\s]+", "",
                                            combined_text).split()
                                )
                                matched = [
                                    kw for kw in keywords if kw in c_words]
                                ratio = len(matched) / \
                                    len(keywords) if keywords else 0.0

                                if ratio >= 0.6 or len(matched) >= 2:
                                    cid = sub.get("conditionId") or sub.get(
                                        "condition_id") or ""
                                    if not cid and event_slug:
                                        _event_slugs.add(event_slug)
                                    market = {
                                        "condition_id": cid,
                                        "question": sub.get("question") or sub.get("title"),
                                        "market_slug": sub.get("slug") or event_slug,
                                        "closed": sub.get("closed"),
                                        "active": sub.get("active"),
                                        "tags": sub.get("tags") or [],
                                        "tokens": sub.get("tokens") or [],
                                        "match_score": round(ratio, 2),
                                    }
                                    if not cid or cid not in seen_cids:
                                        if cid:
                                            seen_cids.add(cid)
                                        matches.append(market)
        except Exception:
            pass

        # --- Enrich empty condition_ids by scraping event pages ---
        if _event_slugs:
            # Extract parent slugs (strip sub-category suffixes like -exact-score, -more-markets)
            _parent_slugs: set[str] = set()
            for _s in _event_slugs:
                # Parent slug is everything before date pattern or up to known suffixes
                import re as _re_slug
                _m = _re_slug.search(r"^(.+?\d{4}-\d{2}-\d{2})", _s)
                _parent_slugs.add(_m.group(1) if _m else _s)

            for _parent in _parent_slugs:
                try:
                    ev_resp = requests.get(
                        f"https://polymarket.com/fr/event/{_parent}",
                        headers={"User-Agent": _USER_AGENT},
                        timeout=15,
                    )
                    if ev_resp.status_code == 200 and "__NEXT_DATA__" in ev_resp.text:
                        h = ev_resp.text
                        si = h.find("__NEXT_DATA__")
                        si = h.find(">", si) + 1
                        ei = h.find("</script>", si)
                        ev_nd = json.loads(h[si:ei].strip())
                        for qe in (ev_nd.get("props", {}).get("pageProps", {})
                                   .get("dehydratedState", {}).get("queries") or []):
                            qk_str = str(qe.get("queryKey", []))
                            if _parent not in qk_str:
                                continue
                            sdata = (qe.get("state") or {}).get("data")
                            if not isinstance(sdata, dict):
                                continue
                            for ev_m in (sdata.get("markets") or []):
                                if not isinstance(ev_m, dict):
                                    continue
                                ev_cid = ev_m.get("conditionId") or ""
                                ev_slug = ev_m.get("slug", "")
                                if not ev_cid:
                                    continue
                                # Fill in matching markets
                                for m in matches:
                                    if (m["condition_id"] == ""
                                            and m.get("market_slug") == ev_slug):
                                        m["condition_id"] = ev_cid
                                        seen_cids.add(ev_cid)
                except Exception:
                    pass

        matches.sort(
            key=lambda x: (-x["match_score"], x.get("question") or ""))

        return json.dumps(
            {
                "query": query,
                "matches": matches[:50],
                "count": len(matches),
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def get_market_liquidity_and_fees(condition_id: str) -> str:
    """Get liquidity depth and fee schedule for a Polymarket market.

    Combines the order-book spread, total depth, and the Polymarket taker-fee
    schedule (which depends on the market's outcome count and neg-risk status).

    Args:
        condition_id: The Polymarket condition ID for the market.
    """
    try:
        # Fetch CLOB market info (token_ids, tick size, neg-risk, fees)
        clob_resp = requests.get(
            f"{_CLOB_HOST}/clob-markets/{condition_id}",
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        clob_data = clob_resp.json() if clob_resp.status_code == 200 else {}
        tokens_arr = clob_data.get("t") or []

        if not tokens_arr:
            return json.dumps(
                {"error": f"Could not resolve token for {condition_id}"}, indent=2
            )

        # Fetch market metadata (question, outcomes)
        market_data = _get_with_retry(f"{_CLOB_HOST}/markets/{condition_id}")

        if "error" in market_data:
            return json.dumps(market_data, indent=2)

        # Fetch order book and real-time pricing for EACH token
        token_outcomes = market_data.get("tokens", []) or []
        book_entries: list[dict[str, Any]] = []
        for i, token_info in enumerate(tokens_arr):
            tid = token_info.get("t") or token_info.get("token_id")
            if not tid:
                continue
            outcome = (
                token_outcomes[i].get("outcome")
                if i < len(token_outcomes)
                else f"Token {i}"
            )

            # Use /midpoint + /spread for the real market price (neg-risk AMM pricing)
            mp_resp = _get_with_retry(
                f"{_CLOB_HOST}/midpoint", params={"token_id": tid}
            )
            sp_resp = _get_with_retry(
                f"{_CLOB_HOST}/spread", params={"token_id": tid}
            )
            lp_resp = _get_with_retry(
                f"{_CLOB_HOST}/last-trade-price", params={"token_id": tid}
            )

            # Raw order book for depth data
            bk_resp = _get_with_retry(
                f"{_CLOB_HOST}/book", params={"token_id": tid}
            )
            bids = bk_resp.get("bids", []) or []
            asks = bk_resp.get("asks", []) or []

            midpoint = float(mp_resp.get("mid")) if "mid" in mp_resp else None
            spread = float(sp_resp.get("spread")
                           ) if "spread" in sp_resp else None
            last_price = float(lp_resp.get("price")
                               ) if "price" in lp_resp else None
            last_side = lp_resp.get("side") if "side" in lp_resp else None

            # Depth near the AMM midpoint (within 20% of midpoint price)
            def _depth_near(levels: list[dict], ref_price: float, threshold: float) -> float:
                if not levels or ref_price is None:
                    return 0.0
                total = 0.0
                for lvl in levels:
                    p = float(lvl["price"])
                    if abs(p - ref_price) / ref_price <= threshold:
                        total += float(lvl["size"])
                return total

            def _top_near(levels: list[dict], ref_price: float, n: int = 5) -> list[dict]:
                """Return up to n levels closest to ref_price, sorted by distance."""
                if not levels or ref_price is None:
                    return []
                sorted_levels = sorted(levels, key=lambda l: abs(
                    float(l["price"]) - ref_price))
                return [
                    {"price": lvl["price"], "size": lvl["size"]}
                    for lvl in sorted_levels[:n]
                ]

            bid_depth_total = sum(float(b["size"]) for b in bids)
            ask_depth_total = sum(float(a["size"]) for a in asks)

            book_entries.append({
                "token_id": tid,
                "outcome": outcome,
                "price": midpoint,
                "spread": spread,
                "last_trade_price": last_price,
                "last_trade_side": last_side,
                "bid_depth_total": bid_depth_total,
                "ask_depth_total": ask_depth_total,
                "bid_depth_near_mid": _depth_near(bids, midpoint, 0.20) if midpoint else None,
                "ask_depth_near_mid": _depth_near(asks, midpoint, 0.20) if midpoint else None,
                "levels_near_mid": {
                    "bids": _top_near(bids, midpoint),
                    "asks": _top_near(asks, midpoint),
                },
            })

        token_count = len(token_outcomes)
        neg_risk = clob_data.get(
            "nr", False) or market_data.get("neg_risk") or False

        # Fee rate from CLOB API
        fd = clob_data.get("fd") or {}
        if fd.get("r"):
            taker_fee_bps = int(float(fd["r"]) * 10000)
        elif neg_risk or token_count <= 2:
            taker_fee_bps = 200
        elif token_count <= 5:
            taker_fee_bps = 300
        elif token_count <= 10:
            taker_fee_bps = 400
        else:
            taker_fee_bps = 500

        tick_size = clob_data.get(
            "mts") or market_data.get("minimum_tick_size")

        return json.dumps(
            {
                "condition_id": condition_id,
                "question": market_data.get("question"),
                "token_count": token_count,
                "neg_risk": neg_risk,
                "taker_fee_percent": round(taker_fee_bps / 100, 2),
                "taker_fee_bps": taker_fee_bps,
                "tokens": book_entries,
                "minimum_tick_size": tick_size,
                "minimum_order_size": market_data.get("minimum_order_size"),
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def search_x_twitter(query: str) -> str:
    """Search X (formerly Twitter) for recent posts about a topic.

    Uses DuckDuckGo's ``site:x.com`` operator via the ``ddgs`` library.
    No X API credentials required.

    Args:
        query: The search query string (hashtag, keyword, or phrase).
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(f"site:x.com {query}", max_results=10))
            results = [
                {
                    "title": r["title"],
                    "url": r["href"],
                    "snippet": r.get("body", ""),
                }
                for r in raw
                if "x.com" in r["href"].lower() or "twitter.com" in r["href"].lower()
            ]
        return json.dumps(
            {"query": query, "results": results, "count": len(results)},
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def calculate_expected_value(
    probability: float, payout: float, cost: float
) -> str:
    """Calculate the expected value of a bet.

    Args:
        probability: The probability of winning (between 0 and 1).
        payout: The payout if the bet wins.
        cost: The cost of placing the bet.
    """
    try:
        probability = float(probability)
        payout = float(payout)
        cost = float(cost)
        expected_value = (probability * payout) - cost
        roi = (expected_value / cost * 100) if cost > 0 else 0.0
        return json.dumps(
            {
                "probability": probability,
                "payout": payout,
                "cost": cost,
                "expected_value": round(expected_value, 6),
                "roi_percent": round(roi, 2),
                "profitable": expected_value > 0,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def calculator(expression: str) -> str:
    """A simple calculator that can evaluate mathematical expressions.

    Supports +, -, *, /, **, %, parentheses, and common math functions
    (abs, round, min, max, sqrt via Python builtins).

    Args:
        expression: A string containing a mathematical expression.
    """
    try:
        # Restrict to safe operations only
        allowed = re.compile(r"^[\d\s+\-*/().,%eE]+$")
        if not allowed.match(expression.replace(" ", "")):
            return json.dumps(
                {"error": "Expression contains disallowed characters"}, indent=2
            )

        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def news_search(query: str, lookback_days: int = 14) -> str:
    """Search news articles for a topic.

    Args:
        query: The news query string.
        lookback_days: Days to look back.
    """
    try:
        from MCP.time_gated_retrieval import live_shaped_news_search

        return live_shaped_news_search(query, lookback_days)
    except Exception as exc:
        try:
            from RL_realtime_strategies.MCP.time_gated_retrieval import (
                live_shaped_news_search as _live_news,
            )

            return _live_news(query, lookback_days)
        except Exception:
            return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def websearch_gated(query: str, time_filter: str | None = None) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: The search query.
        time_filter: Optional. Use 'd' for past day, 'w' for past week, 'm' for past month, 'y' for past year.
                     Leave null for no time limit.
    """
    try:
        from MCP.time_gated_retrieval import live_shaped_websearch

        return live_shaped_websearch(query, time_filter)
    except Exception as exc:
        try:
            from RL_realtime_strategies.MCP.time_gated_retrieval import (
                live_shaped_websearch as _live_web,
            )

            return _live_web(query, time_filter)
        except Exception:
            return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def search_x_twitter_gated(query: str) -> str:
    """Search X (formerly Twitter) for recent posts about a topic.

    Args:
        query: The search query string (hashtag, keyword, or phrase).
    """
    try:
        from MCP.time_gated_retrieval import live_shaped_twitter_search

        return live_shaped_twitter_search(query)
    except Exception as exc:
        try:
            from RL_realtime_strategies.MCP.time_gated_retrieval import (
                live_shaped_twitter_search as _live_tw,
            )

            return _live_tw(query)
        except Exception:
            return json.dumps({"error": str(exc)}, indent=2)


if __name__ == "__main__":
    mcp.run()
    # print(search_active_markets("will France win the world cup 2026"))
    # print(get_resolution_rules("0x9b6fef249040fd17e9c107955b37ac2c3e923509b6b0ff01cc463a331ddeb894"))
    # print(websearch("will France win the world cup 2026"))
