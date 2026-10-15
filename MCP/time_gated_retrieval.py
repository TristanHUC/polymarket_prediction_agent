"""Time-gated retrieval for historical (backtest) pretraining.

This module is purely additive. It does not modify or remove any existing
tool in ``MCP/polymarket_mcp.py`` or in the GRPO notebooks.

Two-phase plan:
  Phase A (historical pretrain): ``RESEARCH_MODE=HISTORICAL`` with a hard
    ``as_of_date`` per market. ``websearch`` goes through Tavily/Exa with an
    end-date cutoff plus a deterministic post-filter. ``news_search`` queries
    a historical news API (NewsAPI when ``NEWSAPI_KEY`` is set, otherwise free
    GDELT DOC 2.1) with an explicit date window. ``search_x_twitter`` returns
    a disabled stub so the policy keeps the same action space without leaking
    future sentiment.
  Phase B (live finetune): ``RESEARCH_MODE=LIVE``. The existing live tools are
    used unchanged, including real X/Twitter search.

All secrets come from environment variables. Nothing is hardcoded here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextvars import ContextVar as _ContextVar
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import requests
from pydantic import BaseModel, field_validator

log = logging.getLogger(__name__)

# Server-side cutoff context (never exposed to the agent). Historical runs set
# one cutoff per process via SIMULATED_AS_OF_DATE, optionally refined per
# market via set_current_as_of_date(). Wrappers read it internally so the
# LLM never sees an as_of_date parameter.
_CURRENT_AS_OF_DATE: _ContextVar[str | None] = _ContextVar(
    "polybrrr_as_of_date", default=None
)


def set_current_as_of_date(value: str | None) -> None:
    """Set the server-side cutoff for subsequent gated calls in this context."""
    _CURRENT_AS_OF_DATE.set(value.strip() if isinstance(value, str) and value.strip() else None)


def get_current_as_of_date() -> str | None:
    """Return the server-side cutoff previously set, if any."""
    try:
        return _CURRENT_AS_OF_DATE.get()
    except LookupError:
        return None


def resolve_server_cutoff() -> str:
    """Resolve the effective cutoff without any agent input.

    Priority is the per-market context value, then SIMULATED_AS_OF_DATE, then
    the current time (which makes the gated path behave like live search).
    """
    context_value = get_current_as_of_date()
    if context_value:
        return context_value
    env_value = os.getenv("SIMULATED_AS_OF_DATE", "").strip()
    if env_value:
        return env_value
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Single source of truth for defaults (do not duplicate these literals).
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULTS: int = 8
DEFAULT_NEWS_LOOKBACK_DAYS: int = 14
DEFAULT_REQUEST_TIMEOUT_SECONDS: int = 30
DEFAULT_AUDIT_MAX_CHARS: int = 20000
HISTORICAL_TWITTER_STUB: str = (
    "X/Twitter search is disabled for historical markets to prevent "
    "future-information leakage. Use websearch and news_search instead."
)


class RetrievalMode(StrEnum):
    """Which backend behaviour the router should use."""

    HISTORICAL = "HISTORICAL"
    LIVE = "LIVE"


def current_mode() -> RetrievalMode:
    """Return the active retrieval mode from the environment."""
    raw = os.getenv("RESEARCH_MODE", "LIVE").strip().upper()
    if raw == "HISTORICAL":
        return RetrievalMode.HISTORICAL
    return RetrievalMode.LIVE


# ---------------------------------------------------------------------------
# Pydantic contracts (all fields required explicitly; pass None for unknown).
# ---------------------------------------------------------------------------


class WebsearchItem(BaseModel):
    title: str | None
    url: str | None
    snippet: str | None
    published_date: str | None


class NewsItem(BaseModel):
    title: str | None
    url: str | None
    snippet: str | None
    published_date: str | None
    source: str | None


class GatedBundle(BaseModel):
    query: str
    as_of_date: str
    mode: str
    results: list[dict[str, Any]]
    dropped_future_dated: int
    provider: str
    warning: str | None

    @field_validator("results")
    @classmethod
    def validate_results(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            msg = "results must be a list"
            raise ValueError(msg)
        return value


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_as_of_date(value: str) -> datetime:
    """Parse an ISO as-of date into an aware UTC datetime."""
    text = (value or "").strip()
    if not text:
        msg = "as_of_date must be a non-empty ISO date string"
        raise ValueError(msg)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        msg = f"unparseable as_of_date: {value!r}"
        raise ValueError(msg) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _published_as_date(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Common shapes: 2024-05-01, 2024-05-01T12:00:00Z, 20240501120000, epoch.
    if re.fullmatch(r"\d{10}(\.\d+)?", text):
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except (ValueError, OSError):
            return None
    if re.fullmatch(r"\d{14}", text):
        try:
            return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # GDELT style: 2024-05-01T12:00:00.000Z or 05/01/2024
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(text[: len(fmt) + 4], fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def filter_results_by_cutoff(
    items: list[dict[str, Any]],
    as_of_date: datetime,
) -> tuple[list[dict[str, Any]], int]:
    """Drop items dated after ``as_of_date``; keep undated items with a flag.

    Returns the kept list and the number of dropped future-dated items.
    Undated items are kept but annotated, because dropping them entirely
    would discard most DDGS fallback results. Callers that need strictness
    can set ``TIME_GATED_STRICT=1`` to drop undated items as well.
    """
    strict = os.getenv("TIME_GATED_STRICT", "0").strip() == "1"
    kept: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        pub = _published_as_date(item.get("published_date") or item.get("date"))
        if pub is None:
            if strict:
                dropped += 1
                continue
            flagged = dict(item)
            flagged["date_unverified"] = True
            kept.append(flagged)
            continue
        if pub > as_of_date:
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


_DATE_RE = re.compile(r"\b(19\d{2}|20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")


def audit_output_for_future_dates(output_text: str, as_of_date: datetime) -> dict[str, Any]:
    """Scan tool output for explicit calendar dates after the cutoff.

    This is a best-effort leakage audit, not a proof of cleanliness.
    """
    haystack = (output_text or "")[:DEFAULT_AUDIT_MAX_CHARS]
    hits: list[str] = []
    for match in _DATE_RE.finditer(haystack):
        try:
            candidate = datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=UTC
            )
        except ValueError:
            continue
        if candidate > as_of_date:
            hits.append(match.group(0))
    return {
        "as_of_date": as_of_date.isoformat(),
        "future_date_mentions": hits[:20],
        "count": len(hits),
    }


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _tavily_search(
    query: str, end_date: datetime, max_results: int, timeout: int
) -> tuple[list[dict[str, Any]], str]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        msg = "missing TAVILY_API_KEY"
        raise RuntimeError(msg)
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    resp = requests.post("https://api.tavily.com/search", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items: list[dict[str, Any]] = []
    for result in data.get("results", []) or []:
        items.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "snippet": result.get("content", "")[:2000],
                "published_date": result.get("published_date"),
            }
        )
    return items, "tavily"


def _exa_search(
    query: str, end_date: datetime, max_results: int, timeout: int
) -> tuple[list[dict[str, Any]], str]:
    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        msg = "missing EXA_API_KEY"
        raise RuntimeError(msg)
    payload = {
        "query": query,
        "numResults": max_results,
        "endPublishedDate": end_date.strftime("%Y-%m-%d"),
    }
    resp = requests.post(
        "https://api.exa.ai/search",
        json=payload,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    items: list[dict[str, Any]] = []
    for result in data.get("results", []) or []:
        items.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "snippet": (result.get("text") or "")[:2000],
                "published_date": result.get("publishedDate"),
            }
        )
    return items, "exa"


def _ddgs_fallback(query: str, max_results: int) -> tuple[list[dict[str, Any]], str]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    items = [
        {
            "title": result.get("title"),
            "url": result.get("href"),
            "snippet": result.get("body", ""),
            "published_date": None,
        }
        for result in raw
    ]
    return items, "ddgs-fallback-unverified"


def websearch_time_gated(
    query: str,
    as_of_date_iso: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Run a time-gated web search and return a JSON string.

    Provider order is Tavily, then Exa, then a DDGS fallback that is clearly
    marked unverified. Every path ends with the same deterministic cutoff
    filter, so the LLM guard is defense in depth rather than the guarantee.
    """
    as_of_date = parse_as_of_date(as_of_date_iso)
    timeout = int(os.getenv("TIME_GATED_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)))
    errors: list[str] = []
    items: list[dict[str, Any]] = []
    provider = "none"
    for attempt in (_tavily_search, _exa_search):
        try:
            items, provider = attempt(query, as_of_date, max_results, timeout)
            break
        except Exception as exc:  # noqa: BLE001 - provider fallback needs broad catch
            errors.append(f"{attempt.__name__}: {exc}")
            continue
    if provider == "none":
        try:
            items, provider = _ddgs_fallback(query, max_results)
            errors.append("tavily/exa unavailable; used DDGS fallback (dates unverified)")
        except Exception as exc:  # noqa: BLE001 - must always return JSON, never raise
            return json.dumps({"query": query, "error": str(exc), "tried": errors}, indent=2)
    kept, dropped = filter_results_by_cutoff(items, as_of_date)
    bundle = GatedBundle(
        query=query,
        as_of_date=as_of_date.isoformat(),
        mode=RetrievalMode.HISTORICAL.value,
        results=kept[:max_results],
        dropped_future_dated=dropped,
        provider=provider,
        warning="; ".join(errors) if errors else None,
    )
    return bundle.model_dump_json(indent=2)


def _gdelt_news(
    query: str, start: datetime, end: datetime, max_results: int, timeout: int
) -> tuple[list[dict[str, Any]], str]:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_results,
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    resp = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params=params,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    items: list[dict[str, Any]] = []
    for article in data.get("articles", []) or []:
        items.append(
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "snippet": article.get("title"),
                "published_date": article.get("seendate"),
                "source": article.get("domain"),
            }
        )
    return items, "gdelt-doc-2.1"


def _newsapi_news(
    query: str, start: datetime, end: datetime, max_results: int, timeout: int
) -> tuple[list[dict[str, Any]], str]:
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        msg = "missing NEWSAPI_KEY"
        raise RuntimeError(msg)
    params = {
        "q": query,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "pageSize": max_results,
        "sortBy": "relevancy",
        "apiKey": api_key,
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items: list[dict[str, Any]] = []
    for article in data.get("articles", []) or []:
        items.append(
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "snippet": article.get("description"),
                "published_date": article.get("publishedAt"),
                "source": (article.get("source") or {}).get("name"),
            }
        )
    return items, "newsapi"


def news_search_time_gated(
    query: str,
    as_of_date_iso: str,
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Query historical news with an automatic ``[as_of - lookback, as_of]`` window."""
    as_of_date = parse_as_of_date(as_of_date_iso)
    start = as_of_date - timedelta(days=lookback_days)
    timeout = int(os.getenv("TIME_GATED_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)))
    errors: list[str] = []
    items: list[dict[str, Any]] = []
    provider = "none"
    # Prefer NewsAPI when a key exists (better relevance); otherwise free GDELT.
    for attempt in (_newsapi_news, _gdelt_news):
        try:
            items, provider = attempt(query, start, as_of_date, max_results, timeout)
            break
        except Exception as exc:  # noqa: BLE001 - provider fallback needs broad catch
            errors.append(f"{attempt.__name__}: {exc}")
            continue
    if provider == "none":
        return json.dumps({"query": query, "error": "; ".join(errors)}, indent=2)
    kept, dropped = filter_results_by_cutoff(items, as_of_date)
    bundle = GatedBundle(
        query=query,
        as_of_date=as_of_date.isoformat(),
        mode=RetrievalMode.HISTORICAL.value,
        results=kept[:max_results],
        dropped_future_dated=dropped,
        provider=provider,
        warning="; ".join(errors) if errors else None,
    )
    return bundle.model_dump_json(indent=2)


def search_x_twitter_gated(query: str, as_of_date_iso: str | None) -> str:
    """Return a disabled stub for historical mode; live mode is handled by the caller.

    The stub keeps the tool schema stable across phases so the policy does not
    face a changing action space between pretraining and finetuning.
    """
    if as_of_date_iso is None:
        return json.dumps(
            {"error": "as_of_date is required for gated X search routing"}, indent=2
        )
    return json.dumps(
        {
            "query": query,
            "as_of_date": parse_as_of_date(as_of_date_iso).isoformat(),
            "results": [],
            "count": 0,
            "disabled": True,
            "notice": HISTORICAL_TWITTER_STUB,
        },
        indent=2,
    )


def read_webpage_gated(url: str, as_of_date_iso: str | None) -> str:
    """Annotate a live page read with a cutoff warning for historical mode.

    A live fetch cannot prove its content predates the bet, so the wrapper
    preserves the existing fetch behaviour but prepends an explicit warning
    that the guard layer cannot verify. Prefer archived snapshots for strict
    guarantees; this keeps the tool available without silent leakage.
    """
    if as_of_date_iso is None:
        return json.dumps({"error": "as_of_date is required for gated page reads"}, indent=2)
    as_of_date = parse_as_of_date(as_of_date_iso)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = requests.get(
            jina_url,
            headers={"User-Agent": "polybrrr-rl/0.1", "Accept": "text/markdown"},
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        content = resp.text[:16000]
    except Exception as exc:
        return json.dumps({"url": url, "error": str(exc)}, indent=2)
    audit = audit_output_for_future_dates(content, as_of_date)
    return json.dumps(
        {
            "url": url,
            "as_of_date": as_of_date.isoformat(),
            "warning": (
                "LIVE fetch in historical mode: content may postdate the cutoff. "
                "Treat post-cutoff facts as inadmissible."
            ),
            "future_date_audit": audit,
            "content": content,
        },
        indent=2,
    )


def _strip_to_live_web_shape(bundle_json: str, time_filter: str | None) -> str:
    """Convert an internal gated bundle to the exact live websearch shape.

    Drops provider, as_of_date, dropped counts, warnings, and per-item dates so
    training traces look like live markets. Enforcement details stay in logs.
    """
    try:
        bundle = json.loads(bundle_json)
    except (json.JSONDecodeError, TypeError):
        return bundle_json
    if isinstance(bundle, dict) and "results" in bundle and "provider" in bundle:
        log.info(
            "gated websearch provider=%s dropped_future=%s warning=%s",
            bundle.get("provider"),
            bundle.get("dropped_future_dated"),
            bundle.get("warning"),
        )
        live_results = [
            {"title": item.get("title"), "url": item.get("url"), "snippet": item.get("snippet")}
            for item in bundle.get("results", []) or []
        ]
        return json.dumps(
            {"query": bundle.get("query"), "time_filter": time_filter, "results": live_results},
            indent=2,
        )
    return bundle_json


def _strip_to_live_news_shape(bundle_json: str, query: str) -> str:
    """Convert an internal gated news bundle to a live-shaped news payload."""
    try:
        bundle = json.loads(bundle_json)
    except (json.JSONDecodeError, TypeError):
        return bundle_json
    if isinstance(bundle, dict) and "results" in bundle and "provider" in bundle:
        log.info(
            "gated news provider=%s dropped_future=%s warning=%s",
            bundle.get("provider"),
            bundle.get("dropped_future_dated"),
            bundle.get("warning"),
        )
        live_results = [
            {"title": item.get("title"), "url": item.get("url"), "snippet": item.get("snippet")}
            for item in bundle.get("results", []) or []
        ]
        return json.dumps(
            {"query": query, "results": live_results, "count": len(live_results)},
            indent=2,
        )
    return bundle_json


def live_shaped_websearch(
    query: str,
    time_filter: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Agent-visible web search with a server-side cutoff and live-shaped output."""
    cutoff = resolve_server_cutoff()
    bundle_json = websearch_time_gated(query, cutoff, max_results)
    return _strip_to_live_web_shape(bundle_json, time_filter)


def live_shaped_news_search(
    query: str,
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Agent-visible news search with a server-side cutoff and live-shaped output."""
    cutoff = resolve_server_cutoff()
    bundle_json = news_search_time_gated(query, cutoff, lookback_days, max_results)
    return _strip_to_live_news_shape(bundle_json, query)


def live_shaped_twitter_search(query: str) -> str:
    """Agent-visible X search that is empty in historical mode without saying why."""
    if current_mode() == RetrievalMode.HISTORICAL:
        log.info("gated X search suppressed for cutoff %s", resolve_server_cutoff())
        return json.dumps({"query": query, "results": [], "count": 0}, indent=2)
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(f"site:x.com {query}", max_results=10))
        results = [
            {"title": r["title"], "url": r["href"], "snippet": r.get("body", "")}
            for r in raw
            if "x.com" in r["href"].lower() or "twitter.com" in r["href"].lower()
        ]
    return json.dumps({"query": query, "results": results, "count": len(results)}, indent=2)


def live_shaped_read_webpage(url: str) -> str:
    """Agent-visible page read with server-side audit and unchanged output shape."""
    if current_mode() == RetrievalMode.HISTORICAL:
        try:
            cutoff = parse_as_of_date(resolve_server_cutoff())
        except ValueError:
            cutoff = None
        try:
            jina_url = f"https://r.jina.ai/{url}"
            resp = requests.get(
                jina_url,
                headers={"User-Agent": "polybrrr-rl/0.1", "Accept": "text/markdown"},
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            content = resp.text
            max_chars = 16000
            truncated = len(resp.text) > max_chars
            if truncated:
                content = content[:max_chars] + "\n\n...[CONTENT TRUNCATED FOR LENGTH]..."
            if cutoff is not None:
                audit = audit_output_for_future_dates(content, cutoff)
                log.info("gated page read url=%s future_mentions=%s", url, audit.get("count"))
            return json.dumps({"url": url, "content": content, "truncated": truncated}, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Failed to read webpage: {exc}"}, indent=2)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = requests.get(
            jina_url,
            headers={"User-Agent": "polybrrr-rl/0.1", "Accept": "text/markdown"},
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        content = resp.text
        max_chars = 16000
        truncated = len(resp.text) > max_chars
        if truncated:
            content = content[:max_chars] + "\n\n...[CONTENT TRUNCATED FOR LENGTH]..."
        return json.dumps({"url": url, "content": content, "truncated": truncated}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Failed to read webpage: {exc}"}, indent=2)
