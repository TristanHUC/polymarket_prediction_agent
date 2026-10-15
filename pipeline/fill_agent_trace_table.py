"""Fill the RL agent traces table with strategies from a remote vLLM.

Workflow:
  1. Load a top_k_markets JSON snapshot.
  2. For each market, send the market context + tools to the vLLM agent.
  3. The agent uses a tool-calling loop to research, then produces 5 strategies.
  4. For each strategy: look up the entry price / liquidity and INSERT
     into ``agent_traces`` (eval fields left as PENDING for later grading).

Usage:
    uv run python -m RL_realtime_strategies.pipeline.fill_agent_trace_table \\
        top_k_markets_2026-06-23.json          # specific file
    uv run python -m RL_realtime_strategies.pipeline.fill_agent_trace_table \\
        --latest                                # newest snapshot in data/top_k_active_market/

Configuration is read from ``RL_realtime_strategies/.env``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
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

# MCP server — single source of truth for tool definitions & implementations
from MCP.polymarket_mcp import (
    calculate_expected_value,
    calculator,
    get_market_liquidity_and_fees,
    get_order_book,
    get_price_history,
    mcp,
    read_webpage,
    search_active_markets,
    search_x_twitter,
    websearch,
)
from db.db_connection import execute

# Additive gated retrieval (historical pretrain vs live finetune).
# The agent never passes a cutoff; the server resolves it from per-market
# context or SIMULATED_AS_OF_DATE and returns live-shaped payloads.
from MCP.time_gated_retrieval import (
    current_mode as _gated_mode,
    live_shaped_news_search as _live_news,
    live_shaped_read_webpage as _live_page,
    live_shaped_twitter_search as _live_twitter,
    live_shaped_websearch as _live_web,
    set_current_as_of_date as _set_cutoff,
)

# ---------------------------------------------------------------------------
# Config — loaded from .env
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

_VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://100.116.30.98:8000/v1")
_VLLM_MODEL: str = os.getenv("VLLM_MODEL_NAME", "qwen3.6-27b")
_VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "not-needed")

_PORTFOLIO_USD: int = int(os.getenv("AGENT_PORTFOLIO_USD", "1000"))
_MODEL_VERSION: str = os.getenv("AGENT_MODEL_VERSION", "base_qwen3.6_27B")

_TOP_K_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "top_k_active_market"

_CLOB_HOST: str = "https://clob.polymarket.com"
_USER_AGENT: str = "polybrrr-rl/0.1"
_REQUEST_TIMEOUT: int = 30
_MAX_RETRIES: int = 3
_RETRY_SLEEP: float = 1.0

# vLLM config
_LLM_MAX_TOKENS: int = 4096
_LLM_TEMPERATURE: float = 0.7  # lower = more focused
_LLM_TOP_P: float = 0.9  # lower = more focused
_LLM_MAX_TOOL_ROUNDS: int = 10  # 1 to research + 1 more + final answer
_VLLM_TIMEOUT: int = 300  # seconds per call (was 120)
_MAX_PARSE_RETRIES: int = 2  # retries when the model produces unparseable JSON

log = logging.getLogger(__name__)


# =============================================================================
# API helpers (shared)
# =============================================================================


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


# =============================================================================
# Tool definitions — extracted from the MCP server (single source of truth)
# =============================================================================


_TOOLS: list[dict[str, Any]] = []


def _load_openai_tools() -> list[dict[str, Any]]:
    """Extract tool definitions from the MCP server and convert to OpenAI format.

    Cached in ``_TOOLS`` so the async extraction only runs once.
    """
    global _TOOLS
    if _TOOLS:
        return _TOOLS

    async def _fetch() -> list[dict[str, Any]]:
        mcp_tools = await mcp.list_tools()
        converted: list[dict[str, Any]] = []
        for t in mcp_tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema if t.inputSchema else {},
                    },
                }
            )
        return converted

    _TOOLS = asyncio.run(_fetch())
    log.info("Loaded %d tools from MCP server", len(_TOOLS))
    return _TOOLS


# =============================================================================
# Tool executor — maps tool name → implementation
# =============================================================================


def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a single tool call — delegates to the MCP module's implementations."""
    try:
        if name == "get_order_book":
            return get_order_book(args["condition_id"])

        elif name == "get_market_liquidity_and_fees":
            return get_market_liquidity_and_fees(args["condition_id"])

        elif name == "websearch":
            return websearch(args["query"], args["time_filter"])

        elif name == "read_webpage":
            return read_webpage(args["url"])

        elif name == "search_x_twitter":
            return search_x_twitter(args["query"])

        elif name == "get_price_history":
            return get_price_history(args["condition_id"])

        elif name == "search_active_markets":
            return search_active_markets(args["query"])

        elif name == "calculate_expected_value":
            return calculate_expected_value(
                float(args["probability"]),
                float(args["payout"]),
                float(args["cost"]),
            )

        elif name == "calculator":
            return calculator(args["expression"])

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _resolve_as_of_date(
    market: dict[str, Any] | None, explicit: str | None = None
) -> str | None:
    """Resolve the historical cutoff date without changing live behaviour.

    Priority is an explicit override, then ``SIMULATED_AS_OF_DATE`` env, then
    per-market ``as_of_date`` or ``created_at`` fields. Returns None for live
    mode so callers fall back to the existing live tools.
    """
    if explicit:
        return explicit
    env_default = os.getenv("SIMULATED_AS_OF_DATE", "").strip()
    if env_default:
        return env_default
    if market:
        for key in ("as_of_date", "created_at", "generated_at"):
            value = market.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def execute_tool_with_mode(
    name: str, args: dict[str, Any], as_of_date_iso: str | None = None
) -> str:
    """Route a tool call through the historical proxy or the live path.

    This function is additive. ``RESEARCH_MODE=LIVE`` (the default) calls the
    original ``_execute_tool`` unchanged. ``RESEARCH_MODE=HISTORICAL`` sets a
    server-side cutoff and returns live-shaped payloads, so the agent cannot
    distinguish historical traces from live ones. The cutoff never appears in
    tool schemas or outputs.
    """
    try:
        mode = _gated_mode() if _gated_mode is not None else "LIVE"
        mode_name = str(getattr(mode, "value", mode)).upper()
    except Exception:
        mode_name = "LIVE"
    if mode_name != "HISTORICAL":
        if name == "news_search" and _live_news is not None:
            return _live_news(
                str(args.get("query", "")),
                int(args.get("lookback_days", 14)),
            )
        if name == "websearch_gated":
            return _execute_tool("websearch", args)
        if name == "search_x_twitter_gated":
            return _execute_tool("search_x_twitter", args)
        return _execute_tool(name, args)
    # Historical pretrain path: publish the cutoff server-side, then serve
    # live-shaped outputs for the same tool names the agent sees live.
    if _set_cutoff is not None:
        try:
            _set_cutoff(as_of_date_iso)
        except Exception:
            pass
    try:
        if name in ("websearch", "websearch_gated") and _live_web is not None:
            return _live_web(str(args.get("query", "")), args.get("time_filter"))
        if name == "news_search" and _live_news is not None:
            return _live_news(
                str(args.get("query", "")),
                int(args.get("lookback_days", 14)),
            )
        if name in ("search_x_twitter", "search_x_twitter_gated") and _live_twitter is not None:
            return _live_twitter(str(args.get("query", "")))
        if name == "read_webpage" and _live_page is not None:
            return _live_page(str(args.get("url", "")))
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        if _set_cutoff is not None:
            try:
                _set_cutoff(None)
            except Exception:
                pass
    return _execute_tool(name, args)


# =============================================================================
# Resolution rules — called during agent initialization, NOT a tool
# =============================================================================

_PROCESSED_MARKETS = Path(os.getenv("POLYBRRR_MARKETS_DIR", "data/processed/market_universe"))


def _latest_snapshot_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    snapshots = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("snapshot=")],
        key=lambda p: p.name,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _markets_parquet() -> Path | None:
    snap = _latest_snapshot_dir(_PROCESSED_MARKETS)
    if snap is None:
        return None
    return snap / "markets.parquet"


def _query_parquet(path: Path, sql: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import duckdb

        con = duckdb.connect(":memory:")
        result = con.execute(sql).fetchdf()
        con.close()
        return result.to_dict(orient="records")
    except Exception as exc:
        return [{"error": str(exc)}]


def _resolution_from_local(condition_id: str) -> str:
    path = _markets_parquet()
    if path is None:
        return json.dumps({"error": "No local market data available"}, indent=2)
    rows = _query_parquet(
        path,
        f"""
        SELECT condition_id, question, description, market_slug,
               end_date, closed, is_resolved, winner_token_id, winner_outcome
        FROM read_parquet('{path.as_posix()}')
        WHERE condition_id = '{condition_id}'
        """,
    )
    if not rows:
        return json.dumps({"error": f"Market not found: {condition_id}"}, indent=2)
    return json.dumps(rows[0], indent=2, default=str)


def get_resolution_rules(condition_id: str) -> str:
    """Get resolution rules for a market (CLOB API with local fallback)."""
    try:
        url = f"{_CLOB_HOST}/markets/{condition_id}"
        data = _get_with_retry(url)

        if "error" in data:
            return _resolution_from_local(condition_id)

        tokens = data.get("tokens", []) or []
        return json.dumps(
            {
                "condition_id": condition_id,
                "question": data.get("question"),
                "description": data.get("description"),
                "market_slug": data.get("market_slug"),
                "end_date": data.get("end_date_iso"),
                "closed": data.get("closed"),
                "enable_order_book": data.get("enable_order_book"),
                "outcomes": [
                    {"token_id": t.get("token_id"), "outcome": t.get("outcome")} for t in tokens
                ],
            },
            indent=2,
        )
    except Exception:
        return _resolution_from_local(condition_id)


# =============================================================================
# System prompts — base + 5 personas (one independent run per persona)
# =============================================================================

_BASE_PROMPT = (
    "You are a professional algorithmic trader operating on Polymarket, "
    "a prediction-market platform.\n\n"
    "## Your task\n"
    "You are given a market (condition_id, question, description, outcomes, end_date).\n"
    "You must produce exactly **ONE trading strategy** for this market.\n\n"
    "## Constraints\n"
    "- You can only **BUY** tokens (YES or NO). They will be **sold automatically\n"
    "  24 hours later** at whatever the market price is at that time.\n"
    "- Your goal is to maximize profit — or at minimum avoid losing money —\n"
    "  over that 24-hour window.\n"
    "- ``bet_amount_usd`` is the dollar amount you want to deploy.\n"
    "  It must not exceed the available order-book liquidity (check the\n"
    "  ORDER BOOK TOOLS to see how much depth is available).\n"
    "- ``confidence_score`` (0-100) reflects your conviction — it is\n"
    "  informational only and does NOT directly control bet sizing.\n"
    "- If you are not confident enough, you may **PASS** (no trade).\n\n"
    "## Available tools\n"
    "You have access to tools that let you:\n"
    "- Inspect the current order book (midpoint prices, spreads, depth).\n"
    "- Check liquidity depth and taker-fee schedules.\n"
    "- Search the web for news and facts.\n"
    "- Search X/Twitter for real-time sentiment.\n"
    "- Calculate expected value and run arithmetic.\n\n"
    "**Use these tools to gather key information, but do NOT over-research.**\n"
    "## Output format (MANDATORY)\n"
    "Output a single JSON object — no extra commentary:\n\n"
    "```json\n"
    "{{\n"
    '  "strategy_name": "Short label",\n'
    '  "reasoning": "Why you chose this trade (2-5 sentences).",\n'
    '  "final_decision": "BUY YES",\n'
    '  "token_id": "<token_id for YES or NO from the market data>",\n'
    '  "bet_amount_usd": <dollar amount, capped at visible liquidity>,\n'
    '  "confidence_score": <0-100, conviction level>\n'
    "}}\n"
    "```\n\n"
    "``final_decision`` MUST be one of: ``BUY YES``, ``BUY NO``, or ``PASS``.\n"
    "For ``PASS``, set ``bet_amount_usd`` to 0 and ``token_id`` to null.\n\n"
    "Output ONLY the JSON — no extra commentary."
)

# Five independent personas — each runs from scratch with its own system prompt.
_PERSONAS: list[tuple[str, str]] = [
    (
        "Base Neutral",
        "Analyze the market objectively using your tools and make the most logical decision.",
    ),
    (
        "Contrarian",
        "You are a contrarian trader. The market is often wrong and overreacts "
        "to hype. Use your tools to find evidence that the current favorite "
        "outcome will fail. Look for hidden risks.",
    ),
    (
        "Momentum Trader",
        "You are a momentum trader. You believe the crowd is usually right. "
        "Use your tools to find breaking news or social media sentiment that "
        "confirms the current price trend is going to accelerate.",
    ),
    (
        "Quantitative Arbitrager",
        "You are highly risk-averse. Focus heavily on the "
        "get_market_liquidity_and_fees and calculate_expected_value tools. "
        "Only take a position if the spread is tight and the mathematical "
        "expected value is exceptionally high. Otherwise, PASS.",
    ),
    (
        "Social Sentiment Scraper",
        "You ignore traditional news. You believe markets are driven purely by "
        "social media mechanics. Rely heavily on the search_x_twitter tool to "
        "gauge retail emotion before making a trade.",
    ),
]


# =============================================================================
# vLLM client (OpenAI-compatible)
# =============================================================================


def _vllm_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Call the vLLM chat completions endpoint.  Returns the parsed response."""
    payload: dict[str, Any] = {
        "model": _VLLM_MODEL,
        "messages": messages,
        "max_tokens": _LLM_MAX_TOKENS,
        "temperature": _LLM_TEMPERATURE,
        "top_p": _LLM_TOP_P,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    url = f"{_VLLM_BASE_URL}/chat/completions"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {_VLLM_API_KEY}",
                },
                timeout=_VLLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("vLLM call attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_SLEEP * attempt)
    raise RuntimeError(f"vLLM call failed after {_MAX_RETRIES} retries")


# =============================================================================
# Agent loop — tool-calling cycle
# =============================================================================


def _run_agent(
    system_prompt: str, market_context: str
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run the agent with tool-calling loop + JSON-correction retries.

    Returns ``(parsed_strategy, full_message_trace)``.  ``parsed_strategy`` is
    ``None`` when all parse retries are exhausted; the trace always contains the
    full conversation history even on failure.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": market_context},
    ]

    parse_failures = 0

    for _round in range(_LLM_MAX_TOOL_ROUNDS):
        force_final = _round >= _LLM_MAX_TOOL_ROUNDS - 1
        tools = _load_openai_tools() if not force_final else None

        if force_final and _round == _LLM_MAX_TOOL_ROUNDS - 1:
            messages.append(
                {
                    "role": "user",
                    "content": "STOP calling tools now. Produce your final JSON IMMEDIATELY.",
                }
            )

        log.info(
            "Agent round %d/%d — %d messages%s",
            _round + 1,
            _LLM_MAX_TOOL_ROUNDS,
            len(messages),
            " (FORCED FINAL — no tools)" if force_final else "",
        )
        response = _vllm_chat(messages, tools=tools)

        choice = response["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if tool_calls and finish_reason == "tool_calls" and not force_final:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                log.info("  Tool call: %s(%s)", fn_name, json.dumps(fn_args)[:120])
                result = _execute_tool(fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            continue

        if finish_reason in ("stop", "length"):
            content = msg.get("content", "")
            parsed, parse_err = _parse_single_strategy(content)
            if parsed is not None:
                return parsed, messages

            # Parse failed — feed the specific error back and retry
            parse_failures += 1
            if parse_failures <= _MAX_PARSE_RETRIES:
                log.warning(
                    "  JSON parse failed (attempt %d/%d): %s",
                    parse_failures,
                    _MAX_PARSE_RETRIES,
                    parse_err,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last response was NOT valid JSON.\n\n"
                            f"Error: {parse_err}\n\n"
                            "Fix the error and output ONLY a raw JSON object "
                            "— no markdown, no extra text, just the JSON."
                        ),
                    }
                )
                continue
            else:
                log.error("  JSON parse failed after %d retries", _MAX_PARSE_RETRIES)
                return None, messages

        log.warning("finish_reason=%s — attempting to parse", finish_reason)
        content = msg.get("content", "")
        if content:
            parsed, _parse_err = _parse_single_strategy(content)
            if parsed is not None:
                return parsed, messages

    log.error("Agent did not produce a final answer within %d rounds", _LLM_MAX_TOOL_ROUNDS)
    return None, messages


def _parse_single_strategy(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a single-strategy JSON object from the LLM output.

    Returns ``(parsed_dict, error_message)``.  ``parsed_dict`` is ``None`` on
    failure; ``error_message`` carries the parse error for the correction retry.
    """
    # Find any JSON object (no "strategies" wrapper anymore)
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        err = "Could not find a JSON object in the response"
        log.error("Could not find JSON in agent output: %s", content[:500])
        return None, err

    raw = json_match.group(0)
    # Fix common LLM JSON mistakes
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*\]", "]", raw)
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        err = f"JSON parse error: {exc}"
        log.error("Could not parse agent JSON: %s — %s", raw[:500], exc)
        return None, err


# =============================================================================
# Entry price & liquidity lookup
# =============================================================================


def _fetch_entry_data(condition_id: str, token_id: str) -> tuple[float | None, float | None]:
    """Return (entry_price, entry_liquidity) for a token.

    entry_price    = midpoint from /midpoint
    entry_liquidity = total bid+ask size within 20% of midpoint
    """
    try:
        mp = _get_with_retry(f"{_CLOB_HOST}/midpoint", {"token_id": token_id})
        midpoint = float(mp.get("mid")) if "mid" in mp else None

        book = _get_with_retry(f"{_CLOB_HOST}/book", {"token_id": token_id})
        bids = book.get("bids", []) or []
        asks = book.get("asks", []) or []

        liquidity = 0.0
        if midpoint is not None:
            for side in [bids, asks]:
                for lvl in side:
                    p = float(lvl["price"])
                    if abs(p - midpoint) / midpoint <= 0.20:
                        liquidity += float(lvl["size"])

        return midpoint, liquidity if liquidity > 0 else None
    except Exception as exc:
        log.warning("Failed to fetch entry data for %s: %s", token_id, exc)
        return None, None


# =============================================================================
# Database insert
# =============================================================================


def _insert_trace(
    condition_id: str,
    question: str,
    description: str,
    end_date_iso: str | None,
    outcomes: str,  # JSON string
    strategy: dict[str, Any],
    agent_full_trace: str,  # JSON string of full conversation
    entry_price: float | None,
    entry_liquidity: float | None,
) -> None:
    """Insert one strategy row into agent_traces."""
    trace_id = str(uuid.uuid4())
    now_iso = datetime.now(UTC).isoformat()

    execute(
        """
        INSERT INTO agent_traces (
            trace_id, condition_id, model_version, created_at,
            question, description, end_date_iso, outcomes,
            agent_full_trace,
            predicted_token_id, predicted_action, confidence_score,
            bet_amount_usd,
            entry_price, entry_liquidity,
            eval_status
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s,
            %s, %s, %s,
            %s,
            %s, %s,
            'PENDING'
        )
        """,
        (
            trace_id,
            condition_id,
            _MODEL_VERSION,
            now_iso,
            question,
            description,
            end_date_iso,
            outcomes,
            agent_full_trace,
            strategy.get("token_id"),
            strategy.get("final_decision"),
            strategy.get("confidence_score"),
            strategy.get("bet_amount_usd"),
            entry_price,
            entry_liquidity,
        ),
    )
    log.info("  Inserted trace %s → %s", trace_id, strategy.get("strategy_name", ""))


# =============================================================================
# Main orchestrator
# =============================================================================


def _load_market_snapshot(filepath: Path) -> list[dict[str, Any]]:
    """Load the top-k markets JSON snapshot."""
    if not filepath.exists():
        raise FileNotFoundError(f"Market snapshot not found: {filepath}")
    data = json.loads(filepath.read_text())
    return data.get("markets", [])


def process_market(market: dict[str, Any]) -> int:
    """Run 5 independent personas on one market, insert one trace each."""
    condition_id = market["condition_id"]
    question = market.get("question", "")
    end_date = market.get("end_date_iso", "")

    log.info("─" * 70)
    log.info("Processing: %s", question[:100])
    log.info("  condition_id: %s", condition_id)

    # 1. Get resolution rules
    rules_json = get_resolution_rules(condition_id)
    rules = json.loads(rules_json) if isinstance(rules_json, str) else rules_json
    description = rules.get("description", "") if isinstance(rules, dict) else ""
    outcomes = json.dumps(rules.get("outcomes", [])) if isinstance(rules, dict) else "[]"

    # 2. Build the market context (shared across all personas)
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    market_context = f"""
MARKET TO EVALUATE
==================
Today: {today_str}
Condition ID: {condition_id}
Question: {question}
Description: {description}
End Date: {end_date}
Outcomes: {outcomes}

Your confidence_score controls what fraction of the visible order-book depth
you deploy (0-100).  Higher confidence = larger position.

Research this market using the available tools before deciding.
"""

    inserted = 0
    for strategy_num, (persona_name, persona_prompt) in enumerate(_PERSONAS, 1):
        log.info("  Persona %d/5: %s", strategy_num, persona_name)

        # Build the full system prompt: base instructions + persona flavor
        system_prompt = _BASE_PROMPT + "\n\n" + persona_prompt

        # 3. Run this persona independently (catch failures so one crash
        #    doesn't abort the whole market)
        try:
            strategy, messages_trace = _run_agent(system_prompt, market_context)
        except Exception as exc:
            log.error("  Persona '%s' crashed: %s", persona_name, exc)
            strategy = None
            messages_trace = []

        if strategy is None:
            log.warning("  Persona '%s' failed to produce a parseable strategy", persona_name)
            # Still record a PASS trace with the raw conversation
            strategy = {
                "strategy_name": f"FAILED: {persona_name}",
                "reasoning": "Agent failed to produce valid JSON",
                "final_decision": "PASS",
                "token_id": None,
                "bet_amount_usd": 0,
                "confidence_score": 0,
            }
            messages_trace = messages_trace or []

        strategy["strategy_number"] = strategy_num

        # 4. Serialize the FULL trace: system prompt, market context, tool calls,
        #    tool results, assistant messages, and final output.
        agent_full_trace = json.dumps(
            {
                "persona": persona_name,
                "strategy_number": strategy_num,
                "system_prompt": system_prompt,
                "messages": messages_trace,
                "parsed_output": strategy,
            },
            indent=2,
            default=str,
        )

        # 5. Look up entry data if the agent made a trade
        decision = strategy.get("final_decision", "PASS")
        token_id = strategy.get("token_id")
        entry_price, entry_liquidity = None, None
        if decision in ("BUY YES", "BUY NO") and token_id:
            entry_price, entry_liquidity = _fetch_entry_data(condition_id, token_id)

        _insert_trace(
            condition_id=condition_id,
            question=question,
            description=description,
            end_date_iso=end_date or None,
            outcomes=outcomes,
            strategy=strategy,
            agent_full_trace=agent_full_trace,
            entry_price=entry_price,
            entry_liquidity=entry_liquidity,
        )
        inserted += 1

    return inserted


def main(file_arg: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve the input file
    if file_arg is None or file_arg == "--latest":
        snapshots = sorted(
            _TOP_K_DIR.glob("top_k_markets_*.json"),
            key=lambda p: p.name,
            reverse=True,
        )
        if not snapshots:
            raise FileNotFoundError(f"No market snapshots in {_TOP_K_DIR}")
        filepath = snapshots[0]
        log.info("Using latest snapshot: %s", filepath.name)
    else:
        filepath = _TOP_K_DIR / file_arg

    markets = _load_market_snapshot(filepath)
    log.info("Loaded %d markets from %s", len(markets), filepath.name)

    total_inserted = 0
    for idx, market in enumerate(markets):
        log.info("Market %d/%d", idx + 1, len(markets))
        try:
            n = process_market(market)
            total_inserted += n
        except Exception as exc:
            log.error("Failed to process market %s: %s", market.get("condition_id", "?"), exc)

    log.info("=" * 70)
    log.info("Done — %d traces inserted across %d markets", total_inserted, len(markets))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
