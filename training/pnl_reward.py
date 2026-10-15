"""PnL-based reward for GRPO, additive to the existing calculator reward.

The notebook currently uses ``+1 if calculator called else -1`` as a smoke
test. This module adds the economic reward discussed for phase A/B training
without deleting that helper:

  Phase A (historical pretrain): reward from backtested 24h price movement
    (entry midpoint at T versus exit midpoint at T+24h, minus taker fees).
  Phase B (live finetune): same formula with live entry/exit observations.

The ``pnl_reward_func`` signature matches TRL ``reward_funcs`` conventions
(``completions`` plus aligned per-sample kwargs) so it can replace
``calculator_reward_func`` in ``GRPOTrainer`` without changing the loop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single source of truth for reward shaping (do not duplicate these literals).
# ---------------------------------------------------------------------------

PNL_CLIP_LOW: float = -1.0
PNL_CLIP_HIGH: float = 1.0
INVALID_TRADE_PENALTY: float = -0.5
PASS_REWARD: float = 0.0
DEFAULT_TAKER_FEE_RATE: float = 0.02


def _clip(value: float) -> float:
    return max(PNL_CLIP_LOW, min(PNL_CLIP_HIGH, value))


def _extract_final_json(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    raw = re.sub(r",\s*}", "}", match.group(0))
    raw = re.sub(r",\s*\]", "]", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_trade_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the last assistant JSON decision from a TRL message list."""
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        parsed = _extract_final_json(str(message.get("content") or ""))
        if parsed is not None:
            return parsed
    return {}


def pnl_fraction_for_trade(
    decision: str,
    entry_price: float | None,
    exit_price: float | None,
    fee_rate: float = DEFAULT_TAKER_FEE_RATE,
) -> float:
    """Compute a fee-adjusted fractional PnL for one BUY YES/NO trade.

    Buying ``cost`` at ``entry_price`` yields ``cost / entry_price`` shares,
    sold at ``exit_price`` for ``cost * exit / entry``. Fees are charged on
    notional cost. PASS yields zero. Missing prices yield the invalid-trade
    penalty so the policy learns to respect liquidity constraints.
    """
    action = (decision or "").strip().upper()
    if action == "PASS":
        return PASS_REWARD
    if action not in ("BUY YES", "BUY NO"):
        return INVALID_TRADE_PENALTY
    if entry_price is None or exit_price is None:
        return INVALID_TRADE_PENALTY
    try:
        entry = float(entry_price)
        exit_ = float(exit_price)
        fee = float(fee_rate)
    except (TypeError, ValueError):
        return INVALID_TRADE_PENALTY
    if entry <= 0.0 or entry >= 1.0 or exit_ < 0.0 or exit_ > 1.0:
        return INVALID_TRADE_PENALTY
    gross = exit_ / entry - 1.0
    return _clip(gross - fee)


def pnl_reward_func(
    completions: list[list[dict[str, Any]]],
    entry_prices: list[float | None] | None = None,
    exit_prices: list[float | None] | None = None,
    fee_rates: list[float] | float | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible reward function based on realized PnL.

    ``entry_prices`` and ``exit_prices`` must be aligned with ``completions``.
    When they are absent (for example during a smoke test), the function falls
    back to zero reward rather than raising, so training loops keep running.
    """
    del kwargs
    rewards: list[float] = []
    count = len(completions or [])
    for index in range(count):
        messages = completions[index] if index < len(completions) else []
        trade = parse_trade_from_messages(messages if isinstance(messages, list) else [])
        entry = entry_prices[index] if entry_prices and index < len(entry_prices) else None
        exit_ = exit_prices[index] if exit_prices and index < len(exit_prices) else None
        if isinstance(fee_rates, list):
            fee = fee_rates[index] if index < len(fee_rates) else DEFAULT_TAKER_FEE_RATE
        elif fee_rates is None:
            fee = DEFAULT_TAKER_FEE_RATE
        else:
            fee = float(fee_rates)
        decision = str(trade.get("final_decision", "PASS"))
        rewards.append(pnl_fraction_for_trade(decision, entry, exit_, fee))
    return rewards


def evaluate_pnl_reward(
    traces: list[dict[str, Any]],
    label: str = "",
) -> tuple[float, float]:
    """Score stored traces in place and return ``(mean_reward, pass_rate_pct)``."""
    completions = [trace.get("messages", []) for trace in traces]
    entry_prices = [trace.get("entry_price") for trace in traces]
    exit_prices = [trace.get("exit_price", trace.get("price_24h_later")) for trace in traces]
    rewards = pnl_reward_func(
        completions=completions,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
    )
    for trace, reward in zip(traces, rewards, strict=False):
        trace["reward"] = reward
    total = len(rewards)
    mean_reward = sum(rewards) / total if total else 0.0
    passes = sum(1 for trace in traces if str(trace.get("predicted_action", "")).upper() == "PASS")
    pass_rate = passes / total * 100.0 if total else 0.0
    if label:
        log.info(
            "PnL reward %s: mean=%.4f pass=%.1f%% over %d traces",
            label,
            mean_reward,
            pass_rate,
            total,
        )
    return mean_reward, pass_rate
