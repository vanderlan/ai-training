"""
Cost tracker — monitors LLM API usage and enforces budget limits.
Implements Day 5 curriculum cost management patterns.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger("multi_agent.cost")

# Approximate cost per 1K tokens (DeepSeek pricing)
MODEL_COSTS_PER_1K: Dict[str, float] = {
    "deepseek-chat": 0.00014,
    "deepseek-reasoner": 0.00055,
}

CHARS_PER_TOKEN = 4


@dataclass
class CostStats:
    """Running cost statistics."""
    total_calls: int = 0
    total_input_chars: int = 0
    total_output_chars: int = 0
    estimated_cost_usd: float = 0.0
    calls_blocked: int = 0
    model: str = ""


class CostTracker:
    """
    Track estimated LLM costs and enforce a daily budget cap.
    """

    def __init__(self):
        self._daily_budget = float(os.getenv("DAILY_BUDGET_USD", "5.0"))
        self._warning_threshold = 0.8
        self._model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        self._stats = CostStats(model=self._model)
        self._day_start = time.time()
        self._lock = threading.Lock()

    def _reset_if_new_day(self):
        if time.time() - self._day_start > 86400:
            self._stats = CostStats(model=self._model)
            self._day_start = time.time()
            logger.info("Daily cost counters reset")

    def check_budget(self) -> bool:
        """Return True if budget allows another call."""
        with self._lock:
            self._reset_if_new_day()
            if self._stats.estimated_cost_usd >= self._daily_budget:
                self._stats.calls_blocked += 1
                logger.warning(
                    "Daily budget exhausted ($%.4f / $%.2f). Call blocked.",
                    self._stats.estimated_cost_usd, self._daily_budget,
                )
                return False
            if self._stats.estimated_cost_usd >= self._daily_budget * self._warning_threshold:
                logger.warning(
                    "Approaching daily budget: $%.4f / $%.2f (%.0f%%)",
                    self._stats.estimated_cost_usd, self._daily_budget,
                    (self._stats.estimated_cost_usd / self._daily_budget) * 100,
                )
            return True

    def record_call(self, input_text: str, output_text: str):
        """Record an LLM call and estimate its cost."""
        with self._lock:
            self._reset_if_new_day()
            input_chars = len(input_text)
            output_chars = len(output_text)
            total_tokens = (input_chars + output_chars) / CHARS_PER_TOKEN
            cost_per_1k = MODEL_COSTS_PER_1K.get(self._model, 0.00014)
            cost = (total_tokens / 1000) * cost_per_1k

            self._stats.total_calls += 1
            self._stats.total_input_chars += input_chars
            self._stats.total_output_chars += output_chars
            self._stats.estimated_cost_usd += cost

    def get_stats(self) -> dict:
        """Return current cost statistics."""
        with self._lock:
            return {
                "total_calls": self._stats.total_calls,
                "estimated_cost_usd": round(self._stats.estimated_cost_usd, 6),
                "daily_budget_usd": self._daily_budget,
                "calls_blocked": self._stats.calls_blocked,
                "model": self._stats.model,
            }


# Module-level singleton
cost_tracker = CostTracker()
