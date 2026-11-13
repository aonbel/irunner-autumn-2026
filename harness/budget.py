"""Token budget accounting.

The competition budget is measured in *output* tokens, while every 10 input
tokens count as one output token.  We keep everything in integer tenths
(``x10``) to avoid rounding drift:

    cost_x10 = output_tokens * 10 + input_tokens

Reasoning tokens count as output.  openJiuwen 0.1.19 exposes
``usage_metadata.reasoning_tokens``; some providers include them in
``completion_tokens`` and some do not, so we use a conservative heuristic.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Optional


class BudgetExhausted(RuntimeError):
    """Raised when the global budget or a per-task cap is exhausted."""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    @property
    def effective_output_tokens(self) -> int:
        out = max(0, int(self.output_tokens))
        reasoning = max(0, int(self.reasoning_tokens))
        if reasoning <= 0:
            return out
        total = max(0, int(self.total_tokens))
        prompt = max(0, int(self.input_tokens))
        # ``total_tokens`` disambiguates whether the provider included
        # reasoning in completion_tokens:
        #   total == prompt + output              -> output already includes it
        #   total == prompt + output + reasoning  -> reported separately
        if total >= prompt + out + reasoning:
            return out + reasoning
        if total >= prompt + out:
            return out
        # Ambiguous provider: be conservative.
        return out + reasoning if reasoning > out else out

    @property
    def cost_x10(self) -> int:
        return self.effective_output_tokens * 10 + max(0, int(self.input_tokens))

    @property
    def cost(self) -> float:
        return self.cost_x10 / 10.0


def estimate_text_tokens(text: str) -> int:
    """Conservative token estimate for pre-call budget checks.

    We deliberately do not use tiktoken here: its encodings are downloaded on
    first use, and the competition container has no internet.  Exact token
    counts come from the provider's ``usage_metadata``; this heuristic only
    decides whether a call can safely be attempted.
    """
    if not text:
        return 0
    # ~4 chars/token is typical for English/code; //3 is conservative.
    return max(1, len(text) // 3 + 1)


def estimate_messages_tokens(messages: Any) -> int:
    try:
        rendered = json.dumps(messages, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(messages)
    # Small per-message overhead is a reasonable safety margin.
    return estimate_text_tokens(rendered) + 8 * max(1, len(rendered) // 4000)


class TaskBudget:
    """Per-task cap in cost x10 units."""

    def __init__(self, cap_x10: int):
        self.cap_x10 = max(0, int(cap_x10))
        self.spent_x10 = 0

    @property
    def remaining_x10(self) -> int:
        return max(0, self.cap_x10 - self.spent_x10)

    def can_afford(self, estimate_x10: int) -> bool:
        return self.spent_x10 + max(0, estimate_x10) <= self.cap_x10

    def charge(self, cost_x10: int) -> None:
        self.spent_x10 += max(0, int(cost_x10))

    @property
    def spent(self) -> float:
        return self.spent_x10 / 10.0


class TokenMeter:
    """Global meter with reservations for in-flight tasks."""

    def __init__(
        self,
        budget: int,
        *,
        reserve: int = 5_000,
        safety_factor: float = 1.10,
    ):
        self.budget_x10 = max(0, int(budget)) * 10
        # Keep the reserve at 5% of the actual budget, even if the caller's
        # default reserve is larger (important for small qualifier budgets).
        self.reserve_x10 = min(
            max(0, int(reserve)) * 10,
            max(0, self.budget_x10) // 20,
        )
        self.safety_factor = max(1.0, float(safety_factor))
        self.spent_x10 = 0
        self._reserved: dict[str, int] = {}
        self._task_spent: dict[str, int] = {}
        self._lock = threading.RLock()
        self.calls = 0
        self.estimated_calls = 0

    # -- accounting ---------------------------------------------------------
    def record(
        self,
        task_id: str,
        usage: TokenUsage | Any,
        *,
        estimated: Optional[bool] = None,
    ) -> int:
        """Record a completed LLM call and return its cost in x10 units."""
        if not isinstance(usage, TokenUsage):
            usage = TokenUsage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                reasoning_tokens=int(getattr(usage, "reasoning_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                estimated=bool(estimated),
            )
        cost = usage.cost_x10
        with self._lock:
            self.spent_x10 += cost
            self._task_spent[task_id] = self._task_spent.get(task_id, 0) + cost
            self.calls += 1
            if usage.estimated:
                self.estimated_calls += 1
        return cost

    def charge_estimate(self, task_id: str, cost_x10: int) -> None:
        """Account for a call whose usage could not be read."""
        with self._lock:
            self.spent_x10 += max(0, int(cost_x10))
            self._task_spent[task_id] = self._task_spent.get(task_id, 0) + max(0, int(cost_x10))
            self.calls += 1
            self.estimated_calls += 1

    # -- reservations -------------------------------------------------------
    @property
    def reserved_x10(self) -> int:
        with self._lock:
            return sum(self._reserved.values())

    @property
    def remaining_x10(self) -> int:
        with self._lock:
            return max(
                0,
                self.budget_x10 - self.reserve_x10 - self.spent_x10 - sum(self._reserved.values()),
            )

    @property
    def remaining(self) -> float:
        return self.remaining_x10 / 10.0

    def reserve_task(self, task_id: str, estimate_x10: int) -> bool:
        estimate = max(0, int(estimate_x10))
        with self._lock:
            if task_id in self._reserved:
                return True
            if estimate > self.remaining_x10:
                return False
            self._reserved[task_id] = estimate
            return True

    def release_task(self, task_id: str) -> int:
        with self._lock:
            return self._reserved.pop(task_id, 0)

    def task_spent_x10(self, task_id: str) -> int:
        with self._lock:
            return self._task_spent.get(task_id, 0)

    def can_afford_call(
        self,
        estimate_x10: int,
        task_budget: Optional[TaskBudget] = None,
        *,
        task_id: Optional[str] = None,
    ) -> bool:
        estimate = max(0, int(estimate_x10))
        with self._lock:
            reserved_for_self = self._reserved.get(task_id or "", 0)
            available = (
                self.budget_x10
                - self.reserve_x10
                - self.spent_x10
                - sum(self._reserved.values())
                + reserved_for_self
            )
            if estimate > max(0, available):
                return False
        if task_budget is not None and not task_budget.can_afford(estimate):
            return False
        return True

    def estimate_call_x10(self, messages_tokens: int, max_tokens: int) -> int:
        raw = max(0, int(messages_tokens)) + max(1, int(max_tokens)) * 10
        return int(raw * self.safety_factor) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget": self.budget_x10 / 10.0,
                "reserve": self.reserve_x10 / 10.0,
                "spent": self.spent_x10 / 10.0,
                "remaining": self.remaining_x10 / 10.0,
                "reserved": sum(self._reserved.values()) / 10.0,
                "calls": self.calls,
                "estimated_calls": self.estimated_calls,
            }
