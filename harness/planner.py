"""Global portfolio planner.

The planner treats the remaining tasks as a stochastic 0/1 knapsack:

    maximize   sum(p_i * reward_i)
    subject to sum(cost_i) <= budget - reserve

It computes a shadow price ``lambda`` for the budget and then chooses a
portfolio with a small DP over cost buckets.  ``weight = p*reward - lambda*cost``
is the collective weight used for execution order and dynamic replanning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .config import Config
from .tasks import Task


# Cost priors for tasks that were not solved by the free phase.  These are
# deliberately conservative: the planner should not over-commit the budget.
PRIOR_SUCCESS = {
    "math": 0.65,
    "data_analysis": 0.65,
    "data": 0.65,
    "search": 0.65,
    "coding": 0.55,
    "text": 0.65,
    "text_extraction": 0.65,
    "json": 0.60,
    "json_query": 0.60,
    "sql": 0.55,
    "investigation": 0.45,
    "log_analysis": 0.45,
    "multi_csv": 0.55,
    "data_join": 0.55,
    "spreadsheet": 0.55,
    "xlsx": 0.55,
    "document": 0.60,
    "pdf": 0.60,
    "docx": 0.60,
    "logic": 0.40,
    "constraint": 0.40,
    "generic": 0.40,
    "unknown": 0.40,
}

# Expected share of the per-task cap actually consumed.
COST_FACTOR = {
    "math": 0.75,
    "data_analysis": 0.75,
    "data": 0.75,
    "search": 0.75,
    "coding": 0.85,
    "text": 0.80,
    "text_extraction": 0.80,
    "json": 0.85,
    "json_query": 0.85,
    "sql": 0.90,
    "investigation": 0.90,
    "log_analysis": 0.90,
    "multi_csv": 0.85,
    "data_join": 0.85,
    "spreadsheet": 0.85,
    "xlsx": 0.85,
    "document": 0.85,
    "pdf": 0.85,
    "docx": 0.85,
    "logic": 0.85,
    "constraint": 0.85,
    "generic": 0.75,
    "unknown": 0.75,
}


@dataclass
class PlanItem:
    task: Task
    p: float
    cost_x10: int
    reward: int
    weight: float = 0.0
    selected: bool = False

    @property
    def cost(self) -> float:
        return self.cost_x10 / 10.0

    @property
    def expected_reward(self) -> float:
        return self.p * self.reward


def estimate_p(task: Task) -> float:
    return PRIOR_SUCCESS.get(task.task_type, PRIOR_SUCCESS["unknown"])


def estimate_cost_x10(task: Task, config: Config) -> int:
    cap_units = config.task_cap(task.task_type, task.reward)
    factor = COST_FACTOR.get(task.task_type, 0.75)
    return int(max(1.0, cap_units * factor) * 10)


KNOWN_TASK_TYPES = {
    "math", "data_analysis", "data", "search", "coding",
    "text", "text_extraction", "json", "json_query", "sql",
    "investigation", "log_analysis", "multi_csv", "data_join",
    "spreadsheet", "xlsx", "document", "pdf", "docx", "logic", "constraint",
}


def make_items(tasks: Iterable[Task], config: Config) -> list[PlanItem]:
    items: list[PlanItem] = []
    for task in tasks:
        if getattr(task, "_done", False) or getattr(task, "_impossible", False):
            continue
        # The final set is limited to four known types.  Unknown types have no
        # verifiable solver (generic is disabled by default), so do not spend
        # a worker slot or budget on them.
        if task.task_type not in KNOWN_TASK_TYPES:
            continue
        p = getattr(task, "_p_estimate", None)
        cost_x10 = getattr(task, "_cost_estimate_x10", None)
        items.append(
            PlanItem(
                task=task,
                p=float(p if p is not None else estimate_p(task)),
                cost_x10=int(cost_x10 if cost_x10 is not None else estimate_cost_x10(task, config)),
                reward=max(0, int(task.reward)),
            )
        )
    return items


def solve_lambda(items: list[PlanItem], budget_x10: int) -> float:
    if not items or budget_x10 <= 0:
        return 0.0
    total_cost = sum(max(1, item.cost_x10) for item in items)
    if total_cost <= budget_x10:
        return 0.0
    ratios = [
        (item.p * item.reward) / max(1, item.cost_x10)
        for item in items
        if item.p > 0 and item.reward > 0
    ]
    if not ratios:
        return 0.0
    lo, hi = 0.0, max(ratios) * 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        planned = sum(
            max(1, item.cost_x10)
            for item in items
            if item.p * item.reward - mid * max(1, item.cost_x10) > 0
        )
        if planned > budget_x10:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def dp_portfolio(items: list[PlanItem], budget_x10: int, bucket_x10: int = 100) -> list[PlanItem]:
    """0/1 knapsack DP over cost buckets.

    ``bucket_x10=100`` means one bucket is 10 cost units.  With a 100K budget
    this is at most ~9500 buckets and 100 tasks, i.e. still trivial, and the
    rounding error stays small.
    """
    if not items or budget_x10 <= 0:
        return []
    # Keep the DP bounded for unusually large budgets.
    bucket_x10 = max(bucket_x10, int(budget_x10) // 20_000 + 1)
    cap = max(0, int(budget_x10) // bucket_x10)
    if cap == 0:
        return []
    n = len(items)
    costs = [max(1, math.ceil(item.cost_x10 / bucket_x10)) for item in items]
    values = [max(0.0, item.p * item.reward) for item in items]

    # 2D DP for exact reconstruction.  100 x 9500 is a few MB.
    dp2 = [[0.0] * (cap + 1) for _ in range(n + 1)]
    take2 = [[False] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost = costs[i - 1]
        value = values[i - 1]
        for j in range(cap + 1):
            best = dp2[i - 1][j]
            best_take = False
            if j >= cost:
                candidate = dp2[i - 1][j - cost] + value
                if candidate > best + 1e-12:
                    best = candidate
                    best_take = True
            dp2[i][j] = best
            take2[i][j] = best_take

    selected: list[PlanItem] = []
    j = cap
    for i in range(n, 0, -1):
        if take2[i][j]:
            item = items[i - 1]
            item.selected = True
            selected.append(item)
            j -= costs[i - 1]
    return list(reversed(selected))


def build_portfolio(
    items: list[PlanItem],
    budget_x10: int,
    *,
    qualifier_mode: bool = False,
) -> tuple[list[PlanItem], float]:
    if not items:
        return [], 0.0
    if qualifier_mode:
        for item in items:
            item.selected = True
        return sorted(items, key=lambda item: (-item.expected_reward, item.cost_x10)), 0.0

    # Ignore tasks with no expected value; they still consume the budget.
    live = [item for item in items if item.p > 0 and item.reward > 0]
    if not live:
        return [], 0.0

    lam = solve_lambda(live, budget_x10)
    for item in live:
        item.weight = item.p * item.reward - lam * max(1, item.cost_x10)
    selected = dp_portfolio(live, budget_x10)
    for item in live:
        item.selected = item in selected
    selected.sort(key=lambda item: (-item.weight, -item.expected_reward, item.cost_x10))
    return selected, lam


def summarize(items: list[PlanItem], lam: float, budget_x10: int) -> dict:
    selected = [item for item in items if item.selected]
    return {
        "lambda": lam,
        "budget": budget_x10 / 10.0,
        "candidates": len(items),
        "selected": len(selected),
        "expected_reward": round(sum(item.expected_reward for item in selected), 3),
        "planned_cost": round(sum(item.cost_x10 for item in selected) / 10.0, 2),
    }
