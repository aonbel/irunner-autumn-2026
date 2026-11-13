from __future__ import annotations

import unittest
from pathlib import Path

from harness.planner import PlanItem, build_portfolio, solve_lambda
from harness.tasks import Task


def make_task(index: int, reward: int) -> Task:
    return Task(
        task_dir=Path("/tmp"),
        task_id=f"t{index}",
        task_type="math",
        instruction="",
        reward=reward,
        check="exact_int",
        work_dir=Path("/tmp"),
        answer_path=Path("/tmp/answer.txt"),
    )


class PlannerTests(unittest.TestCase):
    def test_all_fit_has_zero_lambda(self) -> None:
        items = [
            PlanItem(make_task(0, 2), 0.9, 1_000, 2),
            PlanItem(make_task(1, 3), 0.8, 2_000, 3),
        ]
        selected, lam = build_portfolio(items, 100_000)
        self.assertEqual(len(selected), 2)
        self.assertEqual(lam, 0.0)

    def test_tight_budget_selects_high_value(self) -> None:
        items = [
            PlanItem(make_task(0, 1), 0.5, 5_000, 1),
            PlanItem(make_task(1, 5), 0.9, 20_000, 5),
            PlanItem(make_task(2, 2), 0.8, 10_000, 2),
        ]
        selected, lam = build_portfolio(items, 25_000)
        self.assertGreater(lam, 0.0)
        self.assertIn("t1", {item.task.task_id for item in selected})
        self.assertLessEqual(sum(item.cost_x10 for item in selected), 25_000 + 1_000)

    def test_lambda_increases_as_budget_shrinks(self) -> None:
        items = [PlanItem(make_task(i, 3), 0.8, 10_000, 3) for i in range(4)]
        loose = solve_lambda(items, 100_000)
        tight = solve_lambda(items, 10_000)
        self.assertEqual(loose, 0.0)
        self.assertGreater(tight, 0.0)


if __name__ == "__main__":
    unittest.main()
