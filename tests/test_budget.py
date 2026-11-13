from __future__ import annotations

import unittest

from harness.budget import TaskBudget, TokenMeter, TokenUsage


class BudgetTests(unittest.TestCase):
    def test_cost_formula(self) -> None:
        usage = TokenUsage(input_tokens=50, output_tokens=100)
        self.assertEqual(usage.cost_x10, 1000 + 50)
        self.assertEqual(usage.cost, 105.0)

    def test_reasoning_separate(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=100, reasoning_tokens=200)
        self.assertEqual(usage.effective_output_tokens, 300)
        self.assertEqual(usage.cost_x10, 3000 + 10)

    def test_reasoning_included(self) -> None:
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=200,
            reasoning_tokens=100,
            total_tokens=210,
        )
        self.assertEqual(usage.effective_output_tokens, 200)
        self.assertEqual(usage.cost_x10, 2000 + 10)

    def test_reasoning_separate_with_total(self) -> None:
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=50,
            reasoning_tokens=150,
            total_tokens=210,
        )
        self.assertEqual(usage.effective_output_tokens, 200)
        self.assertEqual(usage.cost_x10, 2000 + 10)

    def test_reserve_is_capped(self) -> None:
        meter = TokenMeter(100_000, reserve=5_000)
        self.assertEqual(meter.reserve_x10, 50_000)
        small = TokenMeter(50, reserve=5_000)
        self.assertEqual(small.reserve_x10, 25)

    def test_task_budget(self) -> None:
        budget = TaskBudget(1000)
        self.assertTrue(budget.can_afford(600))
        budget.charge(600)
        self.assertFalse(budget.can_afford(500))
        self.assertEqual(budget.remaining_x10, 400)

    def test_reservation_roundtrip(self) -> None:
        meter = TokenMeter(100_000, reserve=5_000)
        self.assertTrue(meter.reserve_task("t1", 30_000))
        self.assertEqual(meter.reserved_x10, 30_000)
        meter.release_task("t1")
        self.assertEqual(meter.reserved_x10, 0)


if __name__ == "__main__":
    unittest.main()
