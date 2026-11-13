from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.handlers.coding import CodingHandler
from harness.handlers.data import solve_deterministic as solve_data
from harness.handlers.math import solve_deterministic as solve_math
from harness.handlers.search import solve_deterministic as solve_search
from harness.tasks import load_tasks
from harness.verify import run_tests
from tests.helpers import copy_tasks_without_expected


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "statement" / "task-dir" / "task-dir"


class DeterministicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tasks_dir = Path(cls._tmp.name) / "tasks"
        copy_tasks_without_expected(EXAMPLES, cls.tasks_dir)
        cls.tasks = {task.task_id: task for task in load_tasks(cls.tasks_dir)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_no_expected_field(self) -> None:
        for task in self.tasks.values():
            self.assertFalse(hasattr(task, "expected"))
            self.assertNotIn("expected", task.raw)

    def test_math(self) -> None:
        self.assertEqual(solve_math(self.tasks["math_0000"].instruction), 389)
        self.assertEqual(solve_math(self.tasks["math_0004"].instruction), 1735)

    def test_data(self) -> None:
        self.assertEqual(int(solve_data(self.tasks["data_analysis_0002"]).value), 158)
        self.assertEqual(int(solve_data(self.tasks["data_analysis_0006"]).value), 284)

    def test_search(self) -> None:
        self.assertEqual(solve_search(self.tasks["search_0001"]).value, "1983")
        self.assertEqual(solve_search(self.tasks["search_0005"]).value, "Osaka")

    def test_coding_repair(self) -> None:
        for task_id in ("coding_0003", "coding_0007"):
            with tempfile.TemporaryDirectory() as tmp:
                copied = Path(tmp) / task_id
                copy_tasks_without_expected(EXAMPLES / task_id, copied)
                task = load_tasks(copied)[0]
                result = CodingHandler().try_free(task)
                self.assertIsNotNone(result, task_id)
                self.assertTrue(result.solved, task_id)
                self.assertEqual(result.answer, "DONE")
                self.assertTrue(run_tests(task.work_dir).passed, task_id)

    def test_coding_already_passing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "coding_0007"
            copy_tasks_without_expected(EXAMPLES / "coding_0007", copied)
            task = load_tasks(copied)[0]
            # fix it first
            CodingHandler().try_free(task)
            result = CodingHandler().try_free(task)
            self.assertIsNotNone(result)
            self.assertEqual(result.method, "coding_already_passing")


if __name__ == "__main__":
    unittest.main()
