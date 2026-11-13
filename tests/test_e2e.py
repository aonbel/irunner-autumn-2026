from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from harness.budget import TokenMeter
from harness.config import Config
from harness.llm import LLMRouter
from harness.orchestrator import Orchestrator
from harness.persist import SolutionStore
from harness.tasks import load_tasks
from tests.helpers import copy_tasks_without_expected


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "statement" / "task-dir" / "task-dir"


class EndToEndTests(unittest.TestCase):
    def test_examples_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp) / "tasks"
            copy_tasks_without_expected(EXAMPLES, tasks_dir)
            out = Path(tmp) / "out"
            config = Config(
                tasks_dir=tasks_dir,
                out=out,
                budget=100_000,
                no_llm=True,
                log_dir=Path(tmp) / "logs",
            )
            tasks = load_tasks(tasks_dir)
            meter = TokenMeter(config.budget, reserve=config.reserve)
            llm = LLMRouter(config, meter)
            store = SolutionStore(out)
            orchestrator = Orchestrator(config, tasks, meter, llm, store)
            asyncio.run(orchestrator.run())

            summary = store.summary()
            self.assertEqual(summary["solved"], 8)
            self.assertEqual(summary["reward"], 20)
            self.assertEqual(summary["cost"], 0.0)

            for task in tasks:
                self.assertTrue(task.answer_path.is_file(), task.task_id)
                self.assertEqual(task.answer_path.parent, task.task_dir)
                workspace = task.task_dir / "workspace"
                if workspace.is_dir():
                    self.assertFalse((workspace / "answer.txt").exists(), task.task_id)
                text = task.answer_path.read_text(encoding="utf-8").strip()
                if task.task_type == "coding":
                    self.assertEqual(text, "DONE")
            self.assertTrue((out / "solutions.json").is_file())
            payload = json.loads((out / "solutions.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["solved"], 8)


if __name__ == "__main__":
    unittest.main()
