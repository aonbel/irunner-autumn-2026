from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.persist import SolutionStore
from harness.tasks import load_tasks


class StoreTests(unittest.TestCase):
    def test_answer_next_to_task_json_and_impossible_removes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "math_1"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "math_1",
                        "task_type": "math",
                        "instruction": "x",
                        "reward": 1,
                        "check": "exact_int",
                    }
                ),
                encoding="utf-8",
            )
            task = load_tasks(task_dir)[0]
            store = SolutionStore(Path(tmp) / "out")
            store.record(task, status="solved", answer="42", solved=True)
            self.assertTrue(task.answer_path.is_file())
            self.assertEqual(task.answer_path.read_text(encoding="utf-8").strip(), "42")

            store.record(task, status="impossible", answer=None)
            self.assertFalse(task.answer_path.exists())
            self.assertTrue((Path(tmp) / "out" / "solutions.json").is_file())

    def test_out_can_be_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(
                json.dumps({"task_id": "t", "task_type": "math", "instruction": "x", "reward": 1, "check": "exact_int"}),
                encoding="utf-8",
            )
            task = load_tasks(task_dir)[0]
            out_file = Path(tmp) / "answers.jsonl"
            store = SolutionStore(out_file)
            store.record(task, status="solved", answer="1", solved=True)
            self.assertTrue(out_file.is_file())
            self.assertIn("t", out_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
