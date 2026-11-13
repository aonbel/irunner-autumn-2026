from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.handlers.coding import CodingHandler
from harness.handlers.data import DataHandler
from harness.handlers.math import MathHandler
from harness.handlers.search import SearchHandler
from harness.tasks import load_tasks


def make_task(root: Path, task_id: str, task_type: str, instruction: str, files: dict[str, str] | None = None):
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    for name, content in (files or {}).items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task_type": task_type,
                "instruction": instruction,
                "reward": 2,
                "check": "exact_int",
            }
        ),
        encoding="utf-8",
    )
    return load_tasks(task_dir)[0]


class ImpossibleTests(unittest.TestCase):
    def test_math_missing_start(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "m",
            "math",
            "A warehouse starts with some boxes. It ships 10 boxes per day for 3 days. How many boxes?",
        )
        result = MathHandler().try_free(task)
        self.assertIsNotNone(result)
        self.assertTrue(result.impossible)

    def test_data_missing_csv(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "d",
            "data_analysis",
            "How many units of product 'widget' were sold in months 1 through 2?",
        )
        result = DataHandler().try_free(task)
        self.assertIsNotNone(result)
        self.assertTrue(result.impossible)

    def test_search_missing_entity_is_left_to_fallback(self) -> None:
        # A name can be spelled differently in an unseen KB; do not call it
        # impossible based on one failed lookup.  The LLM/grep fallback must
        # confirm any answer against the source snippets.
        task = make_task(
            Path(tempfile.mkdtemp()),
            "search_unknown",
            "search",
            "Who is the mentor of Nobody Here?",
            {"kb/person_alice_smith.md": "# Alice Smith\n\nBorn: 1990\n"},
        )
        self.assertIsNone(SearchHandler().try_free(task))

    def test_search_without_kb_is_impossible(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "search_nokb",
            "search",
            "In which year was Alice Smith born?",
        )
        result = SearchHandler().try_free(task)
        self.assertIsNotNone(result)
        self.assertTrue(result.impossible)

    def test_coding_literal_false(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "c",
            "coding",
            "Fix the code.",
            {
                "mylib.py": "def f():\n    return 1\n",
                "test_mylib.py": "import unittest\nclass T(unittest.TestCase):\n    def test_f(self):\n        assert False\n",
            },
        )
        result = CodingHandler().try_free(task)
        self.assertIsNotNone(result)
        self.assertTrue(result.impossible)


if __name__ == "__main__":
    unittest.main()
