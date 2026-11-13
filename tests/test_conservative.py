from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.handlers.data import DataHandler, solve_deterministic as solve_data
from harness.handlers.math import solve_deterministic as solve_math
from harness.handlers.search import (
    SearchHandler,
    _line_grounded,
    solve_deterministic as solve_search,
)
from harness.tasks import load_tasks


def make_task(root: Path, task_id: str, task_type: str, instruction: str, files: dict[str, str] | None = None, check: str = "exact_int"):
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
                "check": check,
            }
        ),
        encoding="utf-8",
    )
    return load_tasks(task_dir)[0]


CSV = "order_id,region,product,month,units,price\n1,west,widget,7,5,10\n2,east,gadget,8,7,20\n"


class ConservativeTests(unittest.TestCase):
    def test_math_unknown_daily_verb_falls_back(self) -> None:
        text = "A warehouse starts with 100 boxes. It gains boxes per day for 2 days. How many at the end?"
        self.assertIsNone(solve_math(text))

    def test_math_semantic_hazard_falls_back(self) -> None:
        text = (
            "A warehouse starts with 100 boxes. It receives 2 deliveries of "
            "87 boxes each, but one delivery is cancelled. How many at the end?"
        )
        self.assertIsNone(solve_math(text))

    def test_math_per_week_falls_back(self) -> None:
        text = (
            "A warehouse starts with 100 boxes. It ships 10 boxes per week "
            "for 3 weeks. How many boxes does it have at the end?"
        )
        self.assertIsNone(solve_math(text))

    def test_data_conditional_quantity_falls_back(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "d0",
            "data_analysis",
            "How many units of product 'widget' were sold in months 7 through 8 for orders above 10 units?",
            {"sales.csv": CSV},
        )
        self.assertIsNone(DataHandler().try_free(task))

    def test_data_missing_units_is_not_impossible(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "d",
            "data_analysis",
            "How many units of product 'widget' were sold in months 7 through 8?",
            {"sales.csv": "order_id,region,product,month,quantity,price\n1,west,widget,7,5,10\n"},
        )
        result = DataHandler().try_free(task)
        self.assertIsNone(result)  # LLM may inspect the actual column name

    def test_data_region_without_filter_falls_back(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "d2",
            "data_analysis",
            "How many units of product 'widget' were sold in months 7 through 8 by region?",
            {"sales.csv": CSV},
        )
        self.assertIsNone(DataHandler().try_free(task))

    def test_data_region_filter_falls_back_to_llm(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "d3",
            "data_analysis",
            "How many units were sold in the west region?",
            {"sales.csv": CSV},
        )
        result = solve_data(task)
        self.assertFalse(result.ok)
        self.assertFalse(result.impossible)

    def test_search_ambiguous_entity_falls_back(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "s",
            "search",
            "In which year was John Smith born?",
            {
                "kb/person_john_smith_1.md": "# John Smith 1\n\nBorn: 1980\n",
                "kb/person_john_smith_2.md": "# John Smith 2\n\nBorn: 1990\n",
            },
            check="exact_str",
        )
        # Substring matching would see two candidates; do not guess.
        self.assertIsNone(SearchHandler().try_free(task))

    def test_line_grounding(self) -> None:
        snippets = "# Olga Kowalski\n\nBorn: 1975\n\n# Fjord Data\n\nHeadquarters: Osaka\n"
        self.assertFalse(_line_grounded("Olga Kowalski", snippets, "Who is the CEO of Fjord Data?"))
        self.assertFalse(_line_grounded("1975", snippets, "What is the favorite color of Olga Kowalski?"))
        self.assertTrue(_line_grounded("Osaka", snippets, "In which city is the headquarters of Fjord Data?"))

    def test_search_known_pattern_missing_field_falls_back(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "s3",
            "search",
            "Who is the CEO of Fjord Data?",
            {"kb/company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Osaka\n"},
            check="exact_str",
        )
        # Ultra-safe: do not declare impossible based on one field lookup.
        # The grounded LLM fallback must return no answer if the field is not
        # present.
        self.assertIsNone(SearchHandler().try_free(task))

    def test_search_exact_entity_is_deterministic(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "s2",
            "search",
            "In which year was Alice Smith born?",
            {"kb/person_alice_smith.md": "# Alice Smith\n\nBorn: 1983\n"},
            check="exact_str",
        )
        result = solve_search(task)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, "1983")


if __name__ == "__main__":
    unittest.main()
