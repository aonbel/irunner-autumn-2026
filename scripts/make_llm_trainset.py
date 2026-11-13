#!/usr/bin/env python3
"""Generate a small LLM-heavy development set.

These tasks deliberately fall outside the ultra-safe deterministic templates:
conditional/percentage math, non-trivial CSV aggregates, bugs outside the
mutation library, free-text extraction and JSON queries.  They are used to
measure how well the real LLM fallback performs; the final competition set is
limited to the four original task families.

Usage:
    python scripts/make_llm_trainset.py --out /tmp/irunner-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


TRUTH: dict[str, dict] = {}


def write_task(root: Path, task_id: str, task_type: str, instruction: str, check: str, answer: str | None, files: dict[str, str]):
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
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
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    TRUTH[task_id] = {"solvable": answer is not None, "answer": answer}


def math_tasks(root: Path) -> None:
    cases = [
        (
            "A warehouse starts with 100 boxes. It ships 10% of the original stock per day for 3 days. How many boxes does it have at the end?",
            "70",
        ),
        (
            "A warehouse starts with 100 boxes. It receives 3 deliveries of 20 boxes each, but one delivery is cancelled. How many boxes does it have at the end?",
            "140",
        ),
        (
            "A store has 120 items. It sells 1/4 of them, then receives 30 more. How many items does it have at the end?",
            "120",
        ),
        (
            "A warehouse starts with 500 boxes. It ships 12% of the original stock, then 30 boxes. How many boxes does it have at the end?",
            "410",
        ),
        (
            "A warehouse starts with 100 boxes. It ships 10 boxes per day for 3 days. What is the final number of boxes?",
            "70",
        ),
    ]
    for index, (instruction, answer) in enumerate(cases, 1):
        write_task(
            root,
            f"math_{index:04d}",
            "math",
            instruction + " Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt.",
            "exact_int",
            answer,
            {},
        )


def data_tasks(root: Path) -> None:
    rows = [
        (1, "west", "widget", 7, 5, 10),
        (2, "east", "widget", 8, 7, 20),
        (3, "west", "gadget", 7, 3, 40),
        (4, "east", "gadget", 8, 9, 5),
        (5, "west", "widget", 9, 11, 30),
    ]
    header = "order_id,region,product,month,units,price"
    csv_text = header + "\n" + "\n".join(",".join(map(str, row)) for row in rows) + "\n"
    units = [row[4] for row in rows]
    prices = [row[5] for row in rows]
    units_sorted = sorted(units)
    median = units_sorted[len(units_sorted) // 2]
    cases = [
        ("What is the median number of units per order?", str(median)),
        ("What is the average price?", f"{sum(prices) / len(prices):.1f}"),
        ("What is the total revenue (units * price)?", str(sum(u * p for u, p in zip(units, prices)))),
        ("Which product sold the most units?", "widget"),
        ("What is the average number of units per order?", f"{sum(units) / len(units):.1f}"),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The file sales.csv has columns order_id,region,product,month,units,price. "
            + question
            + " Use python3. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        check = "exact_int" if answer.replace(".", "", 1).isdigit() else "exact_str"
        write_task(root, f"data_analysis_{index:04d}", "data_analysis", instruction, check, answer, {"sales.csv": csv_text})


def coding_tasks(root: Path) -> None:
    base_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_mean(self):
        self.assertEqual(m.mean([2, 4, 6]), 4)


if __name__ == '__main__':
    unittest.main()
"""
    cases = [
        ("def total(xs):\n    return 0\n\n\ndef mean(xs):\n    return 0\n", base_test),
        ("def total(xs):\n    return sum(xs) * 2\n\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n", base_test),
        (
            "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n\n\ndef mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
            base_test,
        ),
        (
            "def total(xs):\n    return sum(xs) + 1\n\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n",
            base_test,
        ),
        (
            "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
            base_test,
        ),
    ]
    for index, (code, test) in enumerate(cases, 1):
        instruction = (
            "Run python3 test_mylib.py; some tests fail. Fix the bugs in mylib.py "
            "(do not edit the tests) until all tests pass. Work only inside the "
            "current directory. When you are done, write ONLY the word DONE to answer.txt."
        )
        write_task(
            root,
            f"coding_{index:04d}",
            "coding",
            instruction,
            "unittest",
            "DONE",
            {"mylib.py": code, "test_mylib.py": test},
        )


def text_tasks(root: Path) -> None:
    notes = (
        "Project Orion\n"
        "Owner: Maya Sato\n"
        "Deadline: 2026-03-15\n"
        "Budget: 42000\n"
        "Status: active\n"
    )
    cases = [
        ("Who is the owner of project Orion?", "Maya Sato"),
        ("What is the deadline of project Orion?", "2026-03-15"),
        ("What is the budget of project Orion?", "42000"),
        ("What is the status of project Orion?", "active"),
        ("What is the favorite color of Maya Sato?", None),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The file notes.txt contains project information. "
            + question
            + " Use the text file. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        write_task(root, f"text_{index:04d}", "text", instruction, "exact_str", answer, {"notes.txt": notes})


def json_tasks(root: Path) -> None:
    data = {
        "project": "Orion",
        "owner": "Maya Sato",
        "users": [
            {"name": "A", "active": True, "score": 10},
            {"name": "B", "active": False, "score": 20},
            {"name": "C", "active": True, "score": 30},
        ],
        "budget": 42000,
    }
    text = json.dumps(data)
    cases = [
        ("How many active users are there?", "2"),
        ("What is the owner of the project?", "Maya Sato"),
        ("What is the total score of all users?", "60"),
        ("What is the name of the project?", "Orion"),
        ("How many users are there?", "3"),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The file data.json contains project data. "
            + question
            + " Use python3 with the json module. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        write_task(root, f"json_{index:04d}", "json", instruction, "exact_str", answer, {"data.json": text})


def search_tasks(root: Path) -> None:
    kb = {
        "kb/person_alice_smith.md": "# Alice Smith\n\nBorn: 1983\nEmployer: Dune Robotics\nMentor: Maya Sato\n",
        "kb/person_maya_sato.md": "# Maya Sato\n\nBorn: 1975\nEmployer: Fjord Data\nMentor: Alice Smith\n",
        "kb/company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Hanoi\nIndustry: Robotics\nFounded: 1989\n",
        "kb/company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Osaka\nIndustry: Data\nFounded: 1964\n",
    }
    cases = [
        ("For Dune Robotics, what is the founded value?", "1989"),
        ("Tell me the headquarters city for Fjord Data.", "Osaka"),
        ("Who is the mentor for Alice Smith?", "Maya Sato"),
        ("What is the CEO of Fjord Data?", None),
        ("Who is the mentor of Nobody Here?", None),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The directory kb/ contains a small knowledge base of markdown files. "
            + question
            + " Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        write_task(root, f"search_{index:04d}", "search", instruction, "exact_str", answer, kb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out).expanduser().resolve()
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True)
    math_tasks(root)
    data_tasks(root)
    coding_tasks(root)
    text_tasks(root)
    json_tasks(root)
    search_tasks(root)
    (root / "_truth.json").write_text(json.dumps({"tasks": TRUTH}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(root), "tasks": len(TRUTH)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
