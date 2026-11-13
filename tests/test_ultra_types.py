from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from harness.config import Config
from harness.handlers.base import HandlerContext
from harness.handlers.logic import LogicHandler
from harness.handlers.multi_csv import MultiCsvHandler
from harness.handlers.sql_query import SqlHandler
from harness.tasks import load_tasks


@dataclass
class FakeResult:
    content: str
    cost_x10: int = 1


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    async def call(self, task, messages, **kwargs):
        return FakeResult(self.content)


def make_task(root: Path, task_id: str, task_type: str, instruction: str, files: dict[str, str] | None = None):
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    for name, content in (files or {}).items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "task_type": task_type, "instruction": instruction, "reward": 2, "check": "exact_str"}),
        encoding="utf-8",
    )
    return load_tasks(task_dir)[0]


class UltraTypeTests(unittest.TestCase):
    def test_multi_csv_script(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "mc",
            "multi_csv",
            "How many customers are there?",
            {"customers.csv": "id,name\n1,A\n2,B\n", "orders.csv": "id,customer_id\n1,1\n"},
        )
        script = "import csv\nprint(sum(1 for _ in open('customers.csv')) - 1)"
        result = asyncio.run(MultiCsvHandler().run(task, HandlerContext(Config(tasks_dir=task.task_dir), FakeLLM(script))))
        self.assertTrue(result.solved)
        self.assertEqual(result.answer, "2")

    def test_sql_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(Path(tmp), "sq", "sql", "How many customers?")
            db_path = task.work_dir / "data.db"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE customers (id INTEGER)")
            conn.executemany("INSERT INTO customers VALUES (?)", [(1,), (2,)])
            conn.commit()
            conn.close()
            script = "import sqlite3\nc=sqlite3.connect('data.db')\nprint(c.execute('select count(*) from customers').fetchone()[0])"
            result = asyncio.run(SqlHandler().run(task, HandlerContext(Config(tasks_dir=task.task_dir), FakeLLM(script))))
            self.assertTrue(result.solved)
            self.assertEqual(result.answer, "2")

    def test_logic_answer(self) -> None:
        task = make_task(Path(tempfile.mkdtemp()), "lg", "logic", "What is 2+2?")
        result = asyncio.run(LogicHandler().run(task, HandlerContext(Config(tasks_dir=task.task_dir), FakeLLM("4"))))
        self.assertTrue(result.solved)
        self.assertEqual(result.answer, "4")

    def test_logic_not_found_rejected(self) -> None:
        task = make_task(Path(tempfile.mkdtemp()), "lg2", "logic", "Unknown?")
        result = asyncio.run(LogicHandler().run(task, HandlerContext(Config(tasks_dir=task.task_dir), FakeLLM("NOT_FOUND"))))
        self.assertFalse(result.solved)


if __name__ == "__main__":
    unittest.main()
