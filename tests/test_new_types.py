from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from harness.config import Config
from harness.handlers.base import HandlerContext
from harness.handlers.json_query import JsonHandler
from harness.handlers.text import TextHandler
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


def make_task(root: Path, task_id: str, task_type: str, instruction: str, files: dict[str, str]):
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
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
                "check": "exact_str",
            }
        ),
        encoding="utf-8",
    )
    return load_tasks(task_dir)[0]


class NewTypeTests(unittest.TestCase):
    def test_text_grounded_answer(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "text_1",
            "text",
            "Who is the owner of project Orion?",
            {"notes.txt": "Project Orion\nOwner: Maya Sato\nDeadline: 2026-03-15\n"},
        )
        cfg = Config(tasks_dir=task.task_dir)
        result = asyncio.run(TextHandler().run(task, HandlerContext(cfg, FakeLLM("Maya Sato"))))
        self.assertTrue(result.solved)
        self.assertEqual(result.answer, "Maya Sato")

    def test_text_ungrounded_answer_rejected(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "text_2",
            "text",
            "What is the favorite color of Maya Sato?",
            {"notes.txt": "Project Orion\nOwner: Maya Sato\nDeadline: 2026-03-15\n"},
        )
        cfg = Config(tasks_dir=task.task_dir)
        result = asyncio.run(TextHandler().run(task, HandlerContext(cfg, FakeLLM("green"))))
        self.assertFalse(result.solved)
        self.assertIsNone(result.answer)

    def test_json_script_answer(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "json_1",
            "json",
            "How many active users are there?",
            {"data.json": '{"users": [{"active": true}, {"active": false}]}'},
        )
        cfg = Config(tasks_dir=task.task_dir)
        script = "import json\nd=json.load(open('data.json'))\nprint(sum(1 for u in d['users'] if u['active']))"
        result = asyncio.run(JsonHandler().run(task, HandlerContext(cfg, FakeLLM(script))))
        self.assertTrue(result.solved)
        self.assertEqual(result.answer, "1")

    def test_no_llm_does_not_invent_answer(self) -> None:
        task = make_task(
            Path(tempfile.mkdtemp()),
            "text_3",
            "text",
            "Who is the owner?",
            {"notes.txt": "Owner: Maya Sato\n"},
        )
        cfg = Config(tasks_dir=task.task_dir, no_llm=True)
        result = asyncio.run(TextHandler().run(task, HandlerContext(cfg, None)))
        self.assertFalse(result.solved)
        self.assertIsNone(result.answer)


if __name__ == "__main__":
    unittest.main()
