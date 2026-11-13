"""Shared handler result type and context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..llm import LLMRouter


@dataclass
class HandlerResult:
    answer: Optional[str] = None
    solved: bool = False
    impossible: bool = False
    cost_x10: int = 0
    method: str = ""
    detail: str = ""
    confidence: float = 0.0
    attempts: int = 0

    @property
    def cost(self) -> float:
        return self.cost_x10 / 10.0


@dataclass
class HandlerContext:
    config: "Config"
    llm: Optional["LLMRouter"] = None

    @property
    def llm_enabled(self) -> bool:
        return self.llm is not None and not self.config.no_llm


def instruction_core(text: str) -> str:
    """Drop the boilerplate tail before parsing the actual task.

    Example tails: "Work only inside ...", "When you are done ...",
    "Compute it with python3 ...".  They contain numbers like the ``42`` in
    "echo 42 > answer.txt", which must not be parsed as task data.
    """
    return re.split(
        r"\b(?:Work only inside|When you are done|Compute it with|Use bash|"
        r"Use the file tools|Use the document|Use the text|Use the spreadsheet|"
        r"Use the database|Write only)\b",
        text or "",
        maxsplit=1,
        flags=re.I,
    )[0]


def read_answer_file(task, *, consume: bool = False) -> Optional[str]:
    """Read an answer a model may have written on its own.

    Models sometimes follow the task instruction literally and write
    ``answer.txt`` in the working directory instead of returning the answer in
    the response.  Prefer the workspace copy, then the canonical task-dir copy.
    With ``consume=True`` a workspace copy is removed after reading; the
    orchestrator will write the canonical answer next to ``task.json``.
    """
    workspace_path = Path(task.work_dir) / "answer.txt"
    canonical_path = Path(task.answer_path)
    candidates = [workspace_path, canonical_path]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            if consume and path != canonical_path:
                try:
                    path.unlink()
                except OSError:
                    pass
            return text
        except OSError:
            continue
    return None
