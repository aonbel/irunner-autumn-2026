"""Task discovery and loading.

The evaluator mounts a directory containing one sub-directory per task.  Each
task has a ``task.json`` and (usually) a ``workspace`` directory with the files
the agent may work with.  ``answer.txt`` must be written next to ``task.json``
(confirmed by the organisers).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


IGNORED_DIR_NAMES = {"__MACOSX", ".git", ".svn", "__pycache__", ".pytest_cache"}


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def infer_task_type(task_id: str, instruction: str, files: Iterable[Path]) -> str:
    """Best-effort fallback when ``task_type`` is missing."""
    lowered = instruction.lower()
    file_names = {p.name.lower() for p in files}

    if any(name.startswith("test_") and name.endswith(".py") for name in file_names):
        return "coding"
    if "unittest" in lowered or "fix the bugs" in lowered:
        return "coding"
    if any(name.endswith(".csv") for name in file_names):
        return "data_analysis"
    if "kb/" in lowered or "knowledge base" in lowered:
        return "search"
    if "born" in lowered or "headquarters" in lowered or "mentor" in lowered:
        return "search"
    if any(word in lowered for word in ("boxes", "starts with", "warehouse", "ship")):
        return "math"
    if task_id.startswith("math"):
        return "math"
    if task_id.startswith("data"):
        return "data_analysis"
    if task_id.startswith("coding"):
        return "coding"
    if task_id.startswith("search"):
        return "search"
    return "unknown"


@dataclass
class Task:
    task_dir: Path
    task_id: str
    task_type: str
    instruction: str
    reward: int
    check: str
    work_dir: Path
    answer_path: Path
    files: list[Path] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_coding(self) -> bool:
        return self.task_type == "coding" or self.check == "unittest"

    def read_text(self, relative: str, limit: int = 16_000) -> str:
        path = (self.work_dir / relative).resolve()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if len(text) > limit:
            return text[:limit] + "\n... [truncated] ...\n"
        return text

    def listed_files(self, limit: int = 80) -> list[str]:
        result: list[str] = []
        for path in sorted(self.files):
            try:
                rel = path.relative_to(self.work_dir)
            except ValueError:
                rel = path
            result.append(str(rel))
            if len(result) >= limit:
                break
        return result


def _load_one(task_dir: Path) -> Optional[Task]:
    json_path = task_dir / "task.json"
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    task_id = str(raw.get("task_id") or task_dir.name)
    instruction = str(raw.get("instruction") or "")
    try:
        reward = int(raw.get("reward") or 0)
    except (TypeError, ValueError):
        reward = 0

    workspace = task_dir / "workspace"
    work_dir = workspace if workspace.is_dir() else task_dir

    files: list[Path] = []
    for path in work_dir.rglob("*"):
        if not path.is_file() or _is_ignored(path):
            continue
        if path.name == ".DS_Store":
            continue
        if path.name == "answer.txt":
            continue
        files.append(path)

    task_type = str(raw.get("task_type") or "").strip().lower()
    if not task_type:
        task_type = infer_task_type(task_id, instruction, files)

    # ``expected`` is not part of the final task format and the harness never
    # reads it.  Drop it even if a training fixture accidentally contains it.
    raw.pop("expected", None)

    return Task(
        task_dir=task_dir,
        task_id=task_id,
        task_type=task_type,
        instruction=instruction,
        reward=reward,
        check=str(raw.get("check") or ""),
        work_dir=work_dir,
        answer_path=task_dir / "answer.txt",
        files=files,
        raw=raw,
    )


def discover_tasks(root: Path) -> list[Task]:
    """Recursively find directories containing ``task.json``.

    The provided archive has an extra nesting level and ``__MACOSX`` noise, so
    we deliberately do not assume a flat layout.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"tasks directory does not exist: {root}")

    candidates: list[Path] = []
    if (root / "task.json").is_file():
        candidates.append(root)
    for path in root.rglob("task.json"):
        if _is_ignored(path):
            continue
        parent = path.parent
        if parent not in candidates:
            candidates.append(parent)

    seen: set[Path] = set()
    tasks: list[Task] = []
    for candidate in sorted(candidates, key=lambda p: str(p)):
        if candidate in seen:
            continue
        seen.add(candidate)
        task = _load_one(candidate)
        if task is not None:
            tasks.append(task)

    # If task ids collide (e.g. duplicated archives), disambiguate by path.
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.task_id] = counts.get(task.task_id, 0) + 1
    for task in tasks:
        if counts.get(task.task_id, 0) > 1:
            task.task_id = f"{task.task_id}_{abs(hash(str(task.task_dir))) & 0xFFFF:04x}"

    return tasks


def load_tasks(root: Path) -> list[Task]:
    tasks = discover_tasks(root)
    if not tasks:
        raise RuntimeError(f"no tasks found under {root}")
    return tasks
