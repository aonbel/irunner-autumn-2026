"""Result persistence.

Two things must survive a hard timeout:

* ``answer.txt`` next to ``task.json`` (the primary deliverable);
* a solutions file with one record per attempted task.

Answers are written atomically and flushed to disk after every task.  For
tasks classified as impossible we deliberately do **not** create ``answer.txt``
(the organisers explicitly asked not to).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # The evaluator may read answer.txt from the host as a different user.
        # mkstemp creates mode 600; make the final file world-readable.
        try:
            os.chmod(tmp_name, 0o644)
        except OSError:
            pass
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


@dataclass
class SolutionRecord:
    task_id: str
    task_type: str
    reward: int
    status: str  # solved | attempted | impossible | skipped | failed
    answer: Optional[str] = None
    solved: bool = False
    cost: float = 0.0
    cost_x10: int = 0
    method: str = ""
    detail: str = ""
    wrote_answer: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SolutionStore:
    def __init__(self, out: Optional[Path], *, task_root: Optional[Path] = None):
        self.records: dict[str, SolutionRecord] = {}
        self._lock = threading.RLock()
        self.out_dir: Optional[Path] = None
        self.jsonl_path: Optional[Path] = None
        self.json_path: Optional[Path] = None
        self.task_root = task_root
        self._configure_out(out)

    def _configure_out(self, out: Optional[Path]) -> None:
        if out is None:
            # The regulation command does not always pass/mount OUT.  Prefer
            # a hidden directory inside the writable tasks mount so solutions
            # survive the container; /tmp is only the last-resort fallback.
            candidate = None
            if self.task_root is not None:
                candidate = Path(self.task_root) / ".irunner"
            if candidate is not None:
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    self.out_dir = candidate
                except OSError:
                    self.out_dir = Path("/tmp/irunner")
            else:
                self.out_dir = Path("/tmp/irunner")
        elif out.exists() and out.is_file():
            self.out_dir = out.parent if str(out.parent) else Path(".")
            self.jsonl_path = out
        elif out.exists() and out.is_dir():
            self.out_dir = out
        elif str(out).endswith(os.sep) or out.suffix == "":
            self.out_dir = out
        else:
            # OUT was described as a file in the statement.
            self.out_dir = out.parent if str(out.parent) else Path(".")
            self.jsonl_path = out
        try:
            (self.out_dir or Path("/tmp/irunner")).mkdir(parents=True, exist_ok=True)
        except OSError:
            # OUT may be set but not mounted.  Fall back to the writable
            # tasks mount first, then to /tmp.
            self.jsonl_path = None
            fallback = None
            if self.task_root is not None:
                fallback = Path(self.task_root) / ".irunner"
                try:
                    fallback.mkdir(parents=True, exist_ok=True)
                except OSError:
                    fallback = None
            self.out_dir = fallback or Path("/tmp/irunner")
            self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.jsonl_path is None:
            self.jsonl_path = (self.out_dir or Path("/tmp/irunner")) / "solutions.jsonl"
        self.json_path = (self.out_dir or Path("/tmp/irunner")) / "solutions.json"
        # If OUT itself is named solutions.json, keep the JSONL stream and
        # the snapshot in separate files.
        if self.jsonl_path is not None and self.json_path == self.jsonl_path:
            self.json_path = self.json_path.with_name("solutions_snapshot.json")

    # -- answer.txt ---------------------------------------------------------
    def write_answer(self, task, answer: str) -> bool:
        answer = str(answer).strip()
        if not answer:
            return False
        atomic_write_text(task.answer_path, answer + "\n")
        return True

    def remove_answer(self, task) -> None:
        remove_file(task.answer_path)

    # -- solutions file -----------------------------------------------------
    def record(
        self,
        task,
        *,
        status: str,
        answer: Optional[str] = None,
        solved: bool = False,
        cost_x10: int = 0,
        method: str = "",
        detail: str = "",
        write_answer: Optional[bool] = None,
    ) -> SolutionRecord:
        if write_answer is None:
            write_answer = status in {"solved", "attempted"} and answer is not None

        wrote = False
        if write_answer and answer is not None and str(answer).strip():
            wrote = self.write_answer(task, str(answer))
        elif status == "impossible" or answer is None:
            # Never leave a stale answer for an impossible/unsolved task.
            self.remove_answer(task)

        rec = SolutionRecord(
            task_id=task.task_id,
            task_type=task.task_type,
            reward=task.reward,
            status=status,
            answer=None if answer is None else str(answer),
            solved=bool(solved),
            cost=cost_x10 / 10.0,
            cost_x10=int(cost_x10),
            method=method,
            detail=detail,
            wrote_answer=wrote,
        )
        with self._lock:
            self.records[task.task_id] = rec
            self._append_jsonl(rec)
            self._write_snapshot()
        return rec

    def _append_jsonl(self, rec: SolutionRecord) -> None:
        path = self.jsonl_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(path, 0o644)
            except OSError:
                pass
        except OSError:
            pass

    def _write_snapshot(self) -> None:
        path = self.json_path
        if path is None:
            return
        try:
            payload = {
                "generated_at": time.time(),
                "solutions": {
                    task_id: rec.to_dict() for task_id, rec in sorted(self.records.items())
                },
                "summary": self.summary(),
            }
            atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        except OSError:
            pass

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records.values())
        solved = [r for r in records if r.solved]
        by_type: dict[str, dict[str, Any]] = {}
        for rec in records:
            bucket = by_type.setdefault(
                rec.task_type,
                {"count": 0, "solved": 0, "reward": 0, "cost": 0.0},
            )
            bucket["count"] += 1
            if rec.solved:
                bucket["solved"] += 1
                bucket["reward"] += rec.reward
            bucket["cost"] = round(bucket["cost"] + rec.cost, 2)
        return {
            "tasks_recorded": len(records),
            "solved": len(solved),
            "reward": sum(r.reward for r in solved),
            "cost": round(sum(r.cost for r in records), 2),
            "by_status": {
                status: sum(1 for r in records if r.status == status)
                for status in sorted({r.status for r in records})
            },
            "by_type": by_type,
        }

    def write_summary(self, extra: Optional[dict[str, Any]] = None) -> None:
        summary = self.summary()
        if extra:
            summary.update(extra)
        with self._lock:
            self._write_snapshot()
        path = (self.out_dir or Path("/tmp/irunner")) / "summary.json"
        try:
            atomic_write_text(path, json.dumps(summary, ensure_ascii=False, indent=2))
        except OSError:
            pass
