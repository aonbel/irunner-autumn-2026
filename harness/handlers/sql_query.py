"""SQLite query tasks (hard LLM-only type).

The LLM writes a stdlib ``sqlite3`` script; we execute it locally and take the
last non-empty output line.  Schema discovery is done in Python, so the prompt
contains only the tables/columns and a few sample rows.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file
from .data import extract_script


def _db_files(task) -> list[Path]:
    suffixes = {".db", ".sqlite", ".sqlite3"}
    return [path for path in sorted(task.work_dir.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes][:3]


def _schema(path: Path) -> str:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cursor = conn.cursor()
            tables = [
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            chunks = [f"--- {path.name} ---"]
            for table in tables:
                columns = [row[1] for row in cursor.execute(f'PRAGMA table_info("{table}")').fetchall()]
                chunks.append(f"TABLE {table} ({', '.join(columns)})")
                try:
                    sample = cursor.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
                    chunks.append("sample: " + repr(sample))
                except sqlite3.Error:
                    pass
            return "\n".join(chunks)
        finally:
            conn.close()
    except sqlite3.Error:
        return f"--- {path.name} --- (cannot read schema)"


class SqlHandler:
    name = "sql"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="sql_llm", detail="LLM disabled")

        files = _db_files(task)
        if not files:
            return HandlerResult(impossible=True, method="sql_llm", detail="no SQLite database")
        schema = "\n\n".join(_schema(path) for path in files)

        messages = [
            {
                "role": "system",
                "content": (
                    "Write a short Python script using only the standard library "
                    "and sqlite3. It must read the database file in the current "
                    "directory, run the necessary SQL, and print exactly one answer "
                    "value. If the requested table/column is absent, print exactly "
                    "NOT_FOUND. Output only the script."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {task.instruction}\n\nSchema:\n{schema}\n\n"
                    "Use the real file name in the script."
                ),
            },
        ]
        cost = 0
        attempts = 0
        try:
            attempts += 1
            result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=400)
            cost += result.cost_x10
            script = extract_script(result.content or "")
            if not script:
                return HandlerResult(
                    solved=False,
                    cost_x10=cost,
                    method="sql_llm",
                    detail="no script in response",
                    attempts=attempts,
                )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(task.work_dir),
                capture_output=True,
                text=True,
                timeout=25,
            )
            output = (proc.stdout or "").strip()
            last_line = next((line for line in reversed(output.splitlines()) if line.strip()), "")
            if not last_line:
                last_line = (read_answer_file(task, consume=True) or "").strip()
            if last_line.strip().casefold() in {"not_found", "not found", "none", "null"}:
                last_line = ""
            if last_line:
                return HandlerResult(
                    answer=last_line.strip(),
                    solved=True,
                    cost_x10=cost,
                    method="sql_llm_script",
                    confidence=0.7,
                    attempts=attempts,
                    detail=(proc.stderr or "")[-500:],
                )
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="sql_llm",
                detail=f"script produced no answer (rc={proc.returncode})",
                attempts=attempts,
            )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="sql_llm",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
