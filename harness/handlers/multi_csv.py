"""Multi-CSV join/aggregation tasks (hard LLM-only type)."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file
from .data import extract_script


def _csv_files(task) -> list[Path]:
    return [path for path in sorted(task.work_dir.rglob("*.csv")) if path.is_file()][:6]


def _schema(path: Path, limit: int = 2) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            rows = []
            for index, row in enumerate(reader):
                rows.append(row)
                if index >= limit:
                    break
        header = rows[0] if rows else []
        sample = rows[1:limit + 1]
        return f"--- {path.name} ---\nheader: {header}\nsample: {sample}"
    except OSError:
        return f"--- {path.name} --- (cannot read)"


class MultiCsvHandler:
    name = "multi_csv"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="multi_csv", detail="LLM disabled")
        files = _csv_files(task)
        if len(files) < 2:
            return HandlerResult(impossible=True, method="multi_csv", detail="less than two CSV files")

        schema = "\n\n".join(_schema(path) for path in files)
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a short Python script using only the standard library "
                    "that reads the CSV files in the current directory, performs "
                    "the join/aggregation, and prints exactly one answer value. "
                    "If the requested data is absent, print exactly NOT_FOUND. "
                    "Output only the script."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {task.instruction}\n\nSchemas:\n{schema}\n\nUse the real file names.",
            },
        ]
        cost = 0
        attempts = 0
        try:
            attempts += 1
            result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=450)
            cost += result.cost_x10
            script = extract_script(result.content or "")
            if not script:
                return HandlerResult(solved=False, cost_x10=cost, method="multi_csv", detail="no script", attempts=attempts)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(task.work_dir),
                capture_output=True,
                text=True,
                timeout=30,
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
                    method="multi_csv_script",
                    confidence=0.65,
                    attempts=attempts,
                    detail=(proc.stderr or "")[-500:],
                )
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="multi_csv",
                detail=f"script produced no answer (rc={proc.returncode})",
                attempts=attempts,
            )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="multi_csv",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
