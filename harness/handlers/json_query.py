"""JSON query tasks (new LLM-only type).

The LLM writes a tiny stdlib ``json`` script, we execute it in the task
directory and take the last non-empty line as the answer.  This is a
mechanical, locally executed path: the model reasons about the schema, but the
actual computation happens in Python.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file
from .data import extract_script


def _json_files(task) -> list[Path]:
    files = [path for path in sorted(task.work_dir.rglob("*.json")) if path.is_file()]
    return files[:3]


def _json_preview(files: list[Path], limit: int = 6000) -> str:
    chunks: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(f"--- {path.name} ---\n{text[:limit]}")
    return "\n\n".join(chunks)


class JsonHandler:
    name = "json"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="json_llm", detail="LLM disabled")

        files = _json_files(task)
        if not files:
            return HandlerResult(impossible=True, method="json_llm", detail="no JSON files")
        preview = _json_preview(files)
        if not preview.strip():
            return HandlerResult(impossible=True, method="json_llm", detail="empty JSON files")

        messages = [
            {
                "role": "system",
                "content": (
                    "Write a short Python script using only the standard library. "
                    "It must read the JSON file in the current directory, compute "
                    "the answer, and print exactly one value. If the requested "
                    "key/record is absent, print exactly NOT_FOUND. Output only the script."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {task.instruction}\n\n"
                    f"JSON files:\n{preview}\n\n"
                    "Use the real file name in the script."
                ),
            },
        ]
        cost = 0
        attempts = 0
        try:
            attempts += 1
            result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=300)
            cost += result.cost_x10
            script = extract_script(result.content or "")
            if not script:
                return HandlerResult(
                    solved=False,
                    cost_x10=cost,
                    method="json_llm",
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
                last_line = (read_answer_file(task, consume=True) or "").strip().splitlines()
                last_line = last_line[-1] if last_line else ""
            if last_line.strip().casefold() in {"not_found", "not found", "none", "null"}:
                last_line = ""
            if last_line:
                return HandlerResult(
                    answer=last_line.strip(),
                    solved=True,
                    cost_x10=cost,
                    method="json_llm_script",
                    confidence=0.7,
                    attempts=attempts,
                    detail=(proc.stderr or "")[-500:],
                )
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="json_llm",
                detail=f"script produced no answer (rc={proc.returncode})",
                attempts=attempts,
            )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="json_llm",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
