"""Excel/spreadsheet tasks (hard LLM-only type)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file
from .data import extract_script


def _xlsx_files(task) -> list[Path]:
    return [
        path
        for path in sorted(task.work_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xlsm"}
    ][:4]


def _preview(path: Path, limit: int = 12) -> str:
    try:
        import openpyxl

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        chunks = [f"--- {path.name} ---"]
        for sheet in workbook.sheetnames[:5]:
            ws = workbook[sheet]
            chunks.append(f"sheet {sheet}:")
            for index, row in enumerate(ws.iter_rows(values_only=True)):
                if index >= limit:
                    break
                chunks.append(repr(row))
        workbook.close()
        return "\n".join(chunks)
    except Exception as exc:
        return f"--- {path.name} --- (cannot read: {exc})"


class SpreadsheetHandler:
    name = "spreadsheet"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="spreadsheet", detail="LLM disabled")
        files = _xlsx_files(task)
        if not files:
            return HandlerResult(impossible=True, method="spreadsheet", detail="no spreadsheet files")

        preview = "\n\n".join(_preview(path) for path in files)
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a short Python script using openpyxl and the standard "
                    "library. It must read the spreadsheet in the current directory, "
                    "compute the answer, and print exactly one value. If the "
                    "requested data is absent, print exactly NOT_FOUND. Output only "
                    "the script."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {task.instruction}\n\nSpreadsheet preview:\n{preview}",
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
                return HandlerResult(solved=False, cost_x10=cost, method="spreadsheet", detail="no script", attempts=attempts)
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
                    method="spreadsheet_script",
                    confidence=0.65,
                    attempts=attempts,
                    detail=(proc.stderr or "")[-500:],
                )
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="spreadsheet",
                detail=f"script produced no answer (rc={proc.returncode})",
                attempts=attempts,
            )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="spreadsheet",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
