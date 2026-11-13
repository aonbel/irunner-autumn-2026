"""CSV / data-analysis tasks."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, instruction_core, read_answer_file


@dataclass
class DataSolve:
    ok: bool
    value: Optional[float] = None
    text: Optional[str] = None
    impossible: bool = False
    reason: str = ""


def find_csv(task) -> Optional[Path]:
    # If the instruction names a CSV explicitly, use exactly that file.
    named = re.search(r"([A-Za-z0-9_.-]+\.csv)\b", task.instruction or "", re.I)
    if named:
        candidate = task.work_dir / named.group(1)
        if candidate.is_file():
            return candidate
        return None
    preferred = task.work_dir / "sales.csv"
    if preferred.is_file():
        return preferred
    candidates = [path for path in sorted(task.work_dir.rglob("*.csv")) if path.is_file()]
    if len(candidates) == 1:
        return candidates[0]
    return None


def read_rows(path: Path, limit: int = 100_000) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict] = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append({str(k).strip().lower(): v for k, v in row.items() if k is not None})
        fieldnames = [str(name).strip().lower() for name in (reader.fieldnames or [])]
    return rows, fieldnames


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _field(row: dict, name: str) -> str:
    return str(row.get(name, "") or "").strip()


def _to_number(value: object) -> Optional[float]:
    try:
        text = str(value).strip().replace(",", "")
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _parse_month_range(text: str) -> Optional[tuple[int, int]]:
    lower = text.lower()
    match = re.search(
        r"months?\s+(?:\w+\s+)?(\d{1,2})\s*(?:through|thru|to|-|–|and|,)\s*(?:\w+\s+)?(\d{1,2})",
        lower,
    )
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if 1 <= start <= 12 and 1 <= end <= 12:
            return start, end
    match = re.search(
        r"(?:in|during|between|from)\s+(?:months?\s+)?([a-z]+)\s*(?:through|to|-|and)\s*([a-z]+)",
        lower,
    )
    if match:
        start = _MONTH_NAMES.get(match.group(1))
        end = _MONTH_NAMES.get(match.group(2))
        if start and end:
            return start, end
    match = re.search(r"months?\s+(\d{1,2})\s+and\s+(\d{1,2})", lower)
    if match:
        return int(match.group(1)), int(match.group(2))
    # A single month, e.g. "in month 5".
    match = re.search(r"\bmonth\s+(\d{1,2})\b", lower)
    if match and not re.search(r"\bmonth\s+\d{1,2}\s*(?:through|to|and|-)\b", lower):
        month = int(match.group(1))
        if 1 <= month <= 12:
            return month, month
    return None


def _parse_region(text: str) -> Optional[str]:
    match = re.search(r"\bregion\s+['\"]?([A-Za-z][A-Za-z -]*?)['\"]?(?:\s|,|\.|$)", text, re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(west|east|north|south)\s+region\b", text, re.I)
    if match:
        return match.group(1)
    match = re.search(r"\bin\s+the\s+(west|east|north|south)\b", text, re.I)
    if match:
        return match.group(1)
    return None


def _parse_product(text: str) -> Optional[str]:
    match = re.search(r"product\s+['\"]([^'\"]+)['\"]", text, re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"product\s+([A-Za-z][A-Za-z0-9_ -]*?)\s+(?:were|was|in|sold|sales)", text, re.I)
    if match:
        return match.group(1).strip()
    return None


_STRICT_UNITS_RE = re.compile(
    r"how\s+many\s+units\s+of\s+product\s+['\"]?(?P<product>[A-Za-z0-9_ -]+?)['\"]?"
    r"\s+were\s+sold\s+in\s+months?\s+(?P<start>\d{1,2})\s*"
    r"(?:through|to|and|-)\s*(?P<end>\d{1,2})(?:\s+inclusive)?",
    re.I,
)


def solve_deterministic(task) -> DataSolve:
    csv_path = find_csv(task)
    if csv_path is None:
        return DataSolve(False, impossible=True, reason="no CSV file")
    try:
        rows, fields = read_rows(csv_path)
    except OSError as exc:
        return DataSolve(False, impossible=True, reason=f"cannot read CSV: {exc}")

    if not rows:
        return DataSolve(False, impossible=True, reason="CSV has no data rows")

    # Ultra-safe: only the exact example query shape is answered without an
    # LLM.  Any extra condition, different aggregate or unrecognised phrasing
    # is delegated to the LLM script fallback.
    core = instruction_core(task.instruction)
    core = re.sub(r"\b(?:columns?|headers?)\b[^.]*\.", " ", core)
    match = _STRICT_UNITS_RE.search(core)
    if not match:
        return DataSolve(False, reason="query does not match the strict template")
    tail = core[match.end():].strip(" .?")
    if tail:
        return DataSolve(False, reason="extra text after the strict template")

    product = match.group("product").strip()
    start = int(match.group("start"))
    end = int(match.group("end"))
    if not (1 <= start <= 12 and 1 <= end <= 12):
        return DataSolve(False, reason="invalid month range")
    if start > end:
        start, end = end, start
    if "units" not in fields or "product" not in fields or "month" not in fields:
        return DataSolve(False, reason="CSV schema does not match the strict template")

    selected = []
    for row in rows:
        if _field(row, "product").lower() != product.lower():
            continue
        month = _to_number(_field(row, "month"))
        if month is None:
            continue
        if start <= int(month) <= end:
            selected.append(row)

    total = sum((_to_number(_field(row, "units")) or 0.0) for row in selected)
    return DataSolve(True, value=total)


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_script(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    if matches:
        return matches[0].strip()
    if "import " in text or "print(" in text:
        return text.strip()
    return None


class DataHandler:
    name = "data_analysis"

    def try_free(self, task) -> Optional[HandlerResult]:
        result = solve_deterministic(task)
        if result.impossible:
            return HandlerResult(
                impossible=True,
                method="data_template",
                detail=result.reason,
                confidence=0.95,
            )
        if result.ok and (result.text is not None or result.value is not None):
            if result.text is not None:
                answer = result.text
            else:
                answer = format_number(result.value) if not isinstance(result.value, int) else str(result.value)
            return HandlerResult(
                answer=answer,
                solved=True,
                method="data_template",
                confidence=0.95,
                detail="deterministic CSV query",
            )
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        result = self.try_free(task)
        if result is not None:
            return result
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="data_template", detail="no deterministic parse and LLM disabled")

        csv_path = find_csv(task)
        if csv_path is None:
            return HandlerResult(impossible=True, method="data_template", detail="no CSV file")
        try:
            rows, fields = read_rows(csv_path)
        except OSError as exc:
            return HandlerResult(impossible=True, method="data_template", detail=str(exc))
        header = ", ".join(fields)
        sample = "\n".join(
            ", ".join(str(row.get(field, "")) for field in fields) for row in rows[:3]
        )
        want = "number" if task.check == "exact_int" else "answer value"
        # Tell the model what to do when the requested data is absent.
        absent_hint = "If the requested column/value is absent, print exactly NOT_FOUND."
        prompt = (
            "Write a short Python script using only the standard library that "
            f"reads the CSV file in the current directory and prints exactly one "
            f"{want} answering the question. Do not print explanations. "
            f"{absent_hint} Output only the script.\n\n"
            f"Question: {task.instruction}\n"
            f"CSV file: {csv_path.name}\n"
            f"Columns: {header}\n"
            f"First rows:\n{sample}\n"
        )
        messages = [
            {
                "role": "system",
                "content": "You write tiny, correct Python data-analysis scripts. Output only code.",
            },
            {"role": "user", "content": prompt},
        ]
        cost = 0
        attempts = 0
        try:
            attempts += 1
            llm_result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=300)
            cost += llm_result.cost_x10
            script = extract_script(llm_result.content)
            if not script:
                return HandlerResult(
                    solved=False,
                    cost_x10=cost,
                    method="data_llm",
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
            match = _NUMBER_RE.search(last_line)
            text_answer = last_line
            if not match:
                # The model may have followed the instruction literally and
                # written answer.txt itself instead of printing the value.
                file_answer = read_answer_file(task, consume=True)
                if file_answer:
                    text_answer = file_answer
                    match = _NUMBER_RE.search(file_answer)
            if match:
                value = float(match.group(0))
                return HandlerResult(
                    answer=format_number(value),
                    solved=True,
                    cost_x10=cost,
                    method="data_llm_script",
                    confidence=0.7,
                    attempts=attempts,
                    detail=(proc.stderr or "")[-500:],
                )
            if text_answer.strip().casefold() in {"not_found", "not found", "none", "null"}:
                text_answer = ""
            if task.check != "exact_int" and text_answer:
                return HandlerResult(
                    answer=text_answer.strip(),
                    solved=True,
                    cost_x10=cost,
                    method="data_llm_script",
                    confidence=0.6,
                    attempts=attempts,
                    detail=(proc.stderr or "")[-500:],
                )
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="data_llm",
                detail=f"script produced no number (rc={proc.returncode})",
                attempts=attempts,
            )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="data_llm",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
