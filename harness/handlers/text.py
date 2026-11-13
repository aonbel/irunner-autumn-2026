"""Free-text extraction tasks (new LLM-only type).

The workspace contains one or more ``.txt``/``.md`` files with facts and the
question asks for one of them.  There is no reliable deterministic parser for
arbitrary prose, so this type is solved by the LLM and the answer is accepted
only if it is grounded in a source line together with a question keyword.
"""

from __future__ import annotations

from pathlib import Path

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file
from .search import _line_grounded


def _text_files(task) -> list[Path]:
    files: list[Path] = []
    for path in sorted(task.work_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".text"}:
            continue
        if "answer.txt" == path.name.lower():
            continue
        files.append(path)
    return files[:3]


def _snippets(files: list[Path], limit: int = 6000) -> str:
    chunks: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(f"--- {path.name} ---\n{text[:limit]}")
    return "\n\n".join(chunks)


class TextHandler:
    name = "text"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="text_llm", detail="LLM disabled")

        files = _text_files(task)
        if not files:
            return HandlerResult(impossible=True, method="text_llm", detail="no text files")
        snippets = _snippets(files)
        if not snippets.strip():
            return HandlerResult(impossible=True, method="text_llm", detail="empty text files")

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied text. Return only "
                    "the exact answer, without a sentence or explanation. If the "
                    "answer is not explicitly present, return exactly NOT_FOUND."
                ),
            },
            {"role": "user", "content": f"Question: {task.instruction}\n\nText:\n{snippets}"},
        ]
        cost = 0
        attempts = 0
        answer = ""
        try:
            attempts += 1
            result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=60)
            cost += result.cost_x10
            answer = (result.content or "").strip().splitlines()[0] if result.content else ""
            candidates = [answer, read_answer_file(task, consume=True) or ""]
            for candidate in candidates:
                candidate = candidate.strip().strip("'\"")
                if not candidate or candidate.casefold() in {"not_found", "not found", "none", "unknown"}:
                    continue
                if _line_grounded(candidate, snippets, task.instruction):
                    return HandlerResult(
                        answer=candidate,
                        solved=True,
                        cost_x10=cost,
                        method="text_llm",
                        confidence=0.75,
                        attempts=attempts,
                    )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="text_llm",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="text_llm",
            detail=f"ungrounded answer: {answer[:80]}" if answer else "empty answer",
            attempts=attempts,
        )
