"""PDF/DOCX document extraction tasks (hard LLM-only type)."""

from __future__ import annotations

from pathlib import Path

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file
from .search import _line_grounded


def _docs(task) -> list[Path]:
    suffixes = {".pdf", ".docx"}
    return [path for path in sorted(task.work_dir.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes][:3]


def _extract(path: Path, limit: int = 20_000) -> str:
    text = ""
    try:
        if path.suffix.lower() == ".pdf":
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                parts = []
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
                    if sum(len(part) for part in parts) > limit:
                        break
                text = "\n".join(parts)
        elif path.suffix.lower() == ".docx":
            import docx

            doc = docx.Document(str(path))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as exc:
        return f"(cannot read {path.name}: {exc})"
    return text[:limit]


class DocumentHandler:
    name = "document"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="document", detail="LLM disabled")
        files = _docs(task)
        if not files:
            return HandlerResult(impossible=True, method="document", detail="no PDF/DOCX files")
        snippets = "\n\n".join(f"--- {path.name} ---\n{_extract(path)}" for path in files)
        if not snippets.strip():
            return HandlerResult(impossible=True, method="document", detail="empty documents")

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied document text. "
                    "Return only the exact answer. If it is not explicitly present, "
                    "return exactly NOT_FOUND."
                ),
            },
            {"role": "user", "content": f"Question: {task.instruction}\n\nDocuments:\n{snippets}"},
        ]
        cost = 0
        attempts = 0
        answer = ""
        try:
            attempts += 1
            result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=80)
            cost += result.cost_x10
            answer = (result.content or "").strip().splitlines()[0] if result.content else ""
            for candidate in [answer, read_answer_file(task, consume=True) or ""]:
                candidate = candidate.strip().strip("'\"")
                if not candidate or candidate.casefold() in {"not_found", "not found", "none"}:
                    continue
                if _line_grounded(candidate, snippets, task.instruction):
                    return HandlerResult(
                        answer=candidate,
                        solved=True,
                        cost_x10=cost,
                        method="document_llm",
                        confidence=0.7,
                        attempts=attempts,
                    )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="document",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="document",
            detail=f"ungrounded answer: {answer[:80]}" if answer else "empty answer",
            attempts=attempts,
        )
