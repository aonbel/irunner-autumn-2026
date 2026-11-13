"""Hard logic/constraint puzzles (LLM-only type)."""

from __future__ import annotations

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult, read_answer_file


class LogicHandler:
    name = "logic"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="logic", detail="LLM disabled")
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the logic/constraint puzzle carefully. Check every "
                    "constraint. Return only the final answer, with no explanation."
                ),
            },
            {"role": "user", "content": task.instruction},
        ]
        cost = 0
        attempts = 0
        try:
            attempts += 1
            result = await ctx.llm.call(task, messages, model_key="smart", max_tokens=800)
            cost += result.cost_x10
            answer = (result.content or "").strip().splitlines()[0] if result.content else ""
            if not answer:
                answer = (read_answer_file(task, consume=True) or "").strip()
            if answer and answer.casefold() not in {"not_found", "not found", "none", "unknown"}:
                return HandlerResult(
                    answer=answer.strip(),
                    solved=True,
                    cost_x10=cost,
                    method="logic_llm",
                    confidence=0.5,
                    attempts=attempts,
                )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="logic",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="logic",
            detail="empty answer",
            attempts=attempts,
        )
