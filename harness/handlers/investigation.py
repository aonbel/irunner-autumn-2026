"""Investigation / log-analysis tasks (hard LLM-only type).

The model gets a small tool loop (bash/read/finish) and must inspect large
logs or a tree of files.  This is the type that stresses context growth: each
step re-sends the previous tool output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult
from .generic import TOOLS, _parse_arguments, _parse_text_tool, _run_tool


def _files(task) -> list[Path]:
    return [path for path in sorted(task.work_dir.rglob("*")) if path.is_file()][:80]


class InvestigationHandler:
    name = "investigation"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="investigation", detail="LLM disabled")
        files = _files(task)
        if not files:
            return HandlerResult(impossible=True, method="investigation", detail="empty workspace")

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a forensic task-solving agent in a sandbox. Work only "
                    "inside the current directory. Use bash/read tools to inspect "
                    "files, search logs and compute. When you have the final answer, "
                    "call finish. If it is not present, call finish with NOT_FOUND."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task.instruction}\n\n"
                    f"Working directory: {task.work_dir}\n"
                    f"Files: {', '.join(str(p.relative_to(task.work_dir)) for p in files[:60])}"
                ),
            },
        ]
        cost = 0
        attempts = 0
        for step in range(5):
            model_key = "fast" if step < 2 else "smart"
            try:
                attempts += 1
                result = await ctx.llm.call(
                    task,
                    messages,
                    model_key=model_key,
                    max_tokens=400,
                    tools=TOOLS,
                )
                cost += result.cost_x10
            except BudgetExhausted:
                raise
            except Exception as exc:
                return HandlerResult(
                    solved=False,
                    cost_x10=cost,
                    method="investigation",
                    detail=f"{type(exc).__name__}: {exc}",
                    attempts=attempts,
                )

            tool_calls = result.tool_calls
            if not tool_calls:
                parsed = _parse_text_tool(result.content)
                if parsed is not None:
                    tool_calls = [{"id": f"text-{step}", "name": parsed["name"], "arguments": json.dumps(parsed["arguments"])}]
                elif (result.content or "").strip():
                    answer = result.content.strip().splitlines()[0].strip()
                    if answer.casefold() not in {"not_found", "not found", "none"}:
                        return HandlerResult(
                            answer=answer,
                            solved=True,
                            cost_x10=cost,
                            method="investigation_text",
                            confidence=0.5,
                            attempts=attempts,
                        )
                    break
                else:
                    break

            assistant_calls = [
                {
                    "id": call.get("id") or f"call-{step}",
                    "type": "function",
                    "function": {"name": call.get("name", ""), "arguments": call.get("arguments", "{}")},
                }
                for call in tool_calls
            ]
            messages.append({"role": "assistant", "content": result.content or "", "tool_calls": assistant_calls})
            for call in tool_calls:
                name = call.get("name", "")
                args = _parse_arguments(call.get("arguments", "{}"))
                output, final = _run_tool(task.work_dir, name, args)
                if final is not None:
                    if final.strip().casefold() in {"not_found", "not found", "none", ""}:
                        return HandlerResult(
                            solved=False,
                            cost_x10=cost,
                            method="investigation",
                            detail="model returned NOT_FOUND",
                            attempts=attempts,
                        )
                    return HandlerResult(
                        answer=final,
                        solved=True,
                        cost_x10=cost,
                        method="investigation_finish",
                        confidence=0.55,
                        attempts=attempts,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or f"call-{step}",
                        "name": name,
                        "content": output,
                    }
                )

        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="investigation",
            detail="no final answer",
            attempts=attempts,
        )
