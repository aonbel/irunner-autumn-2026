"""Fallback handler for unknown task shapes.

A compact ReAct loop over openJiuwen's ``Model.invoke`` with four tools:
``bash``, ``read_file``, ``write_file`` and ``finish``.  It is deliberately
small (at most four model calls) because the final competition set is limited
to the four known task families; this is only insurance against a wording
variation that the deterministic handlers do not recognise.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..budget import BudgetExhausted
from .base import HandlerContext, HandlerResult


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the current working directory. Use it to inspect and modify files.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Command to run"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file relative to the current directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file relative to the current directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return the final answer and stop.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def _safe_path(work_dir: Path, relative: str) -> Optional[Path]:
    try:
        candidate = (work_dir / relative).resolve()
        candidate.relative_to(work_dir.resolve())
        return candidate
    except (ValueError, OSError):
        return None


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated] ..."


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except ValueError:
        pass
    try:  # openJiuwen ships json-repair as a dependency
        from json_repair import repair_json

        value = repair_json(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _run_tool(work_dir: Path, name: str, args: dict) -> tuple[str, Optional[str]]:
    """Return ``(tool_output, final_answer)``."""
    if name == "finish":
        answer = str(args.get("answer", "")).strip()
        return answer, answer or None
    if name == "bash":
        cmd = str(args.get("cmd", "")).strip()
        if not cmd:
            return "empty command", None
        try:
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=20,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return _truncate(output or f"(no output, rc={proc.returncode})"), None
        except subprocess.TimeoutExpired:
            return "command timed out after 20s", None
        except OSError as exc:
            return f"command failed: {exc}", None
    if name == "read_file":
        path = _safe_path(work_dir, str(args.get("path", "")))
        if path is None or not path.is_file():
            return "file not found or path outside the working directory", None
        try:
            return _truncate(path.read_text(encoding="utf-8", errors="replace")), None
        except OSError as exc:
            return f"read failed: {exc}", None
    if name == "write_file":
        path = _safe_path(work_dir, str(args.get("path", "")))
        if path is None:
            return "path outside the working directory", None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {path.name}", None
        except OSError as exc:
            return f"write failed: {exc}", None
    return f"unknown tool: {name}", None


_TEXT_TOOL_RE = re.compile(r"\{.*\}", re.S)


def _parse_text_tool(content: str) -> Optional[dict]:
    """Fallback for providers without native tool calling."""
    match = _TEXT_TOOL_RE.search(content or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    if "final" in value:
        return {"name": "finish", "arguments": {"answer": str(value["final"])}}
    if "tool" in value:
        return {"name": str(value["tool"]), "arguments": value.get("args") or value.get("arguments") or {}}
    return None


class GenericHandler:
    name = "generic"

    def try_free(self, task):
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        # The final set is limited to four known task families.  An unknown
        # type would require an unverifiable free-form agent answer, so by
        # default we skip it instead of risking a hallucinated answer.
        # Set IRUNNER_ENABLE_GENERIC=1 for experiments.
        if os.environ.get("IRUNNER_ENABLE_GENERIC", "0") != "1":
            return HandlerResult(
                solved=False,
                method="generic",
                detail="generic fallback disabled for unknown task types",
            )
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="generic", detail="LLM disabled")

        work_dir = task.work_dir
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a careful task-solving agent in a sandbox. Work only "
                    "inside the current directory. Use the tools to inspect files, "
                    "run commands, and write results. When you have the final answer, "
                    "call finish with it. Do not ask questions. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task.instruction}\n\n"
                    f"Working directory: {work_dir}\n"
                    f"Files: {', '.join(task.listed_files(limit=60))}"
                ),
            },
        ]

        cost = 0
        attempts = 0
        for step in range(4):
            model_key = "fast" if step < 2 else "fast_reason"
            try:
                attempts += 1
                result = await ctx.llm.call(
                    task,
                    messages,
                    model_key=model_key,
                    max_tokens=350,
                    tools=TOOLS,
                )
                cost += result.cost_x10
            except BudgetExhausted:
                raise
            except Exception as exc:
                return HandlerResult(
                    solved=False,
                    cost_x10=cost,
                    method="generic_react",
                    detail=f"{type(exc).__name__}: {exc}",
                    attempts=attempts,
                )

            tool_calls = result.tool_calls
            if not tool_calls:
                parsed = _parse_text_tool(result.content)
                if parsed is not None:
                    tool_calls = [
                        {
                            "id": f"text-{step}",
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"]),
                        }
                    ]
                elif (result.content or "").strip():
                    answer = result.content.strip().splitlines()[0].strip()
                    return HandlerResult(
                        answer=answer,
                        solved=False,
                        cost_x10=cost,
                        method="generic_react_text",
                        confidence=0.35,
                        attempts=attempts,
                    )
                else:
                    break

            assistant_calls = []
            for call in tool_calls:
                assistant_calls.append(
                    {
                        "id": call.get("id") or f"call-{step}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": call.get("arguments", "{}"),
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": assistant_calls,
                }
            )

            for call in tool_calls:
                name = call.get("name", "")
                args = _parse_arguments(call.get("arguments", "{}"))
                output, final = _run_tool(work_dir, name, args)
                if final is not None:
                    return HandlerResult(
                        answer=final,
                        solved=True,
                        cost_x10=cost,
                        method="generic_react_finish",
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

        # If the agent wrote answer.txt itself, respect it.
        if task.answer_path.is_file():
            try:
                answer = task.answer_path.read_text(encoding="utf-8").strip()
            except OSError:
                answer = ""
            if answer:
                return HandlerResult(
                    answer=answer,
                    solved=True,
                    cost_x10=cost,
                    method="generic_react_answer_file",
                    confidence=0.6,
                    attempts=attempts,
                )

        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="generic_react",
            detail="no final answer",
            attempts=attempts,
        )
