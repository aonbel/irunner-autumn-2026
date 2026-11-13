"""OpenAI-compatible LLM access through openJiuwen.

All model calls go through :class:`openjiuwen.core.foundation.llm.model.Model`.
We intentionally do not use openJiuwen's high-level DeepAgent for the main
loop: every ReAct step re-sends the whole conversation, which is expensive at
a 100K-token budget.  DeepAgent remains available as an optional fallback.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .budget import (
    BudgetExhausted,
    TaskBudget,
    TokenMeter,
    TokenUsage,
    estimate_messages_tokens,
)
from .config import Config
from .tracing import Tracer


@dataclass
class LLMResult:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_x10: int = 0
    model: str = ""
    latency: float = 0.0
    raw: Any = None

    @property
    def cost(self) -> float:
        return self.cost_x10 / 10.0


def quiet_openjiuwen() -> None:
    """Silence the SDK's very chatty loguru logger.

    The CLI does something similar internally.  We keep the patch minimal and
    defensive so a future openJiuwen version cannot break the harness.
    """
    try:  # pragma: no cover - depends on the runtime dependency
        from openjiuwen.core.common.logging import manager as log_manager

        manager_cls = log_manager.LogManager
        if getattr(manager_cls, "_initialized", False):
            return

        class _NullLogger:
            def __getattr__(self, _name):
                def _noop(*_args, **_kwargs):
                    return None

                return _noop

        manager_cls._initialized = True  # type: ignore[attr-defined]
        manager_cls.get_logger = classmethod(  # type: ignore[assignment]
            lambda cls, log_type="default": _NullLogger()
        )
    except Exception:
        pass


_BUDGET_HINTS = (
    "quota",
    "insufficient_quota",
    "budget",
    "too many tokens",
    "token limit",
    "payment required",
    "402",
)


def _looks_like_budget_error(message: str) -> bool:
    lowered = (message or "").casefold()
    return any(hint in lowered for hint in _BUDGET_HINTS)


def _is_transient_failure(message: str) -> bool:
    lowered = (message or "").casefold()
    return any(
        hint in lowered
        for hint in (
            "timeout",
            "timed out",
            "connection",
            "connect",
            "429",
            "502",
            "503",
            "504",
            "temporarily unavailable",
        )
    )


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for call in raw_calls or []:
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("arguments")
            if args is None:
                args = call.get("function", {}).get("arguments", "")
            call_id = call.get("id")
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "arguments", "")
            call_id = getattr(call, "id", None)
        if not name:
            continue
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        result.append({"id": call_id, "name": str(name), "arguments": args})
    return result


class LLMRouter:
    def __init__(self, config: Config, meter: TokenMeter):
        self.config = config
        self.meter = meter
        self.tracer = Tracer(config.log_dir / "traces.jsonl")
        self._models: dict[str, Any] = {}
        self._task_budgets: dict[str, TaskBudget] = {}
        self._import_error: Optional[str] = None
        self._base_url = (config.llm_base_url or "").strip().rstrip("/")
        self._v1_fallback_tried = False
        self._model_names = dict(config.models)
        self._consecutive_failures = 0
        self._failure_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(
            not self.config.no_llm
            and self._base_url
            and self.config.llm_api_key
        )

    def register_task_budget(self, task_id: str, budget: TaskBudget) -> None:
        self._task_budgets[task_id] = budget

    def unregister_task_budget(self, task_id: str) -> None:
        self._task_budgets.pop(task_id, None)

    def _model_name(self, key: str) -> str:
        if key in self._model_names:
            return self._model_names[key]
        return key

    def _make_model(self, model_name: str):
        if model_name in self._models:
            return self._models[model_name]
        if self._import_error is not None:
            raise RuntimeError(self._import_error)
        quiet_openjiuwen()
        try:
            from openjiuwen.core.foundation.llm.model import Model
            from openjiuwen.core.foundation.llm.schema.config import (
                ModelClientConfig,
                ModelRequestConfig,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._import_error = f"openjiuwen import failed: {exc}"
            raise RuntimeError(self._import_error) from exc

        client_config = ModelClientConfig(
            client_provider="OpenAI",
            api_key=self.config.llm_api_key,
            api_base=self._base_url,
            timeout=self.config.model_timeout,
            max_retries=0,
            verify_ssl=False,
        )
        request_config = ModelRequestConfig(
            model=model_name,
            temperature=0.0,
            top_p=0.1,
            max_tokens=512,
        )
        model = Model(client_config, request_config)
        self._models[model_name] = model
        return model

    async def call(
        self,
        task,
        messages: Any,
        *,
        model_key: str = "fast",
        max_tokens: int = 400,
        temperature: float = 0.0,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResult:
        if not self.available:
            raise RuntimeError("LLM is not configured (LLM_BASE_URL / LLM_API_KEY)")

        model_name = self._model_name(model_key)
        task_budget = self._task_budgets.get(getattr(task, "task_id", ""))
        input_tokens = estimate_messages_tokens(messages)
        if tools:
            input_tokens += estimate_messages_tokens(tools)
        estimate_x10 = self.meter.estimate_call_x10(input_tokens, max_tokens)

        task_id = getattr(task, "task_id", "")
        if not self.meter.can_afford_call(estimate_x10, task_budget, task_id=task_id):
            raise BudgetExhausted(
                f"budget exhausted before call: estimate={estimate_x10 / 10.0:.0f}, "
                f"remaining={self.meter.remaining:.0f}"
            )

        model = self._make_model(model_name)
        attempts = max(1, int(self.config.max_retries) + 1)
        last_error: Optional[Exception] = None
        started = time.monotonic()

        for attempt in range(attempts):
            try:
                response = await model.invoke(
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.1,
                )
                latency = time.monotonic() - started
                usage = self._usage_from_response(response, messages, max_tokens, task)
                cost_x10 = self.meter.record(getattr(task, "task_id", ""), usage)
                if task_budget is not None:
                    task_budget.charge(cost_x10)
                result = LLMResult(
                    content=str(getattr(response, "content", "") or ""),
                    tool_calls=_normalize_tool_calls(getattr(response, "tool_calls", None)),
                    usage=usage,
                    cost_x10=cost_x10,
                    model=model_name,
                    latency=latency,
                    raw=response,
                )
                with self._failure_lock:
                    self._consecutive_failures = 0
                self.tracer.log(
                    {
                        "event": "llm_call",
                        "task_id": getattr(task, "task_id", ""),
                        "task_type": getattr(task, "task_type", ""),
                        "model": model_name,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "reasoning_tokens": usage.reasoning_tokens,
                        "cost": cost_x10 / 10.0,
                        "latency": round(latency, 3),
                        "prompt_chars": len(str(messages)),
                        "response_chars": len(result.content),
                        "tool_calls": len(result.tool_calls),
                        "estimated": usage.estimated,
                    }
                )
                return result
            except Exception as exc:
                last_error = exc
                message = f"{type(exc).__name__}: {exc}"
                if _looks_like_budget_error(message):
                    self.tracer.log(
                        {
                            "event": "budget_exhausted",
                            "task_id": getattr(task, "task_id", ""),
                            "model": model_name,
                            "error": message,
                        }
                    )
                    raise BudgetExhausted(message) from exc
                # Some proxies expect the OpenAI base URL without /v1 and
                # some with it.  If the first attempt looks like a 404 and we
                # have not tried the /v1 suffix yet, retry once with it.
                if (
                    attempt == 0
                    and not self._v1_fallback_tried
                    and "404" in message
                    and not self._base_url.endswith("/v1")
                ):
                    self._v1_fallback_tried = True
                    self._base_url = self._base_url + "/v1"
                    self._models.clear()
                    continue
                if attempt + 1 >= attempts:
                    break
                delay = min(8.0, 1.5 * (2**attempt) + random.random() * 0.5)
                await asyncio.sleep(delay)

        message = f"{type(last_error).__name__}: {last_error}"
        self.tracer.log(
            {
                "event": "llm_error",
                "task_id": getattr(task, "task_id", ""),
                "model": model_name,
                "error": message,
            }
        )
        with self._failure_lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
        if failures >= 3 and _is_transient_failure(message):
            # The competition proxy simply stops answering once the budget is
            # gone.  Three consecutive transport failures across workers are a
            # strong signal to stop spending time on more calls.
            raise BudgetExhausted(
                f"proxy unavailable after {failures} consecutive failures: {message}"
            ) from last_error
        raise RuntimeError(message) from last_error

    def _usage_from_response(self, response: Any, messages: Any, max_tokens: int, task) -> TokenUsage:
        usage_obj = getattr(response, "usage_metadata", None)
        if usage_obj is not None:
            return TokenUsage(
                input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
                reasoning_tokens=int(getattr(usage_obj, "reasoning_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
                estimated=False,
            )

        # Fallback: estimate from the prompt and response text.
        from .budget import estimate_text_tokens

        input_tokens = estimate_messages_tokens(messages)
        content = str(getattr(response, "content", "") or "")
        reasoning = str(getattr(response, "reasoning_content", "") or "")
        output_tokens = estimate_text_tokens(content) + estimate_text_tokens(reasoning)
        output_tokens = min(output_tokens, max(1, max_tokens))
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=0,
            total_tokens=input_tokens + output_tokens,
            estimated=True,
        )
