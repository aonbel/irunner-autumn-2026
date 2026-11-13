"""Execution orchestration: free phase, portfolio, LLM phase, persistence."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

from .budget import BudgetExhausted, TaskBudget, TokenMeter
from .config import Config
from .handlers.base import HandlerContext, HandlerResult
from .handlers.coding import CodingHandler
from .handlers.data import DataHandler
from .handlers.generic import GenericHandler
from .handlers.math import MathHandler
from .handlers.search import SearchHandler
from .handlers.text import TextHandler
from .handlers.json_query import JsonHandler
from .handlers.sql_query import SqlHandler
from .handlers.investigation import InvestigationHandler
from .handlers.multi_csv import MultiCsvHandler
from .handlers.spreadsheet import SpreadsheetHandler
from .handlers.document import DocumentHandler
from .handlers.logic import LogicHandler
from .llm import LLMRouter
from .persist import SolutionStore
from .planner import PlanItem, build_portfolio, make_items, summarize
from .tasks import Task


LOG = logging.getLogger("irunner")


def handler_for(task_type: str):
    if task_type == "math":
        return MathHandler()
    if task_type in {"data_analysis", "data"}:
        return DataHandler()
    if task_type == "search":
        return SearchHandler()
    if task_type == "coding":
        return CodingHandler()
    if task_type in {"text", "text_extraction"}:
        return TextHandler()
    if task_type in {"json", "json_query"}:
        return JsonHandler()
    if task_type == "sql":
        return SqlHandler()
    if task_type in {"investigation", "log_analysis"}:
        return InvestigationHandler()
    if task_type in {"multi_csv", "data_join"}:
        return MultiCsvHandler()
    if task_type in {"spreadsheet", "xlsx"}:
        return SpreadsheetHandler()
    if task_type in {"document", "pdf", "docx"}:
        return DocumentHandler()
    if task_type in {"logic", "constraint"}:
        return LogicHandler()
    return GenericHandler()


def _task_timeout(task_type: str) -> float:
    if task_type == "coding":
        return 180.0
    if task_type in {"math", "data_analysis", "data", "search"}:
        return 60.0
    return 120.0


class Orchestrator:
    def __init__(
        self,
        config: Config,
        tasks: list[Task],
        meter: TokenMeter,
        llm: LLMRouter,
        store: SolutionStore,
        stop_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.tasks = tasks
        self.meter = meter
        self.llm = llm
        self.store = store
        self.stop_event = stop_event or threading.Event()
        self.started = time.monotonic()
        self.budget_exhausted = False
        self.stats: dict[str, dict[str, int]] = {}
        self.planner_summary: dict = {}

    # -- helpers ------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def deadline_reached(self) -> bool:
        return self.elapsed >= self.config.deadline_seconds

    def _should_stop(self) -> bool:
        return (
            self.stop_event.is_set()
            or self.budget_exhausted
            or self.deadline_reached
        )

    async def _run_local(self, task: Task, handler) -> Optional[HandlerResult]:
        try:
            return await asyncio.to_thread(handler.try_free, task)
        except Exception as exc:  # local handlers must never crash the run
            LOG.warning("free-phase error for %s: %s: %s", task.task_id, type(exc).__name__, exc)
            return None

    def _record_free_result(self, task: Task, result: Optional[HandlerResult]) -> None:
        if result is None:
            task._p_estimate = None  # type: ignore[attr-defined]
            task._cost_estimate_x10 = None  # type: ignore[attr-defined]
            return
        cost_x10 = int(result.cost_x10)
        if result.impossible:
            task._impossible = True  # type: ignore[attr-defined]
            self.store.record(
                task,
                status="impossible",
                answer=None,
                solved=False,
                cost_x10=cost_x10,
                method=result.method,
                detail=result.detail,
            )
            LOG.info("impossible: %s (%s)", task.task_id, result.detail)
            return
        if result.solved and result.answer:
            task._done = True  # type: ignore[attr-defined]
            self.store.record(
                task,
                status="solved",
                answer=result.answer,
                solved=True,
                cost_x10=cost_x10,
                method=result.method,
                detail=result.detail,
            )
            LOG.info("solved free: %s -> %s (%s)", task.task_id, result.answer, result.method)
            return
        # Not solved by the free phase; keep for the LLM portfolio.
        task._p_estimate = None  # type: ignore[attr-defined]
        task._cost_estimate_x10 = None  # type: ignore[attr-defined]

    async def free_phase(self) -> None:
        if not self.tasks:
            return
        cpu_workers = max(1, min(4, os.cpu_count() or 2))
        semaphore = asyncio.Semaphore(cpu_workers)
        handlers = {task.task_id: handler_for(task.task_type) for task in self.tasks}

        async def one(task: Task) -> None:
            async with semaphore:
                if self._should_stop():
                    return
                result = await self._run_local(task, handlers[task.task_id])
                self._record_free_result(task, result)

        await asyncio.gather(*(one(task) for task in self.tasks), return_exceptions=True)

    # -- LLM phase ----------------------------------------------------------
    def _update_type_stats(self, task: Task, solved: bool) -> None:
        bucket = self.stats.setdefault(task.task_type, {"attempts": 0, "successes": 0})
        bucket["attempts"] += 1
        if solved:
            bucket["successes"] += 1
        p = (bucket["successes"] + 1.0) / (bucket["attempts"] + 2.0)
        for other in self.tasks:
            if other.task_type == task.task_type and not getattr(other, "_done", False) and not getattr(other, "_impossible", False):
                other._p_estimate = p  # type: ignore[attr-defined]

    async def _run_llm_task(self, item: PlanItem, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            task = item.task
            if self._should_stop() or getattr(task, "_done", False) or getattr(task, "_impossible", False):
                return
            if self.meter.remaining_x10 < 100:
                return

            cap_units = self.config.task_cap(task.task_type, task.reward)
            cap_x10 = min(int(cap_units * 10), self.meter.remaining_x10)
            if cap_x10 <= 100:
                return
            if not self.meter.reserve_task(task.task_id, cap_x10):
                cap_x10 = min(cap_x10, self.meter.remaining_x10)
                if cap_x10 <= 100 or not self.meter.reserve_task(task.task_id, cap_x10):
                    return

            task_budget = TaskBudget(cap_x10)
            self.llm.register_task_budget(task.task_id, task_budget)
            handler = handler_for(task.task_type)
            ctx = HandlerContext(config=self.config, llm=self.llm)
            result: Optional[HandlerResult] = None
            try:
                result = await asyncio.wait_for(
                    handler.run(task, ctx),
                    timeout=_task_timeout(task.task_type),
                )
            except asyncio.TimeoutError:
                result = HandlerResult(
                    solved=False,
                    method="timeout",
                    detail=f"task timed out after {_task_timeout(task.task_type):.0f}s",
                )
            except BudgetExhausted as exc:
                self.budget_exhausted = True
                result = HandlerResult(
                    solved=False,
                    cost_x10=self.meter.task_spent_x10(task.task_id),
                    method="budget_exhausted",
                    detail=str(exc),
                )
                LOG.warning("budget exhausted while running %s: %s", task.task_id, exc)
            except Exception as exc:
                result = HandlerResult(
                    solved=False,
                    cost_x10=self.meter.task_spent_x10(task.task_id),
                    method="handler_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                LOG.warning("handler error for %s: %s: %s", task.task_id, type(exc).__name__, exc)
            finally:
                self.meter.release_task(task.task_id)
                self.llm.unregister_task_budget(task.task_id)

            if result is None:
                result = HandlerResult(solved=False, method="no_result")

            cost_x10 = max(int(result.cost_x10), self.meter.task_spent_x10(task.task_id))
            if result.impossible:
                task._impossible = True  # type: ignore[attr-defined]
                self.store.record(
                    task,
                    status="impossible",
                    answer=None,
                    solved=False,
                    cost_x10=cost_x10,
                    method=result.method,
                    detail=result.detail,
                )
                LOG.info("impossible (LLM): %s (%s)", task.task_id, result.detail)
            elif result.solved and result.answer:
                task._done = True  # type: ignore[attr-defined]
                self.store.record(
                    task,
                    status="solved",
                    answer=result.answer,
                    solved=True,
                    cost_x10=cost_x10,
                    method=result.method,
                    detail=result.detail,
                )
                LOG.info("solved: %s -> %s (%s, %.0f cost)", task.task_id, result.answer, result.method, cost_x10 / 10.0)
            elif result.answer and result.confidence >= 0.45:
                self.store.record(
                    task,
                    status="attempted",
                    answer=result.answer,
                    solved=False,
                    cost_x10=cost_x10,
                    method=result.method,
                    detail=result.detail,
                )
                LOG.info("attempted: %s -> %s (%s)", task.task_id, result.answer, result.method)
            else:
                self.store.record(
                    task,
                    status="failed",
                    answer=None,
                    solved=False,
                    cost_x10=cost_x10,
                    method=result.method,
                    detail=result.detail,
                )
                LOG.info("failed: %s (%s)", task.task_id, result.detail)
            self._update_type_stats(task, bool(result.solved))

    async def llm_phase(self) -> None:
        pending = [
            task
            for task in self.tasks
            if not getattr(task, "_done", False) and not getattr(task, "_impossible", False)
        ]
        if not pending:
            return

        items = make_items(pending, self.config)
        qualifier = len(self.tasks) <= 5
        selected, lam = build_portfolio(
            items,
            self.meter.remaining_x10,
            qualifier_mode=qualifier,
        )
        self.planner_summary = summarize(items, lam, self.meter.remaining_x10)
        self.planner_summary["qualifier"] = qualifier
        LOG.info(
            "portfolio: candidates=%s selected=%s expected_reward=%.2f planned_cost=%.0f lambda=%.6f",
            self.planner_summary.get("candidates"),
            self.planner_summary.get("selected"),
            self.planner_summary.get("expected_reward", 0.0),
            self.planner_summary.get("planned_cost", 0.0),
            lam,
        )
        if not selected:
            return

        semaphore = asyncio.Semaphore(max(1, min(5, self.config.workers)))
        await asyncio.gather(
            *(self._run_llm_task(item, semaphore) for item in selected),
            return_exceptions=True,
        )

    # -- finalization -------------------------------------------------------
    async def run(self) -> None:
        LOG.info("loaded %d tasks; budget=%d; workers=%d", len(self.tasks), self.config.budget, self.config.workers)
        await self.free_phase()
        LOG.info(
            "free phase done in %.1fs: solved=%d impossible=%d remaining_budget=%.0f",
            self.elapsed,
            sum(1 for task in self.tasks if getattr(task, "_done", False)),
            sum(1 for task in self.tasks if getattr(task, "_impossible", False)),
            self.meter.remaining,
        )
        await self.llm_phase()
        self.finalize()

    def finalize(self) -> None:
        # Record tasks that never got a result; do not create answer.txt for
        # them, because the organisers explicitly asked not to create a file
        # for unsolvable tasks.
        for task in self.tasks:
            if task.task_id not in self.store.records:
                self.store.record(
                    task,
                    status="skipped",
                    answer=None,
                    solved=False,
                    cost_x10=0,
                    method="not_run",
                    detail="budget/time/triage",
                )
        self.store.write_summary(
            {
                "elapsed_seconds": round(self.elapsed, 2),
                "meter": self.meter.snapshot(),
                "planner": self.planner_summary,
                "type_stats": self.stats,
            }
        )
        summary = self.store.summary()
        LOG.info(
            "done: solved=%s reward=%s cost=%.0f elapsed=%.1fs",
            summary.get("solved"),
            summary.get("reward"),
            summary.get("cost", 0.0),
            self.elapsed,
        )
