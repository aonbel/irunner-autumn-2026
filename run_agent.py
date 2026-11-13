#!/usr/bin/env python3
"""Entry point for the irunner budget-aware harness."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import threading

from harness.budget import TokenMeter
from harness.config import load_config
from harness.llm import LLMRouter
from harness.orchestrator import Orchestrator
from harness.persist import SolutionStore
from harness.tasks import load_tasks


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Keep third-party loggers quiet; the proxy may be chatty.
    for name in (
        "httpx",
        "httpcore",
        "openai",
        "openjiuwen",
        "urllib3",
        "alembic",
        "sqlalchemy",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv=None) -> int:
    config = load_config(argv)
    setup_logging(config.verbose)
    log = logging.getLogger("irunner")

    try:
        config.log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        from pathlib import Path

        config.log_dir = Path("/tmp/irunner")
        config.log_dir.mkdir(parents=True, exist_ok=True)

    try:
        file_handler = logging.FileHandler(config.log_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    except OSError as exc:
        log.warning("cannot open log file: %s", exc)

    log.info("tasks_dir=%s out=%s budget=%s", config.tasks_dir, config.out, config.budget)
    import os as _os
    if not _os.environ.get("BUDGET"):
        log.warning("BUDGET is not set; using default %s output tokens", config.budget)
    if not config.llm_base_url or not config.llm_api_key:
        log.warning("LLM_BASE_URL/LLM_API_KEY are not set; running deterministic-only")
        config.no_llm = True
    if config.engine == "deepagent":
        log.warning(
            "--engine=deepagent is experimental; the lean openJiuwen Model/Tool "
            "engine is used for exact budget accounting"
        )

    try:
        tasks = load_tasks(config.tasks_dir)
    except Exception as exc:
        log.error("cannot load tasks: %s: %s", type(exc).__name__, exc)
        return 2

    meter = TokenMeter(
        config.budget,
        reserve=config.reserve,
        safety_factor=config.safety_factor,
    )
    llm = LLMRouter(config, meter)
    store = SolutionStore(config.out, task_root=config.tasks_dir)
    stop_event = threading.Event()

    def handle_signal(signum, _frame):
        log.warning("received signal %s; stopping after current work", signum)
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    except (ValueError, OSError):
        pass

    orchestrator = Orchestrator(
        config=config,
        tasks=tasks,
        meter=meter,
        llm=llm,
        store=store,
        stop_event=stop_event,
    )

    try:
        asyncio.run(orchestrator.run())
    except Exception as exc:
        log.exception("fatal orchestrator error: %s: %s", type(exc).__name__, exc)
        try:
            orchestrator.finalize()
        except Exception:
            log.exception("finalize failed")

    summary = store.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
