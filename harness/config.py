"""Configuration and CLI parsing.

The evaluator may pass arguments, environment variables, or both.  We accept
all reasonable variants (``--tasks-dir`` / ``TASKS``, ``--solutions-file`` /
``OUT``, ``--budget`` / ``BUDGET``).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


DEFAULT_BUDGET = 100_000
DEFAULT_WORKERS = 3
DEFAULT_DEADLINE_SECONDS = 55 * 60
DEFAULT_RESERVE = 5_000
DEFAULT_SAFETY_FACTOR = 1.10

# Conservative per-task caps in cost units (output tokens + input/10).
TASK_CAPS: Dict[str, int] = {
    "math": 700,
    "data_analysis": 900,
    "data": 900,
    "search": 700,
    "coding": 3_000,
    "text": 700,
    "text_extraction": 700,
    "json": 900,
    "json_query": 900,
    "sql": 1_200,
    "investigation": 5_000,
    "log_analysis": 5_000,
    "multi_csv": 2_500,
    "data_join": 2_500,
    "spreadsheet": 2_500,
    "xlsx": 2_500,
    "document": 3_000,
    "pdf": 3_000,
    "docx": 3_000,
    "logic": 1_800,
    "constraint": 1_800,
    "generic": 2_500,
    "unknown": 2_500,
}

DEFAULT_MODELS: Dict[str, str] = {
    "fast": "qwen3.6-35b-a3b-noreason",
    "coding": "qwen3.8-27b-noreason",
    "smart": "qwen3.8-27b",
    "fast_reason": "qwen3.6-35b-a3b",
    "smart_reason": "qwen3.8-27b",
}


def _parse_budget(raw: str, default: int = DEFAULT_BUDGET) -> int:
    """Accept 100000, 100_000, 100K, 1M (case-insensitive)."""
    text = str(raw or "").strip().replace("_", "").replace(" ", "").replace(",", "")
    if not text:
        return default
    multiplier = 1
    lowered = text.casefold()
    if lowered.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif lowered.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return max(0, int(float(text) * multiplier))
    except ValueError:
        return default


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


@dataclass
class Config:
    tasks_dir: Path
    out: Optional[Path] = None
    budget: int = DEFAULT_BUDGET
    reserve: int = DEFAULT_RESERVE
    workers: int = DEFAULT_WORKERS
    deadline_seconds: int = DEFAULT_DEADLINE_SECONDS
    safety_factor: float = DEFAULT_SAFETY_FACTOR

    llm_base_url: str = ""
    llm_api_key: str = ""
    models: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    model_timeout: float = 120.0
    max_retries: int = 2

    engine: str = "lean"  # lean | deepagent
    no_llm: bool = False
    verbose: bool = False
    log_dir: Path = field(default_factory=lambda: Path("/tmp/irunner"))

    # Filled in by the orchestrator at runtime.
    started_at: float = 0.0

    @property
    def budget_x10(self) -> int:
        return max(0, self.budget) * 10

    @property
    def reserve_x10(self) -> int:
        return max(0, self.reserve) * 10

    @property
    def answer_file_name(self) -> str:
        return "answer.txt"

    @property
    def qualifier_mode(self) -> bool:
        # The evaluator's preliminary run uses only a couple of tasks.  In
        # that mode we must not skip anything due to budget concerns.
        return False  # replaced by the orchestrator after loading tasks

    def task_cap(self, task_type: str, reward: int) -> int:
        base = TASK_CAPS.get(task_type, TASK_CAPS["unknown"])
        # Rewards are small integers (1..5 in the examples).  High-reward
        # tasks may use a little more, but never an unbounded amount.
        return int(base * (1.0 + max(0, reward) / 5.0))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-aware task-solving harness")
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--solutions-file", default="")
    parser.add_argument("tasks_dir_pos", nargs="?", default=None)
    parser.add_argument("solutions_file_pos", nargs="?", default=None)
    parser.add_argument("--budget", default=os.environ.get("BUDGET", ""))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--deadline", type=int, default=DEFAULT_DEADLINE_SECONDS)
    parser.add_argument("--reserve", type=int, default=DEFAULT_RESERVE)
    parser.add_argument("--engine", choices=["lean", "deepagent"], default="lean")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_config(argv: Optional[list[str]] = None) -> Config:
    args = _parse_args(argv)

    budget_raw = str(args.budget or "").strip()
    if not budget_raw:
        budget_raw = _env_str(
            "BUDGET", "TOKEN_BUDGET", "TOKENS_BUDGET", "LLM_TOKEN_BUDGET"
        )
    budget = _parse_budget(budget_raw, DEFAULT_BUDGET)

    tasks_raw = (
        args.tasks_dir
        or args.tasks_dir_pos
        or os.environ.get("TASKS")
        or "/data/tasks"
    )
    out_raw = (
        args.solutions_file
        or args.solutions_file_pos
        or args.out
        or os.environ.get("OUT", "")
    ).strip()
    out = Path(out_raw) if out_raw else None

    models = dict(DEFAULT_MODELS)
    for key, env_name in {
        "fast": "MODEL_FAST",
        "coding": "MODEL_CODING",
        "smart": "MODEL_SMART",
        "fast_reason": "MODEL_FAST_REASON",
        "smart_reason": "MODEL_SMART_REASON",
    }.items():
        value = _env_str(env_name)
        if value:
            models[key] = value

    log_dir = Path(os.environ.get("IRUNNER_LOG_DIR", "/tmp/irunner"))

    return Config(
        tasks_dir=Path(str(tasks_raw)).expanduser(),
        out=out,
        budget=budget,
        reserve=args.reserve,
        workers=max(1, min(5, int(args.workers))),
        deadline_seconds=max(60, int(args.deadline)),
        llm_base_url=_env_str("LLM_BASE_URL"),
        llm_api_key=_env_str("LLM_API_KEY"),
        models=models,
        engine=args.engine,
        no_llm=bool(args.no_llm),
        verbose=bool(args.verbose),
        log_dir=log_dir,
    )
