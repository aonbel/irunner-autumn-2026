#!/usr/bin/env python3
"""Evaluate the harness on a training set.

Ground truth lives in a separate ``_truth.json`` next to the task directories;
it is never placed inside ``task.json`` and is never read by the harness.  The
final competition set has no ground truth at all, so this is a development
tool only.

Usage:
    python scripts/eval.py --tasks /tmp/irunner-trainset
    python scripts/eval.py --tasks /tmp/irunner-trainset --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.tasks import load_tasks  # noqa: E402
from harness.verify import extract_int, normalize_str, run_tests  # noqa: E402


def _matches(actual: str, wanted: str, check: str) -> bool:
    if check == "exact_int":
        left = extract_int(actual)
        right = extract_int(wanted)
        return left is not None and right is not None and left == right
    return normalize_str(actual).casefold() == normalize_str(wanted).casefold()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks).expanduser().resolve()
    tasks = load_tasks(tasks_dir)
    truth_path = tasks_dir / "_truth.json"
    truth = {}
    if truth_path.is_file():
        truth = json.loads(truth_path.read_text(encoding="utf-8")).get("tasks", {})
    out_dir = Path(args.out).expanduser().resolve() if args.out else Path(tempfile.mkdtemp(prefix="irunner-eval-"))

    cmd = [
        sys.executable,
        str(ROOT / "run_agent.py"),
        "--tasks-dir",
        str(tasks_dir),
        "--out",
        str(out_dir),
        "--budget",
        str(args.budget),
    ]
    if args.no_llm:
        cmd.append("--no-llm")
    env = dict(os.environ)
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        timeout=args.timeout,
        stdout=subprocess.DEVNULL,
    )
    elapsed = time.monotonic() - started

    solutions_path = out_dir / "solutions.json"
    if not solutions_path.is_file():
        print(json.dumps({"error": "solutions.json not found", "returncode": proc.returncode}, indent=2))
        return 2
    payload = json.loads(solutions_path.read_text(encoding="utf-8"))
    records = payload.get("solutions", {})

    correct_reward = 0
    max_reward = 0
    correct = 0
    solvable = 0
    impossible = 0
    unscored = 0
    false_negative = []
    false_positive_impossible = []
    by_type: dict[str, dict] = {}

    for task in tasks:
        record = records.get(task.task_id, {})
        bucket = by_type.setdefault(
            task.task_type,
            {"tasks": 0, "solvable": 0, "correct": 0, "reward": 0, "max_reward": 0},
        )
        bucket["tasks"] += 1
        entry = truth.get(task.task_id)
        answer_path = task.answer_path
        has_answer = answer_path.is_file()
        answer = answer_path.read_text(encoding="utf-8").strip() if has_answer else ""

        if entry is None:
            # No local ground truth (e.g. a final-set replica): we can only
            # check coding tasks via their own tests.
            unscored += 1
            continue

        if not entry.get("solvable"):
            impossible += 1
            if has_answer or record.get("solved"):
                false_positive_impossible.append(task.task_id)
            continue

        wanted = entry.get("answer")
        solvable += 1
        max_reward += task.reward
        bucket["solvable"] += 1
        bucket["max_reward"] += task.reward

        ok = False
        if task.is_coding:
            ok = bool(record.get("solved")) and answer.strip().upper() == "DONE" and run_tests(task.work_dir).passed
        elif task.check == "exact_int":
            ok = has_answer and _matches(answer, wanted, "exact_int")
        elif task.check == "exact_str":
            ok = has_answer and _matches(answer, wanted, "exact_str")
        else:
            ok = bool(record.get("solved")) and has_answer

        if ok:
            correct += 1
            correct_reward += task.reward
            bucket["correct"] += 1
            bucket["reward"] += task.reward
        else:
            false_negative.append(task.task_id)

    result = {
        "tasks": len(tasks),
        "solvable": solvable,
        "impossible": impossible,
        "unscored": unscored,
        "correct": correct,
        "score": correct_reward,
        "max_score": max_reward,
        "accuracy_on_solvable": round(correct / solvable, 4) if solvable else 0.0,
        "false_negative": false_negative,
        "false_positive_impossible": false_positive_impossible,
        "cost": payload.get("summary", {}).get("cost", 0.0),
        "elapsed_seconds": round(elapsed, 2),
        "returncode": proc.returncode,
        "out_dir": str(out_dir),
        "by_type": by_type,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"score={result['score']}/{result['max_score']} "
            f"correct={result['correct']}/{result['solvable']} "
            f"impossible={result['impossible']} "
            f"cost={result['cost']:.0f} "
            f"time={result['elapsed_seconds']:.1f}s"
        )
        if false_negative:
            print(f"false negatives ({len(false_negative)}): {', '.join(false_negative[:12])}")
        if false_positive_impossible:
            print(
                f"answered impossible ({len(false_positive_impossible)}): "
                f"{', '.join(false_positive_impossible[:12])}"
            )
        for task_type, stats in sorted(by_type.items()):
            print(
                f"  {task_type:14s} {stats['correct']}/{stats['solvable']} solvable, "
                f"reward {stats['reward']}/{stats['max_reward']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
