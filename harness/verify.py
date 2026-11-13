"""Local verification of answers.

The organisers run hidden tests, but for the four known task types we can
replicate most checks locally:

* coding -- run the visible unittest suite;
* exact_int / exact_str -- validate the answer's shape (the final set has no
  ground-truth field);
* otherwise validate that the answer is non-empty.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    passed: bool
    output: str
    command: list[str]
    failure_count: int
    timed_out: bool = False

    @property
    def summary(self) -> str:
        if self.timed_out:
            return "tests timed out"
        if self.passed:
            return "tests passed"
        return f"tests failed ({self.failure_count} failures)"


def find_test_files(work_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(work_dir.rglob("*.py")):
        name = path.name.lower()
        if name.startswith("test_") or name.endswith("_test.py"):
            files.append(path)
    return files


def find_source_files(work_dir: Path) -> list[Path]:
    sources: list[Path] = []
    for path in sorted(work_dir.rglob("*.py")):
        name = path.name.lower()
        if name.startswith("test_") or name.endswith("_test.py"):
            continue
        if "__pycache__" in path.parts:
            continue
        sources.append(path)
    return sources


def _failure_count(output: str) -> int:
    match = re.search(r"FAILED\s*\(([^)]*)\)", output)
    if match:
        parts = match.group(1)
        failures = re.search(r"failures=(\d+)", parts)
        errors = re.search(r"errors=(\d+)", parts)
        return int(failures.group(1) if failures else 0) + int(
            errors.group(1) if errors else 0
        )
    return output.count("\nFAIL:") + output.count("\nERROR:")


def run_tests(work_dir: Path, *, timeout: float = 30.0) -> TestResult:
    test_files = find_test_files(work_dir)
    if not test_files:
        return TestResult(
            passed=False,
            output="no test files found",
            command=[],
            failure_count=1,
        )

    preferred = work_dir / "test_mylib.py"
    if preferred.is_file():
        command = [sys.executable, str(preferred)]
    else:
        command = [sys.executable, "-m", "unittest", "discover", "-v"]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            command,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        output = output[-8000:]
        no_tests = "Ran 0 tests" in output
        return TestResult(
            passed=proc.returncode == 0 and not no_tests,
            output=output,
            command=command,
            failure_count=_failure_count(output) if not no_tests else 1,
        )
    except subprocess.TimeoutExpired as exc:
        output = ""
        for stream in (exc.stdout, exc.stderr):
            if isinstance(stream, bytes):
                output += stream.decode("utf-8", "replace")
            elif stream:
                output += str(stream)
        return TestResult(
            passed=False,
            output=output[-8000:],
            command=command,
            failure_count=1,
            timed_out=True,
        )
    except OSError as exc:
        return TestResult(
            passed=False,
            output=f"failed to run tests: {exc}",
            command=command,
            failure_count=1,
        )


_INT_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_int(text: str) -> Optional[int]:
    if text is None:
        return None
    matches = _INT_RE.findall(str(text))
    if not matches:
        return None
    try:
        value = float(matches[-1].replace(",", ""))
    except ValueError:
        return None
    if not value.is_integer():
        return None
    return int(value)


def normalize_str(text: str) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    value = value.strip().strip(".").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def validate_answer(task, answer: str, *, run_coding_tests: bool = True) -> tuple[bool, str]:
    """Return ``(valid, reason)``.

    For exact checks we can only check that the answer has the right shape.
    The deterministic handlers are high-confidence; the LLM fallbacks are
    best-effort.
    """
    answer = normalize_str(answer)
    if not answer:
        return False, "empty answer"

    check = (task.check or "").strip().lower()
    if task.is_coding or check == "unittest":
        if not run_coding_tests:
            return True, "coding answer accepted"
        result = run_tests(task.work_dir)
        if result.passed:
            return True, result.summary
        return False, result.summary

    if check == "exact_int":
        value = extract_int(answer)
        if value is None:
            return False, "not an integer"
        return True, f"integer {value}"

    if check == "exact_str":
        return True, "string"

    return bool(answer), "non-empty"


def detect_impossible_tests(work_dir: Path) -> tuple[bool, str]:
    """Detect obviously unsatisfiable test suites.

    This is intentionally conservative: we only flag tests with a literal
    false assertion or a syntax error.  Missing imports may be fixable by the
    agent, so they are not flagged here.
    """
    for path in find_test_files(work_dir):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return True, f"syntax error in {path.name}: {exc.msg}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                test = node.test
                if isinstance(test, ast.Constant) and not bool(test.value):
                    return True, f"literal false assertion in {path.name}"
                if (
                    isinstance(test, ast.Compare)
                    and len(test.comparators) == 1
                    and isinstance(test.left, ast.Constant)
                    and isinstance(test.comparators[0], ast.Constant)
                    and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
                ):
                    left = test.left.value
                    right = test.comparators[0].value
                    if isinstance(test.ops[0], ast.Eq) and left != right:
                        return True, f"impossible equality in {path.name}"
                    if isinstance(test.ops[0], ast.NotEq) and left == right:
                        return True, f"impossible inequality in {path.name}"
    return False, ""
