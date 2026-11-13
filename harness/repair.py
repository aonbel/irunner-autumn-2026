"""Deterministic test-driven repair for small coding tasks.

The example coding tasks contain one or two classic bugs (wrong operator,
``sorted`` instead of ``reversed``, wrong parity check).  Before spending any
LLM tokens we try AST mutations and run the visible test suite.  This is a
classical search method and costs zero tokens.

The search is deliberately bounded: single mutations first, then pairs from a
small beam.  If nothing passes, the original source is restored and the caller
can fall back to an LLM.
"""

from __future__ import annotations

import ast
import copy
import itertools
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .verify import find_source_files, run_tests


@dataclass(frozen=True)
class Candidate:
    file: Path
    path: tuple
    kind: str
    description: str


@dataclass
class RepairResult:
    passed: bool
    method: str = ""
    mutations: tuple[str, ...] = ()
    output: str = ""
    evaluations: int = 0
    timed_out: bool = False


def _iter_nodes(node: ast.AST, path: tuple = ()):
    """Yield ``(node, path)`` in a deterministic pre-order traversal."""
    yield node, path
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield from _iter_nodes(child, path + ((field, index),))
        elif isinstance(value, ast.AST):
            yield from _iter_nodes(value, path + ((field, None),))


def _candidate_kind(node: ast.AST) -> Optional[tuple[str, str]]:
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return "binop_add_sub", "swap +/-"
        if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
            return "binop_mul_div", "swap */"
    if isinstance(node, ast.AugAssign):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return "aug_add_sub", "swap +=/-="
        if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
            return "aug_mul_div", "swap *=//="
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = node.ops[0]
        if isinstance(op, (ast.Lt, ast.Gt)):
            return "cmp_lt_gt", "swap </>"
        if isinstance(op, (ast.LtE, ast.GtE)):
            return "cmp_lte_gte", "swap <=/>="
        if isinstance(op, (ast.Eq, ast.NotEq)):
            return "cmp_eq_ne", "swap ==/!="
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        slice_node = node.slice
        if slice_node.lower is None and slice_node.upper is not None:
            return "slice_flip", "[:n] <-> [-n:]"
        if (
            isinstance(slice_node.lower, ast.UnaryOp)
            and isinstance(slice_node.lower.op, ast.USub)
            and slice_node.upper is None
        ):
            return "slice_flip", "[-n:] <-> [:n]"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in {"sorted", "reversed"}:
            return "call_sorted_reversed", "sorted <-> reversed"
        if name in {"min", "max"}:
            return "call_min_max", "min <-> max"
        if name in {"sum", "len"}:
            return "call_sum_len", "sum <-> len"
    # ``x % 2 == 0`` <-> ``x % 2 == 1`` (a very common injected bug).
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        left = node.left
        right = node.comparators[0] if node.comparators else None
        if (
            isinstance(left, ast.BinOp)
            and isinstance(left.op, ast.Mod)
            and isinstance(right, ast.Constant)
            and right.value in (0, 1)
        ):
            return "mod_const_swap", "swap %2 == 0/1"
    return None


def collect_candidates(path: Path, source: str) -> list[Candidate]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    result: list[Candidate] = []
    for node, node_path in _iter_nodes(tree):
        kind = _candidate_kind(node)
        if kind is None:
            continue
        result.append(Candidate(path, node_path, kind[0], kind[1]))
    return result


def _get_parent(root: ast.AST, path: tuple) -> tuple[ast.AST, object, Optional[int]]:
    if not path:
        raise ValueError("empty path")
    node = root
    for field, index in path[:-1]:
        value = getattr(node, field)
        node = value[index] if index is not None else value
    field, index = path[-1]
    return node, field, index


def _mutate_node(node: ast.AST, kind: str) -> bool:
    if kind == "binop_add_sub" and isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
        else:
            return False
        return True
    if kind == "binop_mul_div" and isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult):
            node.op = ast.Div()
        elif isinstance(node.op, (ast.Div, ast.FloorDiv)):
            node.op = ast.Mult()
        else:
            return False
        return True
    if kind == "aug_add_sub" and isinstance(node, ast.AugAssign):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
        else:
            return False
        return True
    if kind == "aug_mul_div" and isinstance(node, ast.AugAssign):
        if isinstance(node.op, ast.Mult):
            node.op = ast.Div()
        elif isinstance(node.op, (ast.Div, ast.FloorDiv)):
            node.op = ast.Mult()
        else:
            return False
        return True
    if kind == "cmp_lt_gt" and isinstance(node, ast.Compare) and len(node.ops) == 1:
        if isinstance(node.ops[0], ast.Lt):
            node.ops[0] = ast.Gt()
        elif isinstance(node.ops[0], ast.Gt):
            node.ops[0] = ast.Lt()
        else:
            return False
        return True
    if kind == "cmp_lte_gte" and isinstance(node, ast.Compare) and len(node.ops) == 1:
        if isinstance(node.ops[0], ast.LtE):
            node.ops[0] = ast.GtE()
        elif isinstance(node.ops[0], ast.GtE):
            node.ops[0] = ast.LtE()
        else:
            return False
        return True
    if kind == "cmp_eq_ne" and isinstance(node, ast.Compare) and len(node.ops) == 1:
        if isinstance(node.ops[0], ast.Eq):
            node.ops[0] = ast.NotEq()
        elif isinstance(node.ops[0], ast.NotEq):
            node.ops[0] = ast.Eq()
        else:
            return False
        return True
    if kind == "slice_flip" and isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        slice_node = node.slice
        if slice_node.lower is None and slice_node.upper is not None:
            upper = copy.deepcopy(slice_node.upper)
            node.slice = ast.Slice(
                lower=ast.UnaryOp(op=ast.USub(), operand=upper),
                upper=None,
                step=None,
            )
            return True
        if (
            isinstance(slice_node.lower, ast.UnaryOp)
            and isinstance(slice_node.lower.op, ast.USub)
            and slice_node.upper is None
        ):
            operand = copy.deepcopy(slice_node.lower.operand)
            node.slice = ast.Slice(lower=None, upper=operand, step=None)
            return True
        return False
    if kind == "call_sorted_reversed" and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "sorted":
            node.func.id = "reversed"
        elif node.func.id == "reversed":
            node.func.id = "sorted"
        else:
            return False
        return True
    if kind == "call_min_max" and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "min":
            node.func.id = "max"
        elif node.func.id == "max":
            node.func.id = "min"
        else:
            return False
        return True
    if kind == "call_sum_len" and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "sum":
            node.func.id = "len"
        elif node.func.id == "len":
            node.func.id = "sum"
        else:
            return False
        return True
    if kind == "mod_const_swap" and isinstance(node, ast.Compare):
        if not node.comparators or not isinstance(node.comparators[0], ast.Constant):
            return False
        value = node.comparators[0].value
        if value not in (0, 1):
            return False
        node.comparators[0] = ast.Constant(value=1 - value)
        return True
    return False


def apply_mutations(source: str, candidates: Iterable[Candidate]) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    by_path = {candidate.path: candidate for candidate in candidates}
    if not by_path:
        return source
    for node, node_path in _iter_nodes(tree):
        candidate = by_path.get(node_path)
        if candidate is None:
            continue
        _mutate_node(node, candidate.kind)
    try:
        return ast.unparse(tree) + "\n"
    except Exception:
        return None


def _write_sources(originals: dict[Path, str], contents: dict[Path, str]) -> None:
    for path, content in contents.items():
        path.write_text(content, encoding="utf-8")


def _restore(originals: dict[Path, str]) -> None:
    for path, content in originals.items():
        try:
            path.write_text(content, encoding="utf-8")
        except OSError:
            pass


def repair(
    work_dir: Path,
    *,
    max_single: int = 80,
    max_pairs: int = 120,
    max_triples: int = 220,
    timeout: float = 5.0,
    max_seconds: float = 45.0,
) -> RepairResult:
    # Stale bytecode can mask a mutation, especially when the source length
    # happens to match; remove it once before trying candidates.
    shutil.rmtree(work_dir / "__pycache__", ignore_errors=True)

    source_paths = find_source_files(work_dir)
    if not source_paths:
        return RepairResult(False, output="no source files")

    originals: dict[Path, str] = {}
    candidates: list[Candidate] = []
    for path in source_paths:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        originals[path] = source
        candidates.extend(collect_candidates(path, source))

    if not candidates:
        return RepairResult(False, output="no mutation candidates")

    # Keep the search bounded and prefer files with many candidates (the
    # module under test usually has the most).
    by_file: dict[Path, list[Candidate]] = {}
    for candidate in candidates:
        by_file.setdefault(candidate.file, []).append(candidate)

    evaluations = 0
    scored: list[tuple[int, tuple[Candidate, ...]]] = []
    deadline = time.monotonic() + max(1.0, max_seconds)
    timed_out = False

    def evaluate(combo: tuple[Candidate, ...]) -> Optional[bool]:
        nonlocal evaluations, timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return None
        evaluations += 1
        contents: dict[Path, str] = {}
        for file in {c.file for c in combo}:
            file_candidates = [c for c in combo if c.file == file]
            mutated = apply_mutations(originals[file], file_candidates)
            if mutated is None:
                return None
            contents[file] = mutated
        try:
            _write_sources(originals, contents)
            result = run_tests(work_dir, timeout=timeout)
            if result.passed:
                return True
            scored.append((result.failure_count, combo))
            return False
        finally:
            _restore({file: originals[file] for file in contents})

    # 1. Single mutations.
    single_candidates = candidates[:max_single]
    for candidate in single_candidates:
        outcome = evaluate((candidate,))
        if outcome:
            source = apply_mutations(originals[candidate.file], (candidate,))
            if source is not None:
                candidate.file.write_text(source, encoding="utf-8")
            return RepairResult(
                True,
                method="mutation_single",
                mutations=(f"{candidate.file.name}:{candidate.description}",),
                evaluations=evaluations,
            )
        if timed_out:
            break

    # 2. Pairs, using a beam of the best-scoring single mutations.
    scored.sort(key=lambda item: item[0])
    beam = [combo for _, combo in scored[:16]]
    pair_count = 0
    for first, second in itertools.combinations(beam, 2):
        if timed_out:
            break
        if first[0].file != second[0].file:
            continue
        pair_count += 1
        if pair_count > max_pairs:
            break
        combo = (first[0], second[0])
        # Avoid duplicate candidate paths.
        if combo[0].path == combo[1].path:
            continue
        outcome = evaluate(combo)
        if outcome:
            mutated = apply_mutations(originals[combo[0].file], combo)
            if mutated is not None:
                combo[0].file.write_text(mutated, encoding="utf-8")
            return RepairResult(
                True,
                method="mutation_pair",
                mutations=tuple(
                    f"{c.file.name}:{c.description}" for c in combo
                ),
                evaluations=evaluations,
            )

    # 3. Triples, using the best single mutations as a beam.  This covers
    # tasks with three independent injected bugs while staying bounded.
    top_singles: list[Candidate] = []
    for _, combo in scored:
        if len(combo) == 1 and combo[0] not in top_singles:
            top_singles.append(combo[0])
        if len(top_singles) >= 12:
            break
    triple_count = 0
    for combo in itertools.combinations(top_singles, 3):
        if timed_out:
            break
        if len({candidate.file for candidate in combo}) != 1:
            continue
        if len({candidate.path for candidate in combo}) != 3:
            continue
        triple_count += 1
        if triple_count > max_triples:
            break
        outcome = evaluate(combo)
        if outcome:
            mutated = apply_mutations(originals[combo[0].file], combo)
            if mutated is not None:
                combo[0].file.write_text(mutated, encoding="utf-8")
            return RepairResult(
                True,
                method="mutation_triple",
                mutations=tuple(f"{c.file.name}:{c.description}" for c in combo),
                evaluations=evaluations,
            )

    return RepairResult(
        False,
        output="repair timed out" if timed_out else "no mutation passed",
        evaluations=evaluations,
        timed_out=timed_out,
    )
