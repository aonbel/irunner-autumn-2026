"""Arithmetic word problems."""

from __future__ import annotations

import ast
import re
from typing import Optional

from ..budget import BudgetExhausted
from ..verify import extract_int
from .base import HandlerContext, HandlerResult, instruction_core, read_answer_file


_NUM = r"(\d[\d,]* )".replace(" ", "")
_START_RE = re.compile(r"(?:starts?|begins?) with\s+(\d[\d,]*)", re.I)
_HAS_RE = re.compile(
    r"^(?:(?:a|an|the)\s+\w+|it)\s+(?:has|contains|holds)\s+(\d[\d,]*)",
    re.I,
)
_SPLIT_RE = re.compile(
    r"split\s+(?:it\s+)?(?:equally\s+into|into\s+\d+\s+equal)\s+(\d+)",
    re.I,
)
_SPLIT_ALT_RE = re.compile(
    r"split\s+(?:it\s+)?(?:equally\s+into|into)\s+(\d+)\s+equal",
    re.I,
)
_DAYS_RE = re.compile(
    r"(?:ships?|sells?|sends?|removes?|returns?|uses?|loses?|buys?|adds?|orders?|"
    r"finds?|gains?|consumes?|produces?|delivers?|collects?|distributes?)\s+"
    r"(\d[\d,]*)\s+(?:boxes?\s+)?"
    r"(?:(?:per|each|every)\s+day|daily)\s+for\s+(\d+)\s+days?",
    re.I,
)
_DELIVERIES_RE = re.compile(
    r"(?:receives?|gets?|buys?|orders?)\s+(\d+)\s+(?:deliveries?|shipments?)\s+of\s+"
    r"(\d[\d,]*)\s+(?:boxes?\s+)?each",
    re.I,
)
_SHIP_RE = re.compile(r"(?:ships?|sells?|sends?|removes?|returns?|uses?|loses?)\s+(\d[\d,]*)\s+boxes?", re.I)
_RECEIVE_RE = re.compile(r"(?:receives?|gets?|buys?|adds?|orders?|finds?)\s+(\d[\d,]*)\s+boxes?", re.I)
_DOUBLE_RE = re.compile(r"\b(?:doubled|doubles|is doubled|are doubled)\b", re.I)
_TRIPLE_RE = re.compile(r"\b(?:tripled|triples|is tripled|are tripled)\b", re.I)
_HALF_RE = re.compile(r"\b(?:halved|halves|is halved|are halved)\b", re.I)
def looks_impossible(instruction: str) -> tuple[bool, str]:
    """Detect the obvious "missing data" impossible variants.

    This is deliberately conservative: it only fires when the text explicitly
    describes an operation but the required quantity is absent.  A task with
    an unusual but complete wording is left to the LLM fallback.
    """
    text = re.sub(r"\s+", " ", instruction or "")
    if not re.search(r"\d", text):
        return True, "no numeric data"
    if re.search(r"\b(?:starts?|begins?)\s+with\b", text, re.I) and not _START_RE.search(text):
        return True, "missing starting quantity"
    # Narrow "quantity is literally absent" patterns.  A complete sentence
    # with an unusual verb is left to the LLM rather than being misclassified.
    if re.search(
        r"\b(?:ships?|sells?|sends?|removes?|returns?|uses?|loses?|buys?)\s+"
        r"boxes?\s+(?:per|each|every)\s+day",
        text,
        re.I,
    ):
        return True, "missing daily quantity"
    if re.search(r"\b(?:deliveries?|shipments?)\s+of\s+boxes?\s+each", text, re.I):
        return True, "missing delivery quantity"
    return False, ""


_SEMANTIC_HAZARD = re.compile(
    r"\b(?:cancelled|canceled|returned|damaged|lost|stolen|expired|discount|"
    r"tax|percent|unless|except|excluding|only|at\s+least|at\s+most|above|"
    r"below|more\s+than|less\s+than|fewer|if|but|rounded\s+up|round\s+up|"
    r"ceiling|ceil|interest|fee|fees|tip|refund|per\s+week|per\s+month|"
    r"per\s+hour|per\s+year|weekly|monthly|hourly|yearly)\b",
    re.I,
)


_OPERATION_WORDS = re.compile(
    r"\b(?:starts?|begins?|split|doubled?|tripled?|halved?|halves|ships?|sells?|sends?|"
    r"receives?|gets?|buys?|adds?|removes?|returns?|uses?|loses?|orders?|finds?|"
    r"contains?|holds?|deliveries?|shipments?|arrives?|more|additional|another|"
    r"gives?|takes?|drops?|increases?|decreases?|twice|thrice|percent)\b",
    re.I,
)


def _to_int(raw: str) -> int:
    return int(raw.replace(",", "").strip())


def solve_deterministic(instruction: str) -> Optional[int]:
    text = re.sub(r"\s+", " ", instruction or "").strip()
    if not text:
        return None
    core = instruction_core(text)
    # Words like "but one delivery was cancelled" change the arithmetic in a
    # way the simple templates cannot express.  Hand those to the LLM.
    if _SEMANTIC_HAZARD.search(core):
        return None
    # Ultra-safe: only the "how many ... at the end" question shape is
    # handled deterministically.
    if not re.search(r"\bhow\s+many\b", core, re.I):
        return None
    # Split on sentence boundaries, but keep the original order.
    sentences = re.split(r"(?<=[.!?])\s+", core)

    value: Optional[int] = None
    operations = 0
    for sentence in sentences:
        lowered = sentence.lower()
        if "how many" in lowered or "compute it" in lowered or "work only" in lowered:
            continue

        matched = False

        match = _START_RE.search(sentence)
        if match:
            candidate = _to_int(match.group(1))
            value = candidate if value is None else value
            matched = True
        elif _HAS_RE.search(sentence):
            candidate = _to_int(_HAS_RE.search(sentence).group(1))
            value = candidate if value is None else value
            matched = True

        match = _SPLIT_RE.search(sentence) or _SPLIT_ALT_RE.search(sentence)
        if match:
            if value is None:
                return None
            value //= int(match.group(1))
            operations += 1
            matched = True

        if _DOUBLE_RE.search(sentence):
            if value is None:
                return None
            value *= 2
            operations += 1
            matched = True

        if _TRIPLE_RE.search(sentence):
            if value is None:
                return None
            value *= 3
            operations += 1
            matched = True

        if _HALF_RE.search(sentence):
            if value is None:
                return None
            value //= 2
            operations += 1
            matched = True

        match = _DAYS_RE.search(sentence)
        if match:
            if value is None:
                return None
            value -= _to_int(match.group(1)) * int(match.group(2))
            operations += 1
            matched = True

        match = _DELIVERIES_RE.search(sentence)
        if match:
            if value is None:
                return None
            value += int(match.group(1)) * _to_int(match.group(2))
            operations += 1
            matched = True

        # A sentence that mentions a daily/periodic rate or deliveries, but
        # did not match the corresponding structured pattern, is ambiguous.
        # Do not fall through to the simple "ships N boxes" pattern.
        if re.search(r"\b(?:per|each|every)\s+(?:day|week|month|hour|year)\b|\b(?:daily|weekly|monthly|hourly|yearly)\b", sentence, re.I) and not matched:
            return None
        if re.search(r"\bfor\s+\d+\s+(?:days?|weeks?|months?|hours?|years?)\b", sentence, re.I) and not _DAYS_RE.search(sentence):
            return None

        if not matched:
            match = _SHIP_RE.search(sentence)
            if match:
                if value is None:
                    return None
                value -= _to_int(match.group(1))
                operations += 1
                matched = True

        if not matched:
            match = _RECEIVE_RE.search(sentence)
            if match:
                if value is None:
                    return None
                value += _to_int(match.group(1))
                operations += 1
                matched = True

        if not matched and (_OPERATION_WORDS.search(sentence) or re.search(r"\d", sentence)):
            # Any unrecognised sentence with an operation word or a number is
            # a risk.  Do not guess; use the LLM instead.
            return None

    if value is None or operations == 0:
        return None
    return value


_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Load,
)


def safe_eval(expression: str) -> Optional[int]:
    expression = (expression or "").strip().strip("`").strip()
    if not expression:
        return None
    # Keep only the first plausible expression line.
    expression = expression.splitlines()[0].strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        result = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(result, float):
        if result.is_integer():
            return int(result)
        return None
    if isinstance(result, int):
        return result
    return None


def extract_expression(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer a fenced code block or a line containing arithmetic.
    for candidate in re.findall(r"```(?:python)?\s*(.*?)```", text, re.S):
        candidate = candidate.strip()
        if candidate:
            return candidate.splitlines()[0].strip()
    for line in text.splitlines():
        line = line.strip().strip("`")
        if re.fullmatch(r"[0-9+\-*/%().\s]+", line) and any(ch.isdigit() for ch in line):
            return line
    return None


class MathHandler:
    name = "math"

    def try_free(self, task) -> Optional[HandlerResult]:
        value = solve_deterministic(task.instruction)
        if value is not None:
            return HandlerResult(
                answer=str(value),
                solved=True,
                method="math_template",
                confidence=0.98,
                detail="deterministic arithmetic template",
            )
        impossible, reason = looks_impossible(task.instruction)
        if impossible:
            return HandlerResult(
                impossible=True,
                method="math_template",
                detail=reason,
                confidence=0.9,
            )
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        result = self.try_free(task)
        if result is not None:
            return result
        if not ctx.llm_enabled:
            return HandlerResult(
                solved=False,
                method="math_template",
                detail="no deterministic parse and LLM disabled",
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the arithmetic word problem. Return ONLY a Python "
                    "expression that computes the answer using the numbers from "
                    "the problem. Do not use variables, imports, or functions. "
                    "Use // for integer division. Example: (1752 // 2 * 2 // 3)"
                ),
            },
            {"role": "user", "content": task.instruction},
        ]
        cost = 0
        attempts = 0
        try:
            attempts += 1
            llm_result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=100)
            cost += llm_result.cost_x10
            expression = extract_expression(llm_result.content)
            value = safe_eval(expression) if expression else None
            if value is not None:
                return HandlerResult(
                    answer=str(value),
                    solved=True,
                    cost_x10=cost,
                    method="math_llm_expression",
                    confidence=0.8,
                    attempts=attempts,
                    detail=expression or "",
                )

            messages.append({"role": "assistant", "content": llm_result.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "Return ONLY the final integer, with no explanation.",
                }
            )
            attempts += 1
            llm_result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=40)
            cost += llm_result.cost_x10
            value = extract_int(llm_result.content or "")
            if value is not None:
                return HandlerResult(
                    answer=str(value),
                    solved=True,
                    cost_x10=cost,
                    method="math_llm_number",
                    confidence=0.6,
                    attempts=attempts,
                )
        except BudgetExhausted:
            raise
        except Exception as exc:  # budget exhaustion is handled upstream
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="math_llm",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
        file_answer = read_answer_file(task, consume=True)
        if file_answer:
            value = extract_int(file_answer)
            if value is not None:
                return HandlerResult(
                    answer=str(value),
                    solved=True,
                    cost_x10=cost,
                    method="math_llm_file",
                    confidence=0.7,
                    attempts=attempts,
                )
        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="math_llm",
            detail="could not extract an answer",
            attempts=attempts,
        )
