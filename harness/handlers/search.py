"""Knowledge-base search tasks.

The knowledge base is a set of small markdown files with ``Key: Value`` lines.
Most questions are one- or two-hop lookups (person -> employer -> company HQ),
so a deterministic graph resolver handles them without any tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..budget import BudgetExhausted
from ..verify import normalize_str
from .base import HandlerContext, HandlerResult, instruction_core, read_answer_file


@dataclass
class Entity:
    name: str
    kind: str
    fields: dict[str, str] = field(default_factory=dict)
    source: Optional[Path] = None


@dataclass
class SearchSolve:
    ok: bool
    value: Optional[str] = None
    impossible: bool = False
    reason: str = ""
    method: str = ""


def _normalize_name(value: str) -> str:
    text = (value or "").strip().strip("'\"")
    text = re.sub(r"'s$", "", text, flags=re.I)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip(" .?")


def _strip_prefix(name: str) -> str:
    for prefix in ("person_", "company_"):
        if name.lower().startswith(prefix):
            return name[len(prefix):]
    return name


_FIELD_ALIASES = {
    "born": "born",
    "born in": "born",
    "birth": "born",
    "birth year": "born",
    "year born": "born",
    "year of birth": "born",
    "date of birth": "born",
    "employer": "employer",
    "works for": "employer",
    "works at": "employer",
    "employed by": "employer",
    "company": "employer",
    "mentor": "mentor",
    "headquarters": "headquarters",
    "headquarters city": "headquarters",
    "headquartered in": "headquarters",
    "hq": "headquarters",
    "head office": "headquarters",
    "city": "headquarters",
    "location": "headquarters",
    "based in": "headquarters",
    "birthplace": "birthplace",
    "place of birth": "birthplace",
    "industry": "industry",
    "sector": "industry",
    "founded": "founded",
    "founded in": "founded",
    "ceo": "ceo",
    "chief executive": "ceo",
    "chief executive officer": "ceo",
    "head": "ceo",
    "founder": "founder",
}


def load_kb(task) -> tuple[dict[str, Entity], list[Path]]:
    base = task.work_dir / "kb"
    if not base.is_dir():
        base = task.work_dir
    entities: dict[str, Entity] = {}
    files: list[Path] = []
    for path in sorted(base.rglob("*.md")):
        if not path.is_file():
            continue
        files.append(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title = ""
        fields: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip().replace("**", "").replace("__", "")
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            if not line:
                continue
            if line.startswith("#") and not title:
                title = line.lstrip("#").strip()
                continue
            match = re.match(r"^([A-Za-z][A-Za-z0-9 /_-]{0,40}):\s*(.+)$", line)
            if match:
                key = match.group(1).strip().casefold()
                key = _FIELD_ALIASES.get(key, key)
                fields[key] = match.group(2).strip()

        name = title or _strip_prefix(path.stem).replace("_", " ").title()
        kind = "company" if "company" in path.stem.lower() else "person"
        entity = Entity(name=name, kind=kind, fields=fields, source=path)
        for key in {
            _normalize_name(name),
            _normalize_name(_strip_prefix(path.stem)),
            _normalize_name(path.stem),
        }:
            if key:
                entities.setdefault(key, entity)
    return entities, files


def find_entity(entities: dict[str, Entity], name: str, kind: Optional[str] = None) -> Optional[Entity]:
    key = _normalize_name(name)
    if not key:
        return None
    if key in entities:
        entity = entities[key]
        if kind is None or entity.kind == kind:
            return entity
    # Substring fallback: "Irina Smith" -> person_irina_smith.  Only accept a
    # unique match; an ambiguous name must go to the LLM/grep fallback rather
    # than silently picking the first entity.
    candidates = [
        (entity_key, entity)
        for entity_key, entity in entities.items()
        if key in entity_key or entity_key in key
    ]
    if kind is not None:
        candidates = [(k, e) for k, e in candidates if e.kind == kind]
    unique: dict[int, Entity] = {id(entity): entity for _, entity in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))
    return None


def _clean_question(question: str) -> str:
    text = (question or "").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _clean_answer(value: str, *, city: bool = False) -> str:
    text = (value or "").strip().strip("'\"")
    text = re.sub(r"\s+", " ", text)
    if city:
        text = text.split(",")[0].strip()
    return text.strip(" .")


def _question_patterns(question: str):
    """Yield (kind, entity_kind, regex) in priority order."""
    patterns = [
        ("born", "person", r"(?:in which year|what year|when)\s+was\s+(.+?)\s+born"),
        ("birthplace", "person", r"where\s+was\s+(.+?)\s+born"),
        ("birthplace", "person", r"what\s+is\s+the\s+birthplace\s+of\s+(.+)"),
        ("employer_hq_city", "person", r"(?:in which city|where)\s+is\s+the\s+headquarters\s+of\s+(.+?)'s\s+employer"),
        ("employer_hq_city", "person", r"(?:in which city|where)\s+is\s+the\s+headquarters\s+of\s+the\s+employer\s+of\s+(.+)"),
        ("employer_hq_city", "person", r"where\s+is\s+(.+?)'s\s+employer\s+headquartered"),
        ("mentor", "person", r"(?:who|what)\s+is\s+(?:the\s+)?mentor\s+of\s+(.+)"),
        ("mentor", "person", r"who\s+is\s+(.+?)'s\s+mentor"),
        ("employer", "person", r"what\s+is\s+the\s+employer\s+of\s+(.+)"),
        ("employer", "person", r"which\s+company\s+employs\s+(.+)"),
        ("employer", "person", r"which\s+company\s+is\s+(.+?)\s+employed\s+by"),
        ("employer", "person", r"who\s+employs\s+(.+)"),
        ("employer", "person", r"where\s+does\s+(.+?)\s+work"),
        ("employer", "person", r"what\s+is\s+the\s+name\s+of\s+(.+?)'s\s+employer"),
        ("hq_city", "company", r"(?:in which city|where)\s+is\s+(.+?)\s+headquartered"),
        ("hq_city", "company", r"what\s+is\s+the\s+headquarters\s+city\s+of\s+(.+)"),
        ("hq_city", "company", r"in\s+which\s+city\s+is\s+(.+?)\s+based"),
        ("hq_city", "company", r"(?:in which city|where)\s+is\s+the\s+headquarters\s+of\s+(.+)"),
        ("hq", "company", r"what\s+is\s+the\s+headquarters\s+of\s+(.+)"),
        ("industry", "company", r"what\s+industry\s+is\s+(.+?)\s+in"),
        ("industry", "company", r"what\s+is\s+the\s+industry\s+of\s+(.+)"),
        ("founded", "company", r"(?:when|in which year)\s+was\s+(.+?)\s+founded"),
        ("founded", "company", r"what\s+is\s+the\s+founded\s+year\s+of\s+(.+)"),
        ("ceo", "company", r"who\s+is\s+the\s+(?:ceo|chief\s+executive(?:\s+officer)?)\s+of\s+(.+)"),
        ("ceo", "company", r"who\s+heads\s+(.+)"),
    ]
    for kind, entity_kind, pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            yield kind, entity_kind, match.group(1).strip().strip("'\" .?")


def solve_deterministic(task) -> SearchSolve:
    entities, files = load_kb(task)
    if not files:
        return SearchSolve(False, impossible=True, reason="no knowledge base files")

    question = _clean_question(task.instruction)
    matched_any = False
    last_reason = ""
    for kind, entity_kind, raw_name in _question_patterns(question):
        matched_any = True
        entity = find_entity(entities, raw_name, entity_kind)
        if entity is None:
            # Maybe the role is ambiguous; try without the kind filter.
            entity = find_entity(entities, raw_name)
        if entity is None:
            last_reason = f"entity not found: {raw_name}"
            continue

        if kind == "born":
            value = entity.fields.get("born")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no born field for {entity.name}")
        elif kind == "birthplace":
            value = entity.fields.get("birthplace")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no birthplace field for {entity.name}")
        elif kind == "mentor":
            value = entity.fields.get("mentor")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no mentor field for {entity.name}")
        elif kind == "employer":
            value = entity.fields.get("employer")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no employer field for {entity.name}")
        elif kind == "employer_hq_city":
            employer = entity.fields.get("employer")
            if not employer:
                return SearchSolve(False, reason=f"no employer for {entity.name}")
            company = find_entity(entities, employer, "company") or find_entity(entities, employer)
            if company is None:
                return SearchSolve(False, reason=f"company not found: {employer}")
            value = company.fields.get("headquarters")
            if value:
                return SearchSolve(True, _clean_answer(value, city=True), method="search_kb_multihop")
            return SearchSolve(False, reason=f"no headquarters for {company.name}")
        elif kind in {"hq_city", "hq"}:
            value = entity.fields.get("headquarters")
            if value:
                return SearchSolve(True, _clean_answer(value, city=(kind == "hq_city")), method="search_kb")
            return SearchSolve(False, reason=f"no headquarters for {entity.name}")
        elif kind == "industry":
            value = entity.fields.get("industry")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no industry for {entity.name}")
        elif kind == "founded":
            value = entity.fields.get("founded")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no founded field for {entity.name}")
        elif kind == "ceo":
            value = entity.fields.get("ceo")
            if value:
                return SearchSolve(True, _clean_answer(value), method="search_kb")
            return SearchSolve(False, reason=f"no ceo for {entity.name}")

    if matched_any and last_reason:
        # Do not call this impossible: a different spelling or a nearby KB
        # entry may still answer it.  The LLM/grep fallback verifies that its
        # answer actually occurs in the snippets.
        return SearchSolve(False, reason=last_reason)
    return SearchSolve(False, reason="unrecognised question shape")


_STOPWORDS = {
    "the", "a", "an", "in", "which", "what", "who", "where", "when", "was", "were",
    "is", "are", "of", "for", "to", "and", "or", "did", "does", "do", "how",
    "many", "much", "city", "year", "born", "headquarters", "employer", "company",
    "knowledge", "base", "directory", "contains", "small", "files", "file", "use",
    "bash", "find", "work", "only", "inside", "current", "when", "you", "are",
    "done", "write", "final", "answer", "example", "with", "it", "its", "their",
}

# For grounding a free-form answer we *want* field words such as "born",
# "mentor" or "headquarters": they are the link between the question and the
# source line.  Only generic question/boilerplate words are dropped here.
_GROUNDING_STOPWORDS = {
    "the", "a", "an", "in", "which", "what", "who", "where", "when", "was",
    "were", "is", "are", "of", "for", "to", "and", "or", "did", "does", "do",
    "how", "many", "much", "knowledge", "base", "directory", "contains",
    "small", "files", "file", "use", "bash", "find", "work", "only", "inside",
    "current", "you", "done", "write", "final", "answer", "example", "with",
    "it", "its", "their", "please", "return",
}


_QUESTION_WORDS = {
    "who", "what", "which", "where", "when", "why", "whose", "how", "is",
    "are", "was", "were", "do", "does", "did", "the", "a", "an", "in", "on",
    "at", "of", "for", "to", "and", "or", "please", "return", "tell", "give",
    "use", "work", "only", "file", "text", "information", "show", "find",
}


def _snippet_sections(snippets: str):
    header = ""
    lines: list[str] = []
    for line in snippets.splitlines():
        if line.startswith("--- ") and line.endswith(" ---"):
            if lines:
                yield header, lines
            header = line
            lines = []
        else:
            lines.append(line)
    if lines:
        yield header, lines


def _line_grounded(answer: str, snippets: str, question: str) -> bool:
    """Require the answer to be grounded in the right KB entity section.

    The answer must occur on a line that shares a meaningful question keyword,
    and (when the question contains proper nouns) the same file section must
    mention those nouns.  This rejects, for example, "Olga Kowalski" as the
    CEO of Fjord Data just because that name appears elsewhere in the KB.
    """
    answer_norm = normalize_str(answer).casefold()
    if not answer_norm:
        return False
    qcore = instruction_core(question)
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", qcore.casefold())
        if len(token) >= 3 and token not in _GROUNDING_STOPWORDS
    }
    proper = {
        token.casefold()
        for token in re.findall(r"\b[A-Z][A-Za-z]+\b", qcore)
        if len(token) >= 3 and token.casefold() not in _QUESTION_WORDS
    }
    # The entity name itself is not enough: "favorite color of Maya Sato"
    # must not be answered by "Owner: Maya Sato".  Require a non-entity
    # question word on the answer line.
    non_entity_tokens = tokens - proper
    sections = list(_snippet_sections(snippets)) or [("", snippets.splitlines())]
    for header, lines in sections:
        section_norm = normalize_str(header + "\n" + "\n".join(lines)).casefold()
        if proper and not all(token in section_norm for token in proper):
            continue
        for line in lines:
            line_norm = normalize_str(line).casefold()
            if answer_norm not in line_norm:
                continue
            required = non_entity_tokens or tokens
            if not required:
                return True
            if any(token in line_norm for token in required):
                return True
            # The answer may be a section title/header (e.g. "# Parent 3")
            # while the question keyword is in the file name/body.
            if any(token in section_norm for token in required):
                return True
    return False


def relevant_snippets(task, files: list[Path], max_files: int = 3, max_chars: int = 5000) -> str:
    question = task.instruction.casefold()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", question)
        if len(token) >= 3 and token not in _STOPWORDS
    }
    scored: list[tuple[int, Path, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lowered = text.casefold()
        score = sum(1 for token in tokens if token in lowered)
        if score:
            scored.append((score, path, text))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    chunks: list[str] = []
    for _, path, text in scored[:max_files]:
        chunks.append(f"--- {path.name} ---\n{text[:max_chars]}")
    if not chunks:
        for path in files[:max_files]:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks.append(f"--- {path.name} ---\n{text[:max_chars]}")
    return "\n\n".join(chunks)[: max_chars * max_files]


class SearchHandler:
    name = "search"

    def try_free(self, task) -> Optional[HandlerResult]:
        result = solve_deterministic(task)
        if result.impossible:
            return HandlerResult(
                impossible=True,
                method="search_kb",
                detail=result.reason,
                confidence=0.9,
            )
        if result.ok and result.value:
            return HandlerResult(
                answer=result.value,
                solved=True,
                method=result.method,
                confidence=0.95,
                detail="deterministic KB lookup",
            )
        return None

    async def run(self, task, ctx: HandlerContext) -> HandlerResult:
        result = self.try_free(task)
        if result is not None:
            return result
        if not ctx.llm_enabled:
            return HandlerResult(solved=False, method="search_kb", detail="no deterministic match and LLM disabled")

        entities, files = load_kb(task)
        if not files:
            return HandlerResult(impossible=True, method="search_kb", detail="no knowledge base files")
        snippets = relevant_snippets(task, files)
        if not snippets:
            return HandlerResult(impossible=True, method="search_kb", detail="no matching snippets")

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied snippets. "
                    "Return only the exact answer, without a sentence or explanation. "
                    "If the answer is not explicitly present in the snippets, "
                    "return exactly NOT_FOUND."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {task.instruction}\n\nSnippets:\n{snippets}",
            },
        ]
        cost = 0
        attempts = 0
        answer = ""
        try:
            attempts += 1
            llm_result = await ctx.llm.call(task, messages, model_key="fast", max_tokens=50)
            cost += llm_result.cost_x10
            answer = _clean_answer((llm_result.content or "").splitlines()[0] if llm_result.content else "")
            candidates = [answer, read_answer_file(task, consume=True) or ""]
            for candidate in candidates:
                candidate = _clean_answer(candidate)
                if not candidate:
                    continue
                if candidate.casefold() in {"not_found", "not found", "none", "unknown"}:
                    continue
                # Accept only an answer that is actually grounded in a source
                # line together with a keyword from the question.  This
                # prevents hallucinated answers on impossible tasks.
                if _line_grounded(candidate, snippets, task.instruction):
                    return HandlerResult(
                        answer=candidate,
                        solved=True,
                        cost_x10=cost,
                        method="search_llm",
                        confidence=0.8,
                        attempts=attempts,
                    )
        except BudgetExhausted:
            raise
        except Exception as exc:
            return HandlerResult(
                solved=False,
                cost_x10=cost,
                method="search_llm",
                detail=f"{type(exc).__name__}: {exc}",
                attempts=attempts,
            )
        return HandlerResult(
            solved=False,
            cost_x10=cost,
            method="search_llm",
            detail=f"ungrounded answer: {answer[:80]}" if answer else "empty answer",
            attempts=attempts,
        )
