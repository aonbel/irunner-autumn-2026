#!/usr/bin/env python3
"""Generate a synthetic 100-task training set.

The final set is limited to the four example task types, so this generator
mirrors their structure and wording while varying numbers and injecting
classic bugs / impossible variants.

Usage:
    python scripts/make_trainset.py --out /tmp/irunner-trainset --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


TYPE_MATH = "math"
TYPE_DATA = "data_analysis"
TYPE_SEARCH = "search"
TYPE_CODING = "coding"

PRODUCTS = ["widget", "sprocket", "gizmo", "gadget", "flange", "bolt"]
REGIONS = ["west", "east", "north", "south"]
CITIES = ["Osaka", "Oslo", "Lisbon", "Toronto", "Nairobi", "Tallinn", "Busan", "Lima"]
INDUSTRIES = ["Robotics", "Data", "Foods", "Optics", "Motors", "Bio", "Systems", "Works"]
FIRST = ["Irina", "Sergei", "Alice", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Jonas", "Kira", "Maya", "Nikolai", "Olga", "Pavel", "Rosa", "Tanya", "Boris", "Quinn"]
LAST = ["Smith", "Garcia", "Petrov", "Novak", "Haddad", "Sato", "Wang", "Muller", "Ivanova", "Kowalski"]

# Local evaluation manifest, deliberately kept OUTSIDE the task directories.
# The harness never reads this file; the final competition set has no ground
# truth at all.
TRUTH: dict[str, dict] = {}


def write_task(
    root: Path,
    task_id: str,
    task_type: str,
    instruction: str,
    reward: int,
    check: str,
    answer: str | None,
    *,
    files: dict[str, str] | None = None,
) -> None:
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "instruction": instruction,
        "reward": reward,
        "check": check,
    }
    (task_dir / "task.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    TRUTH[task_id] = {
        "solvable": answer is not None,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------

def math_task(rng: random.Random, task_id: str, reward: int, impossible: bool) -> None:
    start = rng.randint(100, 5000)
    if rng.random() < 0.45:
        # split/double chain
        ops = []
        value = start
        for _ in range(rng.randint(2, 4)):
            if rng.random() < 0.55:
                k = rng.choice([2, 3, 4])
                ops.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down).")
                value //= k
            else:
                ops.append("Then the stock is doubled by merging with a sister warehouse.")
                value *= 2
        instruction = "A warehouse starts with {n} boxes. {ops} How many boxes does it have at the end? Compute it with the bash tool (python3 -c). Work only inside the current directory. When you are done, write ONLY the final answer to the file answer.txt.".format(
            n=start, ops=" ".join(ops)
        )
        answer = str(value)
    else:
        x = rng.randint(5, 60)
        d = rng.randint(2, 7)
        k = rng.randint(1, 4)
        m = rng.randint(10, 100)
        instruction = (
            f"A warehouse starts with {start} boxes. It ships {x} boxes per day for {d} days. "
            f"It receives {k} deliveries of {m} boxes each. How many boxes does it have at the end? "
            "Compute it with the bash tool (python3 -c). Work only inside the current directory. "
            "When you are done, write ONLY the final answer to the file answer.txt."
        )
        answer = str(start - x * d + k * m)

    if impossible:
        if rng.random() < 0.5:
            instruction = (
                "A warehouse starts with some boxes. It ships 10 boxes per day for 3 days. "
                "How many boxes does it have at the end? Compute it with python3."
            )
        else:
            instruction = (
                f"A warehouse starts with {start} boxes. It ships boxes per day for 3 days. "
                "How many boxes does it have at the end? Compute it with python3."
            )
        answer = None

    write_task(
        ROOT,
        task_id,
        TYPE_MATH,
        instruction,
        reward,
        "exact_int",
        answer,
    )


# ---------------------------------------------------------------------------
# data analysis
# ---------------------------------------------------------------------------

def generate_csv(rng: random.Random, rows: int, *, include_units: bool = True) -> str:
    fields = ["order_id", "region", "product", "month", "units", "price"]
    if not include_units:
        fields.remove("units")
    output = Path("/tmp/irunner_csv_tmp.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1, rows + 1):
            row = {
                "order_id": index,
                "region": rng.choice(REGIONS),
                "product": rng.choice(PRODUCTS),
                "month": rng.randint(1, 12),
                "units": rng.randint(1, 50),
                "price": rng.randint(5, 100),
            }
            writer.writerow({field: row[field] for field in fields})
    content = output.read_text(encoding="utf-8")
    output.unlink(missing_ok=True)
    return content


def data_task(rng: random.Random, task_id: str, reward: int, impossible: bool) -> None:
    rows = rng.choice([20, 300])
    product = rng.choice(PRODUCTS)
    start_month = rng.randint(1, 10)
    end_month = min(12, start_month + rng.randint(0, 2))
    csv_text = generate_csv(rng, rows, include_units=not impossible)
    # Compute the answer from the generated file.
    answer = 0
    for line in csv_text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        _, _, row_product, month, units, _ = parts
        if row_product == product and start_month <= int(month) <= end_month:
            answer += int(units)
    instruction = (
        f"The file sales.csv has columns order_id,region,product,month,units,price ({rows} rows). "
        f"How many units of product '{product}' were sold in months {start_month} through {end_month} inclusive? "
        "Use bash with python3 to compute it. Work only inside the current directory. "
        "When you are done, write ONLY the final answer to the file answer.txt."
    )
    files = {"sales.csv": csv_text}
    if impossible and rng.random() < 0.5:
        files = {}  # missing file
    write_task(
        ROOT,
        task_id,
        TYPE_DATA,
        instruction,
        reward,
        "exact_int",
        None if impossible else str(answer),
        files=files,
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_task(rng: random.Random, task_id: str, reward: int, impossible: bool) -> None:
    n_companies = rng.randint(4, 10)
    n_people = rng.randint(8, 30)
    companies = []
    for index in range(n_companies):
        name = f"{rng.choice(INDUSTRIES)} {rng.choice(['Labs', 'Data', 'Motors', 'Foods', 'Optics', 'Works'])} {index}"
        companies.append(
            {
                "name": name,
                "hq": f"{rng.choice(CITIES)}, {rng.choice(['Japan', 'Norway', 'Portugal', 'Canada', 'Kenya', 'Estonia'])}",
                "industry": rng.choice(INDUSTRIES),
                "founded": str(rng.randint(1950, 2020)),
            }
        )
    people = []
    for index in range(n_people):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)} {index}"
        people.append(
            {
                "name": name,
                "born": str(rng.randint(1950, 2005)),
                "employer": rng.choice(companies)["name"],
                "mentor": rng.choice(people)["name"] if people else "Alice Smith",
            }
        )

    files: dict[str, str] = {}
    for company in companies:
        slug = company["name"].lower().replace(" ", "_")
        files[f"kb/company_{slug}.md"] = (
            f"# {company['name']}\n\n"
            f"Headquarters: {company['hq']}\n"
            f"Industry: {company['industry']}\n"
            f"Founded: {company['founded']}\n"
        )
    for person in people:
        slug = person["name"].lower().replace(" ", "_")
        files[f"kb/person_{slug}.md"] = (
            f"# {person['name']}\n\n"
            f"Born: {person['born']}\n"
            f"Employer: {person['employer']}\n"
            f"Mentor: {person['mentor']}\n"
        )

    person = rng.choice(people)
    company = next(c for c in companies if c["name"] == person["employer"])
    question_kind = rng.choice(["born", "hq", "mentor", "industry"])
    if question_kind == "born":
        question = f"The directory kb/ contains a small knowledge base of markdown files. In which year was {person['name']} born?"
        answer = person["born"]
    elif question_kind == "hq":
        question = (
            f"The directory kb/ contains a small knowledge base of markdown files. "
            f"In which city is the headquarters of {person['name']}'s employer?"
        )
        answer = company["hq"].split(",")[0].strip()
    elif question_kind == "mentor":
        question = f"The directory kb/ contains a small knowledge base of markdown files. Who is the mentor of {person['name']}?"
        answer = person["mentor"]
    else:
        question = f"The directory kb/ contains a small knowledge base of markdown files. What industry is {company['name']} in?"
        answer = company["industry"]
    instruction = (
        question
        + " Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory. "
        "When you are done, write ONLY the final answer to the file answer.txt."
    )
    if impossible:
        instruction = (
            "The directory kb/ contains a small knowledge base of markdown files. "
            "In which year was Nonexistent Person born? Work only inside the current directory. "
            "When you are done, write ONLY the final answer to the file answer.txt."
        )
        answer = None
    write_task(ROOT, task_id, TYPE_SEARCH, instruction, reward, "exact_str", answer, files=files)


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------

BASE_CODE = '''def reverse_words(s):
    return ' '.join(reversed(s.split()))


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def average(xs):
    return sum(xs) / len(xs)


def first_n(xs, n):
    return xs[:n]


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)
'''

BASE_TEST = '''import unittest

import mylib as m


class T(unittest.TestCase):
    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)


if __name__ == '__main__':
    unittest.main()
'''


def coding_task(rng: random.Random, task_id: str, reward: int, impossible: bool) -> None:
    code = BASE_CODE
    test = BASE_TEST
    if impossible:
        test = test.replace(
            "    def test_count_even(self):\n"
            "        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)\n",
            "    def test_count_even(self):\n"
            "        assert False\n",
        )
    else:
        bugs = rng.sample(
            ["total", "reverse", "maximum", "count_even", "first_n"],
            k=rng.choice([1, 2, 2, 3]),
        )
        for bug in bugs:
            if bug == "total":
                code = code.replace("        s += x\n", "        s -= x\n")
            elif bug == "reverse":
                code = code.replace("' '.join(reversed(s.split()))", "' '.join(sorted(s.split()))")
            elif bug == "maximum":
                code = code.replace("if x > best:", "if x < best:")
            elif bug == "count_even":
                code = code.replace("x % 2 == 0", "x % 2 == 1")
            elif bug == "first_n":
                code = code.replace("return xs[:n]", "return xs[-n:]")
    instruction = (
        "Run `python3 test_mylib.py` in the current directory; some tests fail. "
        "Fix the bugs in mylib.py (do not edit the tests) until all tests pass. "
        "Work only inside the current directory. When you are done, write ONLY the word "
        "DONE to the file answer.txt in the current directory."
    )
    write_task(
        ROOT,
        task_id,
        TYPE_CODING,
        instruction,
        reward,
        "unittest",
        None if impossible else "DONE",
        files={"mylib.py": code, "test_mylib.py": test},
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

ROOT: Path


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--impossible-ratio", type=float, default=0.2)
    args = parser.parse_args()

    ROOT = Path(args.out).expanduser().resolve()
    if ROOT.exists():
        import shutil

        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n = args.n
    # 25% each, like the example set.
    types = [TYPE_MATH] * (n // 4) + [TYPE_DATA] * (n // 4) + [TYPE_SEARCH] * (n // 4) + [TYPE_CODING] * (n - 3 * (n // 4))
    rng.shuffle(types)
    impossible_count = round(n * args.impossible_ratio)
    impossible_indices = set(rng.sample(range(n), impossible_count))
    counters = {TYPE_MATH: 0, TYPE_DATA: 0, TYPE_SEARCH: 0, TYPE_CODING: 0}

    for index, task_type in enumerate(types):
        counters[task_type] += 1
        task_id = f"{task_type}_{counters[task_type]:04d}"
        reward = rng.randint(1, 5)
        impossible = index in impossible_indices
        if task_type == TYPE_MATH:
            math_task(rng, task_id, reward, impossible)
        elif task_type == TYPE_DATA:
            data_task(rng, task_id, reward, impossible)
        elif task_type == TYPE_SEARCH:
            search_task(rng, task_id, reward, impossible)
        else:
            coding_task(rng, task_id, reward, impossible)

    (ROOT / "_truth.json").write_text(
        json.dumps({"tasks": TRUTH}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(ROOT), "tasks": n, "impossible": impossible_count, "seed": args.seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
