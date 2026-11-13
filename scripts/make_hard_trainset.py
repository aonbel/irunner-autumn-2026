#!/usr/bin/env python3
"""Generate a deliberately hard development set.

The goal is to measure token consumption and LLM robustness on tasks that are
well outside the ultra-safe templates: multi-step arithmetic, 1000-row CSV
aggregations, multi-bug coding, long-document extraction, nested JSON, SQL
joins and three-hop KB questions.

Usage:
    python scripts/make_hard_trainset.py --out /tmp/irunner-hard
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from pathlib import Path


TRUTH: dict[str, dict] = {}


def write_task(
    root: Path,
    task_id: str,
    task_type: str,
    instruction: str,
    check: str,
    answer: str | None,
    files: dict[str, str] | None = None,
):
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task_type": task_type,
                "instruction": instruction,
                "reward": 2,
                "check": check,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    TRUTH[task_id] = {"solvable": answer is not None, "answer": answer}
    return task_dir


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------

def math_tasks(root: Path) -> None:
    cases = []
    # 1. Long warehouse chain.
    n = 2500
    n //= 4
    n *= 2
    n -= 25 * 6
    n += 3 * 40
    n //= 5
    n *= 3
    n += 2 * 17
    n -= 9 * 4
    cases.append((
        "A warehouse starts with 2500 boxes. It splits the stock equally into 4 warehouses and keeps one share (rounded down). "
        "Then the stock is doubled by merging with a sister warehouse. It ships 25 boxes per day for 6 days. "
        "It receives 3 deliveries of 40 boxes each. Then the stock is split equally into 5 warehouses and it keeps one share (rounded down). "
        "Then the stock is tripled. It receives 2 deliveries of 17 boxes each. It ships 9 boxes per day for 4 days. "
        "How many boxes does it have at the end?",
        str(n),
    ))
    # 2. Compound percentage + explicit original-stock shipment.
    value = 800
    value = value * 115 // 100
    value = value * 80 // 100
    value -= 40
    value += 3 * 25
    cases.append((
        "A warehouse starts with 800 boxes. Its stock increases by 15%, then decreases by 20% (each rounded down). "
        "It ships 40 boxes. It receives 3 deliveries of 25 boxes each. How many boxes does it have at the end?",
        str(value),
    ))
    # 3. Work-rate with integer liters.
    remaining = 1200 - (200 + 300) * 2
    cases.append((
        "A tank holds 1200 liters. Pipe A fills it at 200 liters per hour; pipe B fills it at 300 liters per hour. "
        "Both pipes run for 2 hours. How many liters are still needed to fill the tank?",
        str(remaining),
    ))
    # 4. Multi-entity transfer.
    a, b = 120, 80
    a -= 25
    b += 25
    b -= 10
    a += 10
    a += b // 2
    b -= b // 2
    cases.append((
        "Bin A has 120 boxes and bin B has 80 boxes. Move 25 boxes from A to B. Move 10 boxes from B to A. "
        "Then move half of B (rounded down) to A. How many boxes are in A at the end?",
        str(a),
    ))
    # 5. CRT-style small puzzle.
    cases.append((
        "Find the smallest integer N greater than 100 such that N mod 7 = 3 and N mod 5 = 2. "
        "What is N?",
        "122",
    ))

    for index, (instruction, answer) in enumerate(cases, 1):
        write_task(
            root,
            f"math_{index:04d}",
            "math",
            instruction + " Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt.",
            "exact_int",
            answer,
        )


# ---------------------------------------------------------------------------
# data analysis
# ---------------------------------------------------------------------------

def data_tasks(root: Path) -> None:
    rng = random.Random(17)
    products = ["widget", "sprocket", "gizmo", "gadget", "flange"]
    regions = ["west", "east", "north", "south"]
    rows = []
    for index in range(1, 1001):
        rows.append(
            {
                "order_id": index,
                "region": rng.choice(regions),
                "product": rng.choice(products),
                "month": rng.randint(1, 12),
                "units": rng.randint(1, 60),
                "price": rng.randint(5, 120),
                "customer": f"cust_{rng.randint(1, 40)}",
            }
        )
    header = "order_id,region,product,month,units,price,customer"
    csv_text = header + "\n" + "\n".join(
        ",".join(str(row[key]) for key in header.split(",")) for row in rows
    ) + "\n"

    def sum_units(region=None, product=None, lo=1, hi=12):
        return sum(
            row["units"]
            for row in rows
            if (region is None or row["region"] == region)
            and (product is None or row["product"] == product)
            and lo <= row["month"] <= hi
        )

    def revenue(region=None, lo=1, hi=12):
        return sum(
            row["units"] * row["price"]
            for row in rows
            if (region is None or row["region"] == region) and lo <= row["month"] <= hi
        )

    top_product = max(
        products,
        key=lambda p: sum_units(region="north", product=p, lo=4, hi=10),
    )
    cases = [
        ("What is the total revenue in region west for months 1 through 6?", str(revenue("west", 1, 6))),
        ("Which product sold the most units in region north in months 4 through 10?", top_product),
        ("How many orders have units greater than 30 and price less than 50 in region south?", str(sum(1 for row in rows if row["region"] == "south" and row["units"] > 30 and row["price"] < 50))),
        ("How many distinct customers are there in region east?", str(len({row["customer"] for row in rows if row["region"] == "east"}))),
        ("How many units of product widget were sold in months 7 through 9?", str(sum_units(product="widget", lo=7, hi=9))),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The file sales.csv has columns order_id,region,product,month,units,price,customer (1000 rows). "
            + question
            + " Use bash with python3 to compute it. Work only inside the current directory. "
            "When you are done, write ONLY the final answer to answer.txt."
        )
        check = "exact_int" if answer.replace("-", "", 1).isdigit() else "exact_str"
        write_task(root, f"data_analysis_{index:04d}", "data_analysis", instruction, check, answer, {"sales.csv": csv_text})


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------

def coding_tasks(root: Path) -> None:
    big_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_mean(self):
        self.assertEqual(m.mean([2, 4, 6]), 4)

    def test_median(self):
        self.assertEqual(m.median([3, 1, 2, 5, 4]), 3)

    def test_mode(self):
        self.assertEqual(m.mode([1, 2, 2, 3]), 2)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)


if __name__ == '__main__':
    unittest.main()
"""
    big_code = """def total(xs):
    return 0


def mean(xs):
    return sum(xs) / (len(xs) + 1)


def median(xs):
    return 0


def first_n(xs, n):
    return xs[-n:]


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 1)
"""
    write_task(
        root,
        "coding_0001",
        "coding",
        "Run python3 test_mylib.py; fix all bugs in mylib.py without editing the tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": big_code, "test_mylib.py": big_test},
    )

    helpers = """def clamp(x, lo, hi):
    return min(lo, x)


def normalize(xs):
    top = max(xs)
    return [x / top for x in xs]
"""
    multi_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(m.clamp(15, 0, 10), 10)

    def test_normalize(self):
        self.assertEqual(m.normalize([2, 4]), [0.5, 1.0])


if __name__ == '__main__':
    unittest.main()
"""
    multi_code = """from helpers import clamp, normalize


def bounded(x, lo, hi):
    return clamp(x, lo, hi)


def scaled(xs):
    return normalize(xs)
"""
    write_task(
        root,
        "coding_0002",
        "coding",
        "Run python3 test_mylib.py; fix the bugs in mylib.py and helpers.py without editing the tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": multi_code, "helpers.py": helpers, "test_mylib.py": multi_test},
    )

    algo_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(m.merge_intervals([[1, 3], [2, 6], [8, 10]]), [[1, 6], [8, 10]])

    def test_top_k(self):
        self.assertEqual(m.top_k([4, 1, 9, 3], 2), [9, 4])

    def test_word_freq(self):
        self.assertEqual(m.word_freq('a b a'), {'a': 2, 'b': 1})


if __name__ == '__main__':
    unittest.main()
"""
    algo_code = """def merge_intervals(intervals):
    return intervals


def top_k(xs, k):
    return xs[:k]


def word_freq(text):
    return {}
"""
    write_task(
        root,
        "coding_0003",
        "coding",
        "Run python3 test_mylib.py; implement/fix the functions in mylib.py without editing the tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": algo_code, "test_mylib.py": algo_test},
    )

    hard_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_running_sum(self):
        self.assertEqual(m.running_sum([1, 2, 3]), [1, 3, 6])

    def test_second_largest(self):
        self.assertEqual(m.second_largest([4, 9, 2, 9]), 4)

    def test_is_palindrome(self):
        self.assertTrue(m.is_palindrome('A man, a plan, a canal: Panama'))

    def test_fizz(self):
        self.assertEqual(m.fizz(5), ['1', '2', 'Fizz', '4', 'Buzz'])


if __name__ == '__main__':
    unittest.main()
"""
    hard_code = """def running_sum(xs):
    return xs


def second_largest(xs):
    return max(xs)


def is_palindrome(text):
    return False


def fizz(n):
    return [str(i) for i in range(1, n + 1)]
"""
    write_task(
        root,
        "coding_0004",
        "coding",
        "Run python3 test_mylib.py; implement/fix the functions in mylib.py without editing the tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": hard_code, "test_mylib.py": hard_test},
    )


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

def text_tasks(root: Path) -> None:
    projects = []
    companies = {
        "Dune Robotics": ("Hanoi", "Robotics"),
        "Fjord Data": ("Osaka", "Data"),
        "Granite AI": ("Oslo", "AI"),
        "Cobalt Works": ("Tallinn", "Manufacturing"),
        "Iris Optics": ("Busan", "Optics"),
    }
    for index, (company, (hq, industry)) in enumerate(companies.items()):
        projects.append(
            f"Project {['Zephyr', 'Atlas', 'Orion', 'Nova', 'Helix'][index]}\n"
            f"Owner: {company}\n"
            f"Status: {'blocked' if index == 1 else 'active'}\n"
            f"Budget: {10000 + index * 7500}\n"
            f"Deadline: 2026-0{index + 1}-15\n"
        )
    notes = "\n".join(projects)
    notes += "\nCompany details:\n" + "\n".join(
        f"{company}: headquarters {hq}; industry {industry}.\n" for company, (hq, industry) in companies.items()
    )
    # Make the document long enough to matter.
    notes += "\n" + "\n".join(
        f"Archive note {i}: the {i}th review mentions project {['Zephyr','Atlas','Orion','Nova','Helix'][i % 5]} "
        f"and a budget of {10000 + (i % 5) * 7500}."
        for i in range(1, 120)
    )
    cases = [
        ("Who owns project Zephyr?", "Dune Robotics"),
        ("What is the headquarters city of the company that owns project Atlas?", "Osaka"),
        ("What is the budget of the project with status blocked?", "17500"),
        ("What is the deadline of project Helix?", "2026-05-15"),
        ("What is the favorite color of the owner of project Zephyr?", None),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The file notes.txt contains a long project report. "
            + question
            + " Use the text file. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        write_task(root, f"text_{index:04d}", "text", instruction, "exact_str", answer, {"notes.txt": notes})


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def json_tasks(root: Path) -> None:
    data = {
        "company": "Orion Labs",
        "departments": [
            {
                "name": "Engineering",
                "employees": [
                    {"name": "A", "active": True, "salary": 100, "skills": ["python", "sql"]},
                    {"name": "B", "active": False, "salary": 200, "skills": ["python"]},
                ],
            },
            {
                "name": "Sales",
                "employees": [
                    {"name": "C", "active": True, "salary": 300, "skills": ["excel"]},
                    {"name": "D", "active": True, "salary": 400, "skills": ["python", "excel"]},
                ],
            },
        ],
        "projects": [
            {"name": "Zephyr", "budget": 5000, "department": "Engineering"},
            {"name": "Atlas", "budget": 12000, "department": "Sales"},
        ],
    }
    text = json.dumps(data, indent=2)
    cases = [
        ("What is the total salary of active employees in Engineering?", "100"),
        ("How many active employees have the python skill?", "2"),
        ("Which project has the largest budget?", "Atlas"),
        ("What is the total budget of all projects?", "17000"),
        ("What is the phone number of employee A?", None),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The file data.json contains nested company data. "
            + question
            + " Use python3 with the json module. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        write_task(root, f"json_{index:04d}", "json", instruction, "exact_str" if answer is not None and not answer.isdigit() else "exact_int", answer, {"data.json": text})


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

def sql_tasks(root: Path) -> None:
    task_dir = write_task(
        root,
        "sql_0001",
        "sql",
        "The file data.db contains customers and orders. What is the total amount of all paid orders? "
        "Use python3 with sqlite3. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt.",
        "exact_int",
        "500",
    )
    db_path = task_dir / "workspace" / "data.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER, status TEXT, year INTEGER);
        INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com');
        INSERT INTO customers VALUES (2, 'Bob', 'bob@example.com');
        INSERT INTO orders VALUES (1, 1, 200, 'paid', 2024);
        INSERT INTO orders VALUES (2, 2, 300, 'paid', 2024);
        INSERT INTO orders VALUES (3, 1, 100, 'cancelled', 2023);
        """
    )
    conn.commit()
    conn.close()

    task_dir = write_task(
        root,
        "sql_0002",
        "sql",
        "The file data.db contains customers and orders. Which customer has the highest total amount of paid orders? "
        "Use python3 with sqlite3. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt.",
        "exact_str",
        "Bob",
    )
    db_path = task_dir / "workspace" / "data.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER, status TEXT, year INTEGER);
        INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com');
        INSERT INTO customers VALUES (2, 'Bob', 'bob@example.com');
        INSERT INTO orders VALUES (1, 1, 200, 'paid', 2024);
        INSERT INTO orders VALUES (2, 2, 300, 'paid', 2024);
        INSERT INTO orders VALUES (3, 1, 100, 'cancelled', 2023);
        """
    )
    conn.commit()
    conn.close()

    task_dir = write_task(
        root,
        "sql_0003",
        "sql",
        "The file data.db contains customers and orders. How many order items have quantity greater than 5? "
        "Use python3 with sqlite3. Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt.",
        "exact_int",
        None,
    )
    db_path = task_dir / "workspace" / "data.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER, status TEXT, year INTEGER);
        INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com');
        INSERT INTO orders VALUES (1, 1, 200, 'paid', 2024);
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search_tasks(root: Path) -> None:
    companies = {}
    for index in range(1, 31):
        name = f"Company {index:02d}"
        companies[name] = {
            "hq": ["Osaka", "Hanoi", "Oslo", "Tallinn", "Busan", "Lima"][index % 6],
            "industry": ["Data", "Robotics", "Optics", "Foods", "Motors"][index % 5],
            "parent": f"Parent {index % 7}",
        }
    people = {}
    for index in range(1, 61):
        name = f"Person {index:02d}"
        people[name] = {
            "born": 1950 + index % 55,
            "employer": f"Company {index % 30 + 1:02d}",
            "mentor": f"Person {(index * 7) % 60 + 1:02d}",
        }
    files: dict[str, str] = {}
    for name, company in companies.items():
        slug = name.lower().replace(" ", "_")
        files[f"kb/company_{slug}.md"] = (
            f"# {name}\n\nHeadquarters: {company['hq']}\nIndustry: {company['industry']}\nParent: {company['parent']}\n"
        )
    for name, person in people.items():
        slug = name.lower().replace(" ", "_")
        files[f"kb/person_{slug}.md"] = (
            f"# {name}\n\nBorn: {person['born']}\nEmployer: {person['employer']}\nMentor: {person['mentor']}\n"
        )

    target_person = people["Person 01"]
    employer = companies[target_person["employer"]]
    parent = employer["parent"]
    # Parent companies have their own HQ/industry entries so the LLM can
    # resolve a third hop.
    for index in range(7):
        parent_name = f"Parent {index}"
        files[f"kb/company_parent_{index}.md"] = (
            f"# {parent_name}\n\nHeadquarters: {['Osaka','Hanoi','Oslo','Tallinn','Busan','Lima','Lima'][index]}\n"
            f"Industry: {['Data','Robotics','Optics','Foods','Motors','Data','AI'][index]}\n"
        )

    cases = [
        (
            "In which city is the headquarters of the parent company of Person 01's employer?",
            ["Osaka", "Hanoi", "Oslo", "Tallinn", "Busan", "Lima", "Lima"][int(parent.split()[-1])],
        ),
        ("Which company owns the employer of Person 02?", companies[people["Person 02"]["employer"]]["parent"]),
        ("What industry is the parent company of Person 03's employer?", ["Data", "Robotics", "Optics", "Foods", "Motors", "Data", "AI"][int(companies[people["Person 03"]["employer"]]["parent"].split()[-1])]),
        ("Who is the CEO of Company 01?", None),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The directory kb/ contains a small knowledge base of markdown files. "
            + question
            + " Work only inside the current directory. When you are done, write ONLY the final answer to answer.txt."
        )
        write_task(root, f"search_{index:04d}", "search", instruction, "exact_str", answer, files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out).expanduser().resolve()
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True)
    math_tasks(root)
    data_tasks(root)
    coding_tasks(root)
    text_tasks(root)
    json_tasks(root)
    sql_tasks(root)
    search_tasks(root)
    (root / "_truth.json").write_text(json.dumps({"tasks": TRUTH}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(root), "tasks": len(TRUTH)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
