#!/usr/bin/env python3
"""Generate an order-of-magnitude harder development set.

Requires openpyxl/python-docx (already in the runtime image).  Run it inside
the image:

    docker run --rm -v "$PWD:/app" -v /tmp/ultrahard:/tmp/ultrahard \
      --entrypoint python irunner:latest /app/scripts/make_ultrahard_trainset.py \
      --out /tmp/ultrahard
"""

from __future__ import annotations

import argparse
import csv
import re
import json
import random
import sqlite3
from pathlib import Path


TRUTH: dict[str, dict] = {}


def write_task(root: Path, task_id: str, task_type: str, instruction: str, check: str, answer: str | None, files: dict[str, str] | None = None) -> Path:
    task_dir = root / task_id
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps(
            {"task_id": task_id, "task_type": task_type, "instruction": instruction, "reward": 2, "check": check},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    TRUTH[task_id] = {"solvable": answer is not None, "answer": answer}
    return task_dir


# ---------------------------------------------------------------------------
# investigation
# ---------------------------------------------------------------------------

def investigation_tasks(root: Path) -> None:
    rng = random.Random(23)
    lines = []
    for index in range(1, 12001):
        status = rng.choice([200, 200, 200, 404, 500])
        service = rng.choice(["auth", "cart", "search", "billing"])
        latency = rng.randint(1, 900)
        size = rng.randint(100, 5000)
        line = f"ts=2026-01-{index % 28 + 1:02d}T12:00:00 status={status} service={service} latency_ms={latency} bytes={size} req=req-{index:05d}"
        if index == 7777:
            line += " session=s-abc user=alice code=ZEBRA-42"
        if index == 9999:
            line += " session=s-abc user=alice"
        lines.append(line)
    # Make the maximum latency unique so "which request" has one answer.
    if len(lines) > 12344:
        lines[12344] = re.sub(r"latency_ms=\d+", "latency_ms=999", lines[12344])
    log_text = "\n".join(lines) + "\n"
    error_auth = sum(1 for line in lines if "ERROR" in line or ("status=500" in line and "service=auth" in line))
    max_latency_line = max(lines, key=lambda line: int(line.split("latency_ms=")[1].split()[0]))
    max_req = max_latency_line.split("req=")[1].split()[0]
    bytes_500 = sum(int(line.split("bytes=")[1].split()[0]) for line in lines if "status=500" in line)

    write_task(
        root,
        "investigation_0001",
        "investigation",
        "The file service.log contains HTTP logs. What is the access code for user alice? "
        "Use bash/grep to investigate. Write only the final answer to answer.txt.",
        "exact_str",
        "ZEBRA-42",
        {"service.log": log_text},
    )
    write_task(
        root,
        "investigation_0002",
        "investigation",
        "The file service.log contains HTTP logs. How many status=500 requests were served by service auth? "
        "Use bash/grep to compute. Write only the final answer to answer.txt.",
        "exact_int",
        str(error_auth),
        {"service.log": log_text},
    )
    write_task(
        root,
        "investigation_0003",
        "investigation",
        "The file service.log contains HTTP logs. Which request id has the highest latency_ms? "
        "Use bash/python to investigate. Write only the final answer to answer.txt.",
        "exact_str",
        max_req,
        {"service.log": log_text},
    )
    write_task(
        root,
        "investigation_0004",
        "investigation",
        "The file service.log contains HTTP logs. What is the total bytes served for status=500 requests? "
        "Use bash/python to compute. Write only the final answer to answer.txt.",
        "exact_int",
        str(bytes_500),
        {"service.log": log_text},
    )


# ---------------------------------------------------------------------------
# multi-CSV
# ---------------------------------------------------------------------------

def multi_csv_tasks(root: Path) -> None:
    rng = random.Random(31)
    customers = [(index, f"customer_{index:04d}", rng.choice(["west", "east", "north", "south"]), rng.choice(["gold", "silver", "bronze"])) for index in range(1, 2001)]
    orders = []
    for index in range(1, 10001):
        orders.append((index, rng.randint(1, 2000), rng.randint(10, 900), rng.choice(["paid", "paid", "cancelled"]), rng.choice([2023, 2024, 2025])))
    items = []
    for index in range(1, 30001):
        items.append((index, rng.randint(1, 10000), rng.choice(["widget", "gadget", "gizmo", "flange"]), rng.randint(1, 20), rng.randint(5, 100)))

    customer_text = "customer_id,name,region,tier\n" + "\n".join(",".join(map(str, row)) for row in customers) + "\n"
    order_text = "order_id,customer_id,amount,status,year\n" + "\n".join(",".join(map(str, row)) for row in orders) + "\n"
    item_text = "item_id,order_id,product,qty,price\n" + "\n".join(",".join(map(str, row)) for row in items) + "\n"

    paid_2024 = sum(amount for _, _, amount, status, year in orders if status == "paid" and year == 2024)
    totals: dict[int, int] = {}
    for _, customer_id, amount, status, _ in orders:
        if status == "paid":
            totals[customer_id] = totals.get(customer_id, 0) + amount
    top_customer_id = max(totals, key=totals.get)
    top_customer = next(name for cid, name, _, _ in customers if cid == top_customer_id)
    widget_qty = sum(qty for _, _, product, qty, _ in items if product == "widget")
    cancelled = sum(1 for _, _, _, status, _ in orders if status == "cancelled")

    files = {"customers.csv": customer_text, "orders.csv": order_text, "items.csv": item_text}
    cases = [
        ("What is the total amount of paid orders in 2024?", str(paid_2024)),
        ("Which customer has the highest total paid amount?", top_customer),
        ("What is the total quantity of product widget sold?", str(widget_qty)),
        ("How many cancelled orders are there?", str(cancelled)),
    ]
    for index, (question, answer) in enumerate(cases, 1):
        instruction = (
            "The files customers.csv, orders.csv and items.csv contain related tables. "
            + question
            + " Use bash with python3 to join/compute. Work only inside the current directory. "
            "When you are done, write ONLY the final answer to answer.txt."
        )
        check = "exact_int" if answer.replace("-", "", 1).isdigit() else "exact_str"
        write_task(root, f"multi_csv_{index:04d}", "multi_csv", instruction, check, answer, files)


# ---------------------------------------------------------------------------
# spreadsheet
# ---------------------------------------------------------------------------

def spreadsheet_tasks(root: Path) -> None:
    import openpyxl

    rng = random.Random(41)
    rows = []
    for index in range(1, 3001):
        rows.append((index, rng.choice(["west", "east", "north", "south"]), rng.choice(["widget", "gadget", "gizmo"]), rng.randint(1, 50), rng.randint(5, 100)))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["id", "region", "product", "units", "price"])
    for row in rows:
        ws.append(row)
    ws2 = wb.create_sheet("Targets")
    ws2.append(["region", "target"])
    for region in ["west", "east", "north", "south"]:
        ws2.append([region, 10000])
    path = root / "spreadsheet_0001" / "workspace" / "sales.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    (root / "spreadsheet_0001" / "task.json").write_text(json.dumps({
        "task_id": "spreadsheet_0001",
        "task_type": "spreadsheet",
        "instruction": "The file sales.xlsx contains a Sales sheet. What is the total units in region west? Use python3 with openpyxl. Write only the final answer to answer.txt.",
        "reward": 2,
        "check": "exact_int",
    }), encoding="utf-8")
    TRUTH["spreadsheet_0001"] = {"solvable": True, "answer": str(sum(row[3] for row in rows if row[1] == "west"))}

    wb2 = openpyxl.Workbook()
    ws = wb2.active
    ws.title = "Sales"
    ws.append(["id", "region", "product", "units", "price"])
    for row in rows:
        ws.append(row)
    path = root / "spreadsheet_0002" / "workspace" / "sales.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb2.save(path)
    totals: dict[str, int] = {}
    for _, _, product, units, price in rows:
        totals[product] = totals.get(product, 0) + units * price
    (root / "spreadsheet_0002" / "task.json").write_text(json.dumps({
        "task_id": "spreadsheet_0002",
        "task_type": "spreadsheet",
        "instruction": "The file sales.xlsx contains a Sales sheet. Which product has the highest total revenue (units * price)? Use python3 with openpyxl. Write only the final answer to answer.txt.",
        "reward": 2,
        "check": "exact_str",
    }), encoding="utf-8")
    TRUTH["spreadsheet_0002"] = {"solvable": True, "answer": max(totals, key=totals.get)}

    wb3 = openpyxl.Workbook()
    ws = wb3.active
    ws.title = "Sales"
    ws.append(["id", "region", "product", "units", "price"])
    for row in rows:
        ws.append(row)
    path = root / "spreadsheet_0003" / "workspace" / "sales.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb3.save(path)
    (root / "spreadsheet_0003" / "task.json").write_text(json.dumps({
        "task_id": "spreadsheet_0003",
        "task_type": "spreadsheet",
        "instruction": "The file sales.xlsx contains a Sales sheet. How many rows have units greater than 30? Use python3 with openpyxl. Write only the final answer to answer.txt.",
        "reward": 2,
        "check": "exact_int",
    }), encoding="utf-8")
    TRUTH["spreadsheet_0003"] = {"solvable": True, "answer": str(sum(1 for row in rows if row[3] > 30))}


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

def document_tasks(root: Path) -> None:
    import docx

    facts = [
        "Project Zephyr is owned by Dune Robotics.",
        "Dune Robotics is headquartered in Hanoi.",
        "The access code for Project Zephyr is 7731.",
        "Project Atlas is owned by Fjord Data.",
        "Fjord Data is headquartered in Osaka.",
        "The budget of Project Atlas is 42000.",
    ]
    filler = [f"Archive paragraph {i}: routine status notes and meeting minutes." for i in range(1, 600)]
    doc = docx.Document()
    for paragraph in facts[:3] + filler[:250] + facts[3:] + filler[250:]:
        doc.add_paragraph(paragraph)
    task_dir = root / "document_0001"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    doc.save(workspace / "report.docx")
    (task_dir / "task.json").write_text(json.dumps({
        "task_id": "document_0001",
        "task_type": "document",
        "instruction": "The file report.docx contains a long report. What is the access code for Project Zephyr? Use the document. Write only the final answer to answer.txt.",
        "reward": 2,
        "check": "exact_str",
    }), encoding="utf-8")
    TRUTH["document_0001"] = {"solvable": True, "answer": "7731"}

    doc2 = docx.Document()
    for paragraph in facts + filler:
        doc2.add_paragraph(paragraph)
    task_dir = root / "document_0002"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    doc2.save(workspace / "report.docx")
    (task_dir / "task.json").write_text(json.dumps({
        "task_id": "document_0002",
        "task_type": "document",
        "instruction": "The file report.docx contains a long report. In which city is the company that owns Project Atlas headquartered? Use the document. Write only the final answer to answer.txt.",
        "reward": 2,
        "check": "exact_str",
    }), encoding="utf-8")
    TRUTH["document_0002"] = {"solvable": True, "answer": "Osaka"}

    doc3 = docx.Document()
    for paragraph in facts + filler:
        doc3.add_paragraph(paragraph)
    task_dir = root / "document_0003"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    doc3.save(workspace / "report.docx")
    (task_dir / "task.json").write_text(json.dumps({
        "task_id": "document_0003",
        "task_type": "document",
        "instruction": "The file report.docx contains a long report. What is the favorite color of the owner of Project Zephyr? Use the document. Write only the final answer to answer.txt.",
        "reward": 2,
        "check": "exact_str",
    }), encoding="utf-8")
    TRUTH["document_0003"] = {"solvable": False, "answer": None}


# ---------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------

def logic_tasks(root: Path) -> None:
    cases = [
        (
            "Five houses in a row are painted different colors. The green house is immediately to the right of the ivory house. "
            "The Norwegian lives in the first house. The Englishman lives in the red house. The Spaniard owns the dog. "
            "The Ukrainian drinks tea. The green house owner drinks coffee. The Old Gold smoker owns snails. "
            "Kools are smoked in the yellow house. Milk is drunk in the middle house. The Lucky Strike smoker drinks orange juice. "
            "The Japanese smokes Parliaments. The Norwegian lives next to the blue house. "
            "The Blend smoker lives next to the cat owner. The Kools smoker lives next to the horse owner. "
            "The Blend smoker lives next to the water drinker. Who drinks water?",
            "Norwegian",
        ),
        (
            "Three people A, B and C are either knights (always tell the truth) or knaves (always lie). "
            "A says: 'B is a knave.' B says: 'A and C are of the same type.' What is C?",
            "knave",
        ),
        (
            "Four meetings must be scheduled at 9, 10, 11 and 12. Alice cannot do 9 or 10. Bob cannot do 12. "
            "Carol must be before Bob. Dave must be after Alice. Which person is at 9?",
            "Carol",
        ),
        (
            "A box contains 3 red, 4 blue and 5 green balls. Balls are drawn without replacement. "
            "What is the minimum number that must be drawn to guarantee at least 2 balls of the same color?",
            "4",
        ),
    ]
    for index, (instruction, answer) in enumerate(cases, 1):
        write_task(
            root,
            f"logic_{index:04d}",
            "logic",
            instruction + " Write only the final answer to answer.txt.",
            "exact_str",
            answer,
        )


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------

def coding_tasks(root: Path) -> None:
    test = """import unittest

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

    def test_running(self):
        self.assertEqual(m.running_sum([1, 2, 3]), [1, 3, 6])

    def test_second(self):
        self.assertEqual(m.second_largest([4, 9, 2, 9]), 4)


if __name__ == '__main__':
    unittest.main()
"""
    code = """def total(xs):
    return 0


def mean(xs):
    return sum(xs) / (len(xs) + 1)


def median(xs):
    return 0


def mode(xs):
    return 0


def running_sum(xs):
    return xs


def second_largest(xs):
    return max(xs)
"""
    write_task(
        root,
        "coding_0001",
        "coding",
        "Run python3 test_mylib.py; fix every bug in mylib.py without editing tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": code, "test_mylib.py": test},
    )

    algo_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_lru(self):
        c = m.LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        self.assertEqual(c.get(1), 1)
        c.put(3, 3)
        self.assertEqual(c.get(2), -1)

    def test_merge(self):
        self.assertEqual(m.merge_intervals([[1, 3], [2, 6], [8, 10]]), [[1, 6], [8, 10]])

    def test_dijkstra(self):
        graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
        self.assertEqual(m.dijkstra(graph, 0, 3), 4)


if __name__ == '__main__':
    unittest.main()
"""
    algo_code = """class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity

    def get(self, key):
        return -1

    def put(self, key, value):
        pass


def merge_intervals(intervals):
    return intervals


def dijkstra(graph, start, end):
    return 0
"""
    write_task(
        root,
        "coding_0002",
        "coding",
        "Run python3 test_mylib.py; implement the algorithms in mylib.py without editing tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": algo_code, "test_mylib.py": algo_test},
    )

    multi_test = """import unittest

import mylib as m
import helpers as h


class T(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(h.clamp(15, 0, 10), 10)

    def test_normalize(self):
        self.assertEqual(h.normalize([2, 4]), [0.5, 1.0])

    def test_parse(self):
        self.assertEqual(m.parse('a=1;b=2'), {'a': '1', 'b': '2'})
"""
    multi_code = """from helpers import clamp, normalize


def parse(text):
    return {}
"""
    helpers = """def clamp(x, lo, hi):
    return min(lo, x)


def normalize(xs):
    top = max(xs)
    return [x / top for x in xs]
"""
    write_task(
        root,
        "coding_0003",
        "coding",
        "Run python3 test_mylib.py; fix mylib.py and helpers.py without editing tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": multi_code, "helpers.py": helpers, "test_mylib.py": multi_test},
    )

    hard_test = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_fizz(self):
        self.assertEqual(m.fizz(5), ['1', '2', 'Fizz', '4', 'Buzz'])

    def test_palindrome(self):
        self.assertTrue(m.is_palindrome('A man, a plan, a canal: Panama'))

    def test_word_freq(self):
        self.assertEqual(m.word_freq('a b a'), {'a': 2, 'b': 1})

    def test_top_k(self):
        self.assertEqual(m.top_k([4, 1, 9, 3], 2), [9, 4])
"""
    hard_code = """def fizz(n):
    return [str(i) for i in range(1, n + 1)]


def is_palindrome(text):
    return False


def word_freq(text):
    return {}


def top_k(xs, k):
    return xs[:k]
"""
    write_task(
        root,
        "coding_0004",
        "coding",
        "Run python3 test_mylib.py; implement/fix the functions in mylib.py without editing tests. Write DONE to answer.txt.",
        "unittest",
        "DONE",
        {"mylib.py": hard_code, "test_mylib.py": hard_test},
    )


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

def sql_tasks(root: Path) -> None:
    rng = random.Random(51)
    task_dir = write_task(
        root,
        "sql_0001",
        "sql",
        "The file data.db contains customers, orders and items. What is the total quantity of product widget sold in paid orders? "
        "Use python3 with sqlite3. Write only the final answer to answer.txt.",
        "exact_int",
        None,
    )
    db_path = task_dir / "workspace" / "data.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT, year INTEGER);
        CREATE TABLE items (id INTEGER PRIMARY KEY, order_id INTEGER, product TEXT, qty INTEGER, price INTEGER);
        """
    )
    customers = [(index, f"c{index}", rng.choice(["west", "east"])) for index in range(1, 501)]
    orders = [(index, rng.randint(1, 500), rng.choice(["paid", "paid", "cancelled"]), rng.choice([2023, 2024])) for index in range(1, 5001)]
    items = [(index, rng.randint(1, 5000), rng.choice(["widget", "gadget"]), rng.randint(1, 10), rng.randint(5, 50)) for index in range(1, 20001)]
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", customers)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    conn.executemany("INSERT INTO items VALUES (?,?,?,?,?)", items)
    conn.commit()
    conn.close()
    paid_ids = {oid for oid, _, status, _ in orders if status == "paid"}
    total = sum(qty for _, order_id, product, qty, _ in items if product == "widget" and order_id in paid_ids)
    TRUTH["sql_0001"]["answer"] = str(total)
    TRUTH["sql_0001"]["solvable"] = True

    task_dir = write_task(
        root,
        "sql_0002",
        "sql",
        "The file data.db contains customers, orders and items. Which customer has the highest total paid amount (sum of qty*price)? "
        "Use python3 with sqlite3. Write only the final answer to answer.txt.",
        "exact_str",
        None,
    )
    db_path = task_dir / "workspace" / "data.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT);
        CREATE TABLE items (id INTEGER PRIMARY KEY, order_id INTEGER, qty INTEGER, price INTEGER);
        INSERT INTO customers VALUES (1, 'Alice');
        INSERT INTO customers VALUES (2, 'Bob');
        INSERT INTO orders VALUES (1, 1, 'paid');
        INSERT INTO orders VALUES (2, 2, 'paid');
        INSERT INTO items VALUES (1, 1, 2, 100);
        INSERT INTO items VALUES (2, 2, 1, 50);
        """
    )
    conn.commit()
    conn.close()
    TRUTH["sql_0002"]["answer"] = "Alice"
    TRUTH["sql_0002"]["solvable"] = True

    task_dir = write_task(
        root,
        "sql_0003",
        "sql",
        "The file data.db contains customers and orders. What is the phone number of customer Alice? "
        "Use python3 with sqlite3. Write only the final answer to answer.txt.",
        "exact_str",
        None,
    )
    db_path = task_dir / "workspace" / "data.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com');
        """
    )
    conn.commit()
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out).expanduser().resolve()
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True)
    investigation_tasks(root)
    multi_csv_tasks(root)
    spreadsheet_tasks(root)
    document_tasks(root)
    logic_tasks(root)
    coding_tasks(root)
    sql_tasks(root)
    (root / "_truth.json").write_text(json.dumps({"tasks": TRUTH}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(root), "tasks": len(TRUTH)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
