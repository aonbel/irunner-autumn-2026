from __future__ import annotations

import json
import shutil
from pathlib import Path


def copy_tasks_without_expected(src: Path, dst: Path) -> None:
    """Copy a task tree and remove any legacy ``expected`` fields.

    The final competition tasks have no ground-truth field; the tests must
    therefore not depend on one.
    """
    shutil.copytree(src, dst)
    for task_json in dst.rglob("task.json"):
        data = json.loads(task_json.read_text(encoding="utf-8"))
        data.pop("expected", None)
        task_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
