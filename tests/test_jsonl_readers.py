"""JSONL readers must split on "\\n" only.

str.splitlines() also breaks on U+2028, U+0085 and friends, which json.dumps leaves raw inside strings.
Skill text contains them, so a splitlines() reader cuts records in half (this broke every cloud scoring job).
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_splitlines_breaks_json_records():
    line = json.dumps({"text": "a b"}, ensure_ascii=False)
    assert len(line.splitlines()) == 2
    assert json.loads(line.split("\n")[0]) == {"text": "a b"}


def test_no_splitlines_on_read_text_in_data_code():
    offenders = []
    for folder in ("bench", "cloud", "training"):
        for p in (ROOT / folder).rglob("*.py"):
            for n, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
                if re.search(r"read_text\([^)]*\)\.splitlines\(\)", line):
                    offenders.append(f"{p.relative_to(ROOT)}:{n}")
    assert not offenders, offenders
