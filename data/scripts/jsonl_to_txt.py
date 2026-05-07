"""Convert ICD-10-CM JSONL to a clean flat text file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# (label, getter, add_period)
FIELD_MAP = [
    ("Code",                lambda r: [r["code"]],                          False),
    ("Description",         lambda r: [r["description"]],                   False),
    ("Includes",            lambda r: r["includes"] + r["inclusion_terms"], True),
    ("Excludes1",           lambda r: r["excludes1"],                       True),
    ("Excludes2",           lambda r: r["excludes2"],                       True),
    ("Code First",          lambda r: r["code_first"],                      True),
    ("Use Additional Code", lambda r: r["use_additional"],                  True),
    ("Code Also",           lambda r: r["code_also"],                       True),
]


def record_to_text(record: dict) -> str:
    lines = []
    for label, getter, add_period in FIELD_MAP:
        values = getter(record)
        if not values:
            continue
        joined = ", ".join(v.rstrip(".") for v in values)
        suffix = "." if add_period else ""
        lines.append(f"{label}: {joined}{suffix}")
    return "\n".join(lines)


def convert(src: Path, dst: Path) -> None:
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not first:
                fout.write("---\n")
            fout.write(record_to_text(record) + "\n")
            first = False


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/icd10c-tabular-April-1-2026.jsonl")
    dst = src.with_suffix(".txt")

    if not src.exists():
        sys.exit(f"File not found: {src}")

    print(f"Converting {src} → {dst} …")
    convert(src, dst)
    print(f"Done. Lines written: {sum(1 for _ in open(dst))}")


if __name__ == "__main__":
    main()
