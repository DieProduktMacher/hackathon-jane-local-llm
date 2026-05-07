"""
Extract ICD-10-CM coding rules from the tabular XML, with full ancestor inheritance.

Hierarchy: chapter → section → diag (nested)
Rules at each level are inherited downward and merged with the node's own rules.

Output: JSONL (one record per code) + CSV summary
"""
from __future__ import annotations

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _notes(element: ET.Element, tag: str) -> list[str]:
    """Return all <note> texts inside a direct child with the given tag name."""
    container = element.find(tag)
    if container is None:
        return []
    return [n.text.strip() for n in container.findall("note") if n.text and n.text.strip()]


def _extract(element: ET.Element) -> dict:
    """Pull every annotation type off a single element (no inheritance yet)."""
    return {
        "includes":       _notes(element, "includes"),
        "inclusion_terms": _notes(element, "inclusionTerm"),
        "excludes1":      _notes(element, "excludes1"),
        "excludes2":      _notes(element, "excludes2"),
        "code_first":     _notes(element, "codeFirst"),
        "use_additional": _notes(element, "useAdditionalCode"),
        "code_also":      _notes(element, "codeAlso"),
    }


ANNOTATION_KEYS = [
    "includes", "inclusion_terms", "excludes1", "excludes2",
    "code_first", "use_additional", "code_also",
]


def _merge(parent: dict, own: dict) -> dict:
    """Concatenate parent (inherited) lists with the node's own lists."""
    return {k: parent.get(k, []) + own.get(k, []) for k in ANNOTATION_KEYS}


_VALID_CODE = re.compile(r"^[A-Z]\d{2}(\.\d+)?$")


def is_valid_code(name: str) -> bool:
    """Return True for proper ICD-10-CM code strings (e.g. A00, A00.1, E08.32)."""
    return bool(_VALID_CODE.match(name.strip()))


# ── core traversal ────────────────────────────────────────────────────────────

def _traverse_diag(diag_el: ET.Element, inherited: dict, out: list) -> None:
    """Recursively walk a <diag> tree, emitting one record per valid code."""
    name_el = diag_el.find("name")
    desc_el = diag_el.find("desc")
    if name_el is None or desc_el is None:
        return

    code = (name_el.text or "").strip()
    desc = (desc_el.text or "").strip()

    if not is_valid_code(code):
        return

    own = _extract(diag_el)
    ctx = _merge(inherited, own)

    child_diags = diag_el.findall("diag")

    record = {
        "code":            code,
        "description":     desc,
        "is_leaf":         len(child_diags) == 0,
        "includes":        ctx["includes"],
        "inclusion_terms": ctx["inclusion_terms"],
        "excludes1":       ctx["excludes1"],
        "excludes2":       ctx["excludes2"],
        "code_first":      ctx["code_first"],
        "use_additional":  ctx["use_additional"],
        "code_also":       ctx["code_also"],
    }
    out.append(record)

    for child in child_diags:
        _traverse_diag(child, ctx, out)


def parse(xml_path: str | Path) -> list[dict]:
    """Parse the tabular XML and return a flat list of code records."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    records: list[dict] = []

    for chapter in root.findall("chapter"):
        ch_ctx = _extract(chapter)

        for section in chapter.findall("section"):
            sec_own = _extract(section)
            sec_ctx = _merge(ch_ctx, sec_own)

            for diag in section.findall("diag"):
                _traverse_diag(diag, sec_ctx, records)

    return records


# ── output helpers ────────────────────────────────────────────────────────────

def _list_to_str(lst: list[str]) -> str:
    """Pipe-separated string for CSV cells; empty string when list is empty."""
    return " | ".join(lst) if lst else ""


CSV_FIELDS = [
    "code", "description", "is_leaf",
    "includes", "inclusion_terms",
    "excludes1", "excludes2",
    "code_first", "use_additional", "code_also",
]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(records: list[dict], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            row = {k: v for k, v in rec.items() if k not in ANNOTATION_KEYS}
            for k in ANNOTATION_KEYS:
                row[k] = _list_to_str(rec[k])
            writer.writerow(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    xml_path = Path(
        sys.argv[1] if len(sys.argv) > 1
        else "icd10c-tabular-April-1-2026.xml"
    )
    if not xml_path.exists():
        sys.exit(f"File not found: {xml_path}")

    print(f"Parsing {xml_path} …", flush=True)
    records = parse(xml_path)

    total      = len(records)
    leaf_count = sum(1 for r in records if r["is_leaf"])
    print(f"  {total} codes extracted  ({leaf_count} leaf / {total - leaf_count} parent categories)")

    out_jsonl = xml_path.with_suffix(".jsonl")
    out_csv   = xml_path.with_suffix(".csv")

    write_jsonl(records, out_jsonl)
    print(f"  → {out_jsonl}")

    write_csv(records, out_csv)
    print(f"  → {out_csv}")

    # Quick sanity check: print a well-known code
    sample = next((r for r in records if r["code"] == "U07.1"), None)
    if sample:
        print("\nSample record (U07.1):")
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
