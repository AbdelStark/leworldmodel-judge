"""JSONL/JSON file helpers — the only file I/O layer in the pipeline.

Every pipeline stage reads and writes through these three functions, so file
encoding (UTF-8), newline-per-row JSONL layout and the two-space JSON indent
stay uniform across artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as one compact JSON object per line, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines.

    A malformed line raises ``ValueError`` naming the file path and the
    1-based line number, so callers (the CLI in particular) can report which
    row broke instead of surfacing a bare ``json.JSONDecodeError``.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}, line {line_number}: {exc}") from exc
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a dict as indented JSON with a trailing newline, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
