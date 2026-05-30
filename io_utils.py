from __future__ import annotations

import os
from typing import Iterable


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_joined_rows(
    path: str,
    header: str,
    rows: Iterable[str],
    *,
    chunk_size: int = 8192,
) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        chunk: list[str] = []
        for row in rows:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                f.write("".join(chunk))
                chunk.clear()
        if chunk:
            f.write("".join(chunk))


__all__ = ["ensure_parent_dir", "write_joined_rows"]
