from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
VARIANT_IO = importlib.import_module(f"{PKG}.variant_io")

iter_variant_records_for_prefix = VARIANT_IO.iter_variant_records_for_prefix


def test_iter_pvar_records_reports_short_data_line(tmp_path: Path) -> None:
    prefix = tmp_path / "demo"
    pvar_path = prefix.with_suffix(".pvar")
    pvar_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\n"
        "1\t123\trs1\tA\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"demo\.pvar: line 3 has fewer than 5 fields\."):
        list(iter_variant_records_for_prefix(str(prefix), "pgen"))
