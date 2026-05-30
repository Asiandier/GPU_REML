from __future__ import annotations

import dataclasses
import os
from typing import Iterator


@dataclasses.dataclass(frozen=True)
class VariantRecord:
    chrom: str
    variant_id: str
    cm: str
    bp: str
    a1: str
    a2: str


def _iter_bim_records(path: str) -> Iterator[VariantRecord]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            cols = line.rstrip("\n").split()
            if len(cols) < 6:
                raise ValueError(f"{path}: line {line_no} has fewer than 6 fields.")
            yield VariantRecord(
                chrom=cols[0],
                variant_id=cols[1],
                cm=cols[2],
                bp=cols[3],
                a1=cols[4],
                a2=cols[5],
            )


def _iter_pvar_records(path: str) -> Iterator[VariantRecord]:
    chrom_idx = id_idx = pos_idx = ref_idx = alt_idx = None
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line:
                continue
            if line.startswith("##"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) == 1:
                cols = line.rstrip("\n").split()
            if not cols:
                continue
            if cols[0].startswith("#"):
                header = [c.lstrip("#") for c in cols]
                col_map = {name: idx for idx, name in enumerate(header)}
                required = ["CHROM", "ID", "POS", "REF", "ALT"]
                missing = [name for name in required if name not in col_map]
                if missing:
                    raise ValueError(f"{path}: missing PVAR columns: {missing}")
                chrom_idx = col_map["CHROM"]
                id_idx = col_map["ID"]
                pos_idx = col_map["POS"]
                ref_idx = col_map["REF"]
                alt_idx = col_map["ALT"]
                continue
            if chrom_idx is None:
                raise ValueError(f"{path}: missing header line beginning with #CHROM.")
            max_required_idx = max(chrom_idx, id_idx, pos_idx, ref_idx, alt_idx)
            if len(cols) <= max_required_idx:
                raise ValueError(
                    f"{path}: line {line_no} has fewer than {max_required_idx + 1} fields."
                )
            # Match the rest of the codebase's convention: A1 is the ALT/effect
            # allele and A2 is REF.
            alt = cols[alt_idx].split(",")[0]
            yield VariantRecord(
                chrom=cols[chrom_idx],
                variant_id=cols[id_idx],
                cm=".",
                bp=cols[pos_idx],
                a1=alt,
                a2=cols[ref_idx],
            )


def iter_variant_records_for_prefix(prefix: str, fmt: str) -> Iterator[VariantRecord]:
    fmt_norm = fmt.strip().lower()
    if fmt_norm == "bed":
        yield from _iter_bim_records(prefix + ".bim")
        return
    if fmt_norm == "pgen":
        pvar_path = prefix + ".pvar"
        if not os.path.exists(pvar_path):
            raise FileNotFoundError(
                f"Missing {pvar_path}. Only plain-text .pvar sidecars are currently supported."
            )
        yield from _iter_pvar_records(pvar_path)
        return
    raise ValueError(f"Unsupported variant metadata format: {fmt!r}")


__all__ = ["VariantRecord", "iter_variant_records_for_prefix"]
