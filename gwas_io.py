from __future__ import annotations

import json
import os
from typing import TextIO


GWAS_HEADER = (
    "component_index\tcomponent_name\tvariant_index_global\tvariant_index_local\t"
    "chrom\tvariant_id\tcm\tbp\ta1\ta2\taf\tn_obs\tbeta\tse\tbeta_std\tse_std\tt\tp\n"
)


def open_gwas_tsv(out_prefix: str) -> tuple[str, TextIO]:
    if not out_prefix:
        raise ValueError("GWAS output requires a non-empty out_prefix.")
    out_path = out_prefix + ".gwas.tsv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fh = open(out_path, "w")
    fh.write(GWAS_HEADER)
    return out_path, fh


def write_gwas_metadata(out_prefix: str, payload: dict) -> str:
    meta_path = out_prefix + ".gwas_metadata.json"
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return meta_path


__all__ = ["open_gwas_tsv", "write_gwas_metadata"]
