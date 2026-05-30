from __future__ import annotations

import dataclasses
from typing import Literal


BlockBackendKind = Literal["dense_packed", "sparse12"]


@dataclasses.dataclass(frozen=True)
class DensePackedBlockDescriptor:
    call_idx: int
    snp_start: int
    true_width: int
    packed_width: int
    storage: str = "tmpfile_packed"
    kind: BlockBackendKind = "dense_packed"


@dataclasses.dataclass(frozen=True)
class Sparse12BlockDescriptor:
    call_idx: int
    snp_start: int
    true_width: int
    nnz_het: int = 0
    nnz_hom: int = 0
    has_csc: bool = False
    has_csr: bool = False
    idx_dtype: str = "int32"
    storage: str = "host_sparse_metadata_compact"
    kind: BlockBackendKind = "sparse12"


__all__ = [
    "BlockBackendKind",
    "DensePackedBlockDescriptor",
    "Sparse12BlockDescriptor",
]
