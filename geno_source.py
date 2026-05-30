"""geno_source.py — Format-agnostic genotype block providers.

Each source reads raw genotypes from a specific file format and returns
(n_samples, n_snps) int8 blocks with values in {0, 1, 2, missing_val}.
The downstream streamer (GenoBlockStreamer) consumes these blocks
identically regardless of source format.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class GenoBlockSource(Protocol):
    """Protocol for genotype block providers."""

    n: int            # number of samples
    m: int            # number of variants
    missing_val: int  # int8 value used for missing genotypes

    def read_block(self, snp_start: int, snp_count: int) -> np.ndarray:
        """Return (n, snp_count) int8 Fortran-order block.

        Values must be in {0, 1, 2, missing_val}.
        """
        ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# PLINK1 BED
# ---------------------------------------------------------------------------

class BedGenoSource:
    """Read hardcall genotypes from a PLINK1 BED file.

    Parameters
    ----------
    sample_mask : ndarray of bool, optional
        Boolean mask over the full sample set.  When supplied, only the
        ``True`` rows are returned by ``read_block`` — the subsetting is
        pushed into ``bed_reader.read()`` so the library decodes only the
        needed samples, avoiding a costly post-hoc copy + transpose.
    """

    def __init__(
        self,
        bed_prefix: str,
        threads: int | None = None,
        sample_mask: np.ndarray | None = None,
    ):
        from bed_reader import open_bed

        self._bed_prefix = bed_prefix
        self._bed = open_bed(bed_prefix + ".bed")
        self._n_full: int = int(self._bed.iid_count)
        self.m: int = int(self._bed.sid_count)
        self._threads = max(1, int(threads or (os.cpu_count() or 1)))

        # Push sample subsetting into bed_reader via integer indices.
        if sample_mask is not None:
            sample_mask = np.asarray(sample_mask, dtype=bool)
            self._sample_idx = np.where(sample_mask)[0]
            self.n: int = int(self._sample_idx.shape[0])
        else:
            self._sample_idx = None
            self.n: int = self._n_full

        # Detect missing-value encoding used by bed_reader
        self.missing_val: int = -127
        probe = self._bed.read(
            np.s_[:1, :1], dtype=np.int8, order="F", num_threads=1,
        )
        neg = probe[probe < 0]
        if neg.size > 0:
            self.missing_val = int(neg.reshape(-1)[0])

    def read_block(self, snp_start: int, snp_count: int) -> np.ndarray:
        if snp_count <= 0:
            return np.empty((self.n, 0), dtype=np.int8, order="F")
        row_sel = self._sample_idx if self._sample_idx is not None else np.s_[:]
        return self._bed.read(
            np.s_[row_sel, snp_start : snp_start + snp_count],
            dtype=np.int8, order="F", num_threads=self._threads,
        )

    def read_block_variant_major(self, snp_start: int, snp_count: int) -> np.ndarray:
        if snp_count <= 0:
            return np.empty((0, self.n), dtype=np.int8)
        return np.ascontiguousarray(self.read_block(snp_start, snp_count).T)

    def close(self) -> None:
        bed = getattr(self, "_bed", None)
        if bed is not None and hasattr(bed, "close"):
            try:
                bed.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Failed to close BED source.", exc_info=True)
        self._bed = None


# ---------------------------------------------------------------------------
# PLINK2 PGEN
# ---------------------------------------------------------------------------

class PgenGenoSource:
    """Read hardcall genotypes from a PLINK2 PGEN file.

    Parameters
    ----------
    sample_mask : ndarray of bool, optional
        Boolean mask over the full sample set.  When supplied, only the
        ``True`` rows are returned by ``read_block``.
    """

    def __init__(self, pgen_prefix: str, sample_mask: np.ndarray | None = None):
        try:
            import pgenlib
        except ImportError as exc:
            raise ImportError(
                "pgenlib is required for direct PGEN reading: "
                "pip install pgenlib"
            ) from exc

        self._pgen_prefix = pgen_prefix
        self._reader = pgenlib.PgenReader(
            bytes(pgen_prefix + ".pgen", "utf-8"),
        )
        self._n_full: int = int(self._reader.get_raw_sample_ct())
        self.m: int = int(self._reader.get_variant_ct())

        if sample_mask is not None:
            sample_mask = np.asarray(sample_mask, dtype=bool)
            self._sample_idx = np.where(sample_mask)[0].astype(np.uint32, copy=False)
            # Push sample subsetting into pgenlib so it decodes only
            # the kept samples instead of the full source sample set.
            self._reader.change_sample_subset(self._sample_idx)
            self.n: int = int(self._sample_idx.shape[0])
        else:
            self._sample_idx = None
            self.n: int = self._n_full

        self._chunk_buf_i8: np.ndarray | None = None
        # Preserve pgenlib's native hardcall-missing sentinel so we do not
        # need an extra full-size boolean "bad" mask per block.
        self.missing_val = self._detect_missing_val()

    def _detect_missing_val(self) -> int:
        if self.m <= 0 or self.n <= 0:
            return -9
        probe = np.empty((1, self.n), dtype=np.int8)
        self._reader.read_range(0, 1, probe)
        neg = probe[probe < 0]
        if neg.size > 0:
            return int(neg.reshape(-1)[0])
        return -9

    def _ensure_chunk_buf(self, chunk_w: int) -> np.ndarray:
        buf = self._chunk_buf_i8
        if buf is None or buf.shape[0] < chunk_w:
            buf = np.empty((chunk_w, self.n), dtype=np.int8)
            self._chunk_buf_i8 = buf
        return buf[:chunk_w]

    def read_block_variant_major(self, snp_start: int, snp_count: int) -> np.ndarray:
        if snp_count <= 0:
            return np.empty((0, self.n), dtype=np.int8)
        buf = self._ensure_chunk_buf(snp_count)
        self._reader.read_range(snp_start, snp_start + snp_count, buf)
        # pgenlib already returns hardcalls in {0,1,2,missing_sentinel}; keep
        # its native missing encoding to avoid materializing a huge boolean
        # mask over the entire decode buffer.
        return buf

    def read_block(self, snp_start: int, snp_count: int) -> np.ndarray:
        if snp_count <= 0:
            return np.empty((self.n, 0), dtype=np.int8, order="F")
        block_vm = self.read_block_variant_major(snp_start, snp_count)
        return np.asfortranarray(block_vm.T)

    def close(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            try:
                reader.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Failed to close PGEN reader.", exc_info=True)
        self._reader = None
        self._chunk_buf_i8 = None
