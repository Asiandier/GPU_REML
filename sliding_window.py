"""
sliding_window.py — Pinned staging buffers and mmap page advisory.

This module provides mmap page-access advisory helpers used by the
genotype streaming pipeline.
"""

from __future__ import annotations

import ctypes
import mmap
import os
from functools import lru_cache

import ctypes.util

@lru_cache(maxsize=1)
def _get_libc():
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    return ctypes.CDLL(libc_name, use_errno=True)

def madvise_sequential(mmap_obj: mmap.mmap) -> bool:
    """
    Hint to the kernel that this mapping will be accessed sequentially.

    MADV_SEQUENTIAL causes aggressive readahead: the kernel prefetches
    pages ahead of the access point and drops pages behind it. This is
    the correct policy for the kv() streaming scan pattern.

    This is hardware-agnostic — the kernel automatically adapts
    readahead size to disk speed and available RAM.
    """
    MADV_SEQUENTIAL = 2
    try:
        mmap_obj.madvise(MADV_SEQUENTIAL)
        return True
    except (AttributeError, OSError):
        # Python < 3.8 or non-Linux
        return _madvise_raw(mmap_obj, 0, len(mmap_obj), MADV_SEQUENTIAL)

def _madvise_raw(
    mmap_obj: mmap.mmap, offset: int, length: int, advice: int
) -> bool:
    """Call madvise() on a byte range of the mmap."""
    if length <= 0:
        return False
    # Python 3.8+ has mmap.madvise(option, start, length)
    try:
        mmap_obj.madvise(advice, offset, length)
        return True
    except (AttributeError, OSError, TypeError):
        pass
    # Fallback: ctypes. Requires writable mmap (ACCESS_COPY or ACCESS_WRITE).
    try:
        libc = _get_libc()
    except OSError:
        return False
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        buf = (ctypes.c_char * len(mmap_obj)).from_buffer(mmap_obj)
        base_addr = ctypes.addressof(buf) + offset
        aligned_addr = base_addr & ~(page_size - 1)
        aligned_length = length + (base_addr - aligned_addr)
        aligned_length = ((aligned_length + page_size - 1) // page_size) * page_size
        rc = libc.madvise(
            ctypes.c_void_p(aligned_addr),
            ctypes.c_size_t(aligned_length),
            ctypes.c_int(advice),
        )
        return rc == 0
    except (AttributeError, BufferError, OSError, TypeError, ValueError):
        return False
