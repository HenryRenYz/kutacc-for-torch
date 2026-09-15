# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.

"""PyTorch distributed backend backed by Kutacc."""

import ctypes
import glob
import os
from datetime import timedelta
from typing import Optional, Sequence

import torch
import torch.distributed as dist

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

# Load order matters: runtimes first (libc++/unwind), then OpenMP and the
# verb core, then kupl and kutacc.  RTLD_GLOBAL makes each SONAME available
# to the next load, so _C resolves everything even before its $ORIGIN RPATH
# is consulted.
_PRELOAD_ORDER = (
    "libunwind.so*",
    "libc++abi.so*",
    "libc++.so*",
    "libnuma.so*",
    "libgomp.so*",
    "libomp.so*",
    "libibverbs.so*",
    "libkupl.so*",
    "libkutacc.so*",
)


def _preload_bundled_libraries() -> None:
    for pattern in _PRELOAD_ORDER:
        for path in sorted(glob.glob(os.path.join(_LIB_DIR, pattern))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                raise ImportError(f"failed to load bundled library {path}: {exc}") from exc


_preload_bundled_libraries()

from . import _C


def _create_backend(store, rank: int, world_size: int, timeout: timedelta):
    if "KUTACC_SHM_COPY_THREADS" not in os.environ:
        maximum = max(1, int(os.environ.get("KUTACC_TORCH_MAX_COPY_THREADS", "8")))
        threads = min(maximum, max(1, torch.get_num_threads()))
        os.environ["KUTACC_SHM_COPY_THREADS"] = str(threads)
        os.environ.setdefault("KUTACC_SHM_COPY_POOL_THREADS", str(threads))
    timeout_ms = max(0, int(timeout.total_seconds() * 1000))
    return _C.create_backend(store, rank, world_size, timeout_ms)


def register_backend() -> None:
    """Register ``kutacc`` as a CPU backend if it is not registered yet."""
    if "kutacc" not in dist.Backend.backend_list:
        dist.Backend.register_backend("kutacc", _create_backend, devices=["cpu"])


def all_gatherv_into_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    counts: Sequence[int],
    displacements: Optional[Sequence[int]] = None,
    group=None,
    async_op: bool = False,
):
    """Ragged rank-major AllGather extension using logical element counts."""
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    backend = group._get_backend(torch.device("cpu"))
    if not isinstance(backend, _C.ProcessGroupKutacc):
        raise TypeError("the selected process group does not use the Kutacc backend")
    if displacements is None:
        displacements = []
    work = backend.allgatherv_into_tensor(output, input, list(counts), list(displacements))
    if async_op:
        return work
    work.wait()
    return None


register_backend()

ProcessGroupKutacc = _C.ProcessGroupKutacc

__all__ = ["ProcessGroupKutacc", "all_gatherv_into_tensor", "register_backend"]
