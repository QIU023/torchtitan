# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Park a pipeline rank's saved activations in a pool on another rank."""

from __future__ import annotations

import contextlib
import ctypes
import glob
import json
import logging
import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.tools.utils import round_up

logger = logging.getLogger(__name__)

# The alignment a peer's RDMA write wants.
_ALIGNMENT = 512


def _load_transfer_engine():
    """Import mooncake's TransferEngine, preloading the CUDA 12 runtime its wheel links."""
    try:
        from mooncake.engine import TransferEngine  # pyrefly: ignore [missing-import]
    except ImportError:
        try:
            import nvidia

            for root in nvidia.__path__:
                for lib in glob.glob(
                    os.path.join(root, "cuda_runtime", "lib", "libcudart.so.12*")
                ):
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    break
            from mooncake.engine import (  # pyrefly: ignore [missing-import]
                TransferEngine,
            )
        except (ImportError, OSError) as err:
            raise ImportError(
                "pp_balance_source_ranks is set, which needs the "
                "mooncake-transfer-engine package (and its cu12 runtime, "
                f"nvidia-cuda-runtime-cu12); import failed with: {err}."
            ) from err
    return TransferEngine


def _local_host() -> str:
    """The host part of the address peers reach this rank on."""
    return os.environ.get("MC_LOCAL_HOSTNAME") or socket.gethostname()


def _free_port() -> int:
    port: int
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    return port


@dataclass
class _Handle:
    """One parked tensor: where it sits in the pool and how to rebuild it."""

    offset: int
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class _PoolAllocator:
    """First-fit free list over one span of the pool, coalescing on free."""

    def __init__(self, base: int, capacity: int) -> None:
        self._free: list[tuple[int, int]] = [(base, capacity)]
        self._live: set[int] = set()

    def alloc(self, nbytes: int) -> int | None:
        nbytes = round_up(nbytes, _ALIGNMENT)
        for i, (off, size) in enumerate(self._free):
            if size >= nbytes:
                if size == nbytes:
                    self._free.pop(i)
                else:
                    self._free[i] = (off + nbytes, size - nbytes)
                self._live.add(off)
                return off
        return None

    def is_live(self, offset: int) -> bool:
        return offset in self._live

    def free(self, offset: int, nbytes: int) -> None:
        if offset not in self._live:
            raise RuntimeError(
                f"pool offset {offset} is freed twice; a parked tensor was "
                "unpacked more than once, which this pool cannot serve."
            )
        self._live.discard(offset)
        nbytes = round_up(nbytes, _ALIGNMENT)
        self._free.append((offset, nbytes))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for off, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        self._free = merged


class PPBalanceEngine:
    """Per-rank handle on the balance fabric."""

    def __init__(
        self,
        pp_group,
        *,
        source_ranks: Sequence[int],
        dest_rank: int,
        pool_bytes: int,
        staging_bytes: int,
        min_tensor_bytes: int,
    ) -> None:
        engine_cls = _load_transfer_engine()
        self._rank = dist.get_rank(pp_group)
        self._is_source = self._rank in source_ranks
        self._min_tensor_bytes = min_tensor_bytes
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._staging: torch.Tensor | None = None
        self._staging_bytes = 0

        self._engine = engine_cls()
        host = _local_host()
        # The engine picks its own RPC port regardless of the one requested;
        # a segment is addressed by host:that_port, so the address book shares
        # get_rpc_port(), not the string initialize() was given.
        rc = self._engine.initialize(
            f"{host}:{_free_port()}", "P2PHANDSHAKE", "tcp", ""
        )
        if rc != 0:
            raise RuntimeError(f"mooncake TransferEngine initialize failed: rc={rc}")
        session = f"{host}:{self._engine.get_rpc_port()}"

        # The TCP transport serves host memory only; RDMA serves GPU memory
        # directly. Both sides of a transfer resolve this from their own
        # topology, which is identical across ranks of one homogeneous job.
        on_gpu = any(
            len(hcas) > 0
            for entry in json.loads(self._engine.get_local_topology()).values()
            for hcas in entry
        )

        def _buffer(nbytes: int) -> torch.Tensor:
            if on_gpu:
                return torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

        pool_base = 0
        if self._rank == dest_rank:
            self._pool = _buffer(pool_bytes)
            pool_base = self._pool.data_ptr()
            _check(self._engine.register_memory(pool_base, pool_bytes))
        elif self._is_source:
            self._staging = _buffer(staging_bytes)
            _check(
                self._engine.register_memory(self._staging.data_ptr(), staging_bytes)
            )
            self._staging_bytes = staging_bytes

        book: list[tuple[str, int] | None] = [None] * dist.get_world_size(pp_group)
        dist.all_gather_object(book, (session, pool_base), group=pp_group)
        dest_entry = book[dest_rank]
        assert dest_entry is not None
        self._dest_session, self._pool_base = dest_entry

        # Every source writes into the one pool, so each owns a span of it: one
        # allocator per rank over the whole pool hands two sources the same
        # offset, and the second write takes the first one's bytes.
        if self._is_source:
            share = (pool_bytes // len(source_ranks)) // _ALIGNMENT * _ALIGNMENT
            base = source_ranks.index(self._rank) * share
            role = "source"
        else:
            share = base = 0
            role = "dest" if self._rank == dest_rank else "idle"
        self._alloc = _PoolAllocator(base, share)
        self._stats: dict[str, float] = {
            "park": 0,
            "park_mib": 0.0,
            "fetch": 0,
            "skip_layout": 0,
            "skip_size": 0,
            "skip_pool": 0,
        }
        logger.info(
            "PP balance: rank %d role=%s dest=%s pool=%d MiB span=%d MiB",
            self._rank,
            role,
            self._dest_session,
            pool_bytes >> 20,
            share >> 20,
        )

    # ---- source-side API -------------------------------------------------

    def park(self, t: torch.Tensor) -> _Handle | None:
        """Copy ``t`` into the remote pool; None means "keep it local"."""
        if self._staging is None:
            return None
        nbytes = t.numel() * t.element_size()
        # Contiguous only: rebuilding a non-contiguous save with a normalized
        # layout keeps the VALUES but changes the strides, and backward kernels
        # tile -- and therefore reduce -- by layout, which drifts bitwise.
        if not t.is_cuda or not t.is_contiguous():
            self._stats["skip_layout"] += 1
            return None
        if nbytes < self._min_tensor_bytes or nbytes > self._staging_bytes:
            self._stats["skip_size"] += 1
            return None
        offset = self._alloc.alloc(nbytes)
        if offset is None:
            self._stats["skip_pool"] += 1
            if self._stats["skip_pool"] == 1:
                logger.warning(
                    "PP balance: rank %d ran out of pool at %.1f MiB parked; the "
                    "rest of this step's saves stay local. Raise "
                    "pp_balance_pool_gib to park them.",
                    self._rank,
                    self._stats["park_mib"],
                )
            return None
        flat = t.detach().reshape(-1).view(torch.uint8)
        self._staging[:nbytes].copy_(flat)
        torch.cuda.current_stream().synchronize()
        rc = self._engine.transfer_sync_write(
            self._dest_session,
            self._staging.data_ptr(),
            self._pool_base + offset,
            nbytes,
        )
        if rc != 0:
            logger.warning(
                "PP balance: write to %s failed (rc=%d, %d bytes); keeping the "
                "tensor local.",
                self._dest_session,
                rc,
                nbytes,
            )
            self._alloc.free(offset, nbytes)
            return None
        self._stats["park"] += 1
        self._stats["park_mib"] += nbytes / (1 << 20)
        if self._stats["park"] == 1:
            logger.info(
                "PP balance: rank %d parked its first tensor (%d KiB) on %s.",
                self._rank,
                nbytes >> 10,
                self._dest_session,
            )
        return _Handle(offset, nbytes, t.shape, t.dtype, t.device)

    def fetch(self, h: _Handle) -> torch.Tensor:
        """Read a parked tensor back and free its span; each handle once."""
        if self._staging is None or not self._alloc.is_live(h.offset):
            raise RuntimeError(
                "a parked tensor was unpacked twice: this pool frees a span on "
                "the first read, so a schedule that consumes saved tensors more "
                "than once cannot park them."
            )
        rc = self._engine.transfer_sync_read(
            self._dest_session,
            self._staging.data_ptr(),
            self._pool_base + h.offset,
            h.nbytes,
        )
        if rc != 0:
            raise RuntimeError(f"mooncake read failed: rc={rc}")
        out = torch.empty(h.shape, dtype=h.dtype, device=h.device)
        out.reshape(-1).view(torch.uint8).copy_(self._staging[: h.nbytes])
        torch.cuda.current_stream().synchronize()
        self._alloc.free(h.offset, h.nbytes)
        self._stats["fetch"] += 1
        return out

    def stats(self) -> dict[str, float]:
        """What this rank parked, fetched, and left local."""
        return dict(self._stats)

    def hooks(self):
        """The saved-tensors hooks that park through this engine; a rank that parks nothing gets none."""
        if not self._is_source:
            return contextlib.nullcontext()

        def pack(t: torch.Tensor):
            h = self.park(t)
            return t if h is None else h

        def unpack(obj):
            return self.fetch(obj) if isinstance(obj, _Handle) else obj

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _check(rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"mooncake register_memory failed: rc={rc}")


@dataclass(frozen=True)
class PPBalanceKnobs:
    """Which PP ranks park, where, and how much: ``pipeline_kimi_k3``'s ``pp_balance``."""

    pp_balance_source_ranks: tuple[int, ...] = ()
    pp_balance_dest_rank: int = -1
    pp_balance_pool_gib: float = 2.0
    pp_balance_staging_mib: int = 256
    pp_balance_min_tensor_mib: int = 1


def install_pp_balance(
    stages, pp_group, knobs: PPBalanceKnobs
) -> PPBalanceEngine | None:
    """Build the engine and run ``stages``' forwards under its saved-tensors hooks."""
    source_ranks = list(knobs.pp_balance_source_ranks)
    if not source_ranks:
        return None
    dest = knobs.pp_balance_dest_rank
    if len(set(source_ranks)) != len(source_ranks):
        raise ValueError(f"pp_balance_source_ranks={source_ranks} repeats a rank.")
    if dest in source_ranks:
        raise ValueError(
            f"pp_balance_dest_rank={dest} is also a source rank; the pool "
            "cannot live on a rank that parks into it."
        )
    size = dist.get_world_size(pp_group)
    named = source_ranks + [dest]
    if any(not 0 <= r < size for r in named):
        raise ValueError(
            f"pp_balance names rank(s) {named} outside a pipeline group of {size}."
        )
    engine = PPBalanceEngine(
        pp_group,
        source_ranks=source_ranks,
        dest_rank=dest,
        pool_bytes=int(knobs.pp_balance_pool_gib * (1 << 30)),
        staging_bytes=int(knobs.pp_balance_staging_mib * (1 << 20)),
        min_tensor_bytes=int(knobs.pp_balance_min_tensor_mib * (1 << 20)),
    )
    # Every rank's stages hold the engine, the dest rank's included: it owns the
    # registered pool, and on that rank nothing else refers to it.
    for stage in stages:
        stage.set_forward_context(engine.hooks)
    logger.info(
        "PP balance: rank %d parks the saved tensors of %d stage(s) on rank %d.",
        engine._rank,
        len(stages) if engine._is_source else 0,
        dest,
    )
    return engine
