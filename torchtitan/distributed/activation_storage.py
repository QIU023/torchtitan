# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage backends for the tensors autograd saves, chosen per tensor by a policy."""

from __future__ import annotations

import contextlib
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch_remat as remat

# (pipeline stage, micro-batch, layer): the unit that is saved together and read back together.
ChunkKey = tuple[int, int, int]


class StorageBackend(Protocol):
    """Where a saved tensor waits for backward."""

    def put(self, tensor: torch.Tensor, stream: torch.cuda.Stream | None) -> Any: ...

    def get(self, payload: Any, out: torch.Tensor, stream: torch.cuda.Stream | None) -> None: ...

    def free(self, payload: Any) -> None: ...


class HostBackend:
    """Pinned host memory, copied on the storage stream."""

    def put(self, tensor: torch.Tensor, stream: torch.cuda.Stream | None) -> torch.Tensor:
        pinned = tensor.is_cuda
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=pinned)
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            host.copy_(tensor, non_blocking=pinned)
        return host

    def get(self, payload: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            out.copy_(payload, non_blocking=out.is_cuda)

    def free(self, payload: torch.Tensor) -> None:
        # The caching host allocator keeps a pinned block until the copies that read it finish.
        pass


@dataclass
class _Saved:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    backend: str
    payload: Any
    source: weakref.ref
    tensor: torch.Tensor | None = None
    ready: torch.cuda.Event | None = None
    readers: int = 0


@dataclass
class _Ref:
    saved: _Saved
    chunk: ChunkKey


class ActivationStorage:
    """Routes saved tensors to backends in forward and reads them back one layer ahead in backward.

    Every device allocation happens on the compute stream; a tensor copied out stays referenced
    until its copy finishes instead of being handed to the storage stream's allocator.
    """

    def __init__(
        self,
        device: torch.device,
        backends: dict[str, StorageBackend],
        policy: Callable[[torch.Tensor, ChunkKey], str | None],
        *,
        min_tensor_bytes: int = 1 << 20,
    ) -> None:
        self._device = device
        self._backends = backends
        self._policy = policy
        self._min_bytes = min_tensor_bytes
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._stage_mb: tuple[int, int] | None = None
        self._layer: int | None = None
        self._keep: set[int] = set()
        self._by_id: dict[int, _Saved] = {}
        self._chunks: dict[ChunkKey, list[_Ref]] = defaultdict(list)
        self._layers: dict[int, list[int]] = {}
        self._started: set[ChunkKey] = set()
        self._outgoing: list[tuple[torch.cuda.Event | None, torch.Tensor]] = []
        self.stats: dict[str, float] = defaultdict(float)

    def register_stage(self, stage: int, layers: torch.nn.Module) -> None:
        """Track which layer of ``stage`` is running by a pre-hook on each of its blocks."""
        ids = []
        for name, block in layers.named_children():
            layer = int(name)
            ids.append(layer)
            block.register_forward_pre_hook(self._make_layer_hook(layer))
        self._layers[stage] = sorted(ids)

    def _make_layer_hook(self, layer: int):
        def hook(module, args):
            self._layer = layer

        return hook

    @contextlib.contextmanager
    def forward(self, stage: int, mb: int, keep: Sequence[Any] = ()) -> Iterator[None]:
        """Run ``stage``'s forward of ``mb`` with its saves routed; ``keep`` stay on the device."""
        self._stage_mb = (stage, mb)
        self._layer = None
        self._keep = {id(t) for t in keep if isinstance(t, torch.Tensor)}
        self._by_id = {}
        try:
            with remat.saved_tensors_hooks(self._pack, self._unpack, capture_context=self._capture):
                yield
        finally:
            self._stage_mb = None
            self._layer = None
            self._keep = set()
            self._by_id = {}

    def _capture(self) -> ChunkKey | None:
        if self._stage_mb is None or self._layer is None:
            return None
        return (*self._stage_mb, self._layer)

    def _pack(self, tensor: torch.Tensor) -> Any:
        self._drain()
        chunk = None
        with contextlib.suppress(RuntimeError):
            chunk = remat.current_saved_tensor_info().context
        chunk = chunk if chunk is not None else self._capture()
        if (
            chunk is None
            or id(tensor) in self._keep
            or tensor.device.type != self._device.type
            or not tensor.is_contiguous()
            or tensor.numel() == 0
            or tensor.numel() * tensor.element_size() < self._min_bytes
            # a view leaves its base allocated: moving it copies without freeing
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
        ):
            return tensor
        saved = self._by_id.get(id(tensor))
        if saved is None or saved.source() is not tensor:
            name = self._policy(tensor, chunk)
            if name is None:
                return tensor
            if self._stream is not None:
                self._stream.wait_stream(torch.cuda.current_stream(self._device))
            payload = self._backends[name].put(tensor, self._stream)
            done = None
            if self._stream is not None:
                done = torch.cuda.Event()
                done.record(self._stream)
            self._outgoing.append((done, tensor))
            saved = _Saved(tensor.shape, tensor.dtype, tensor.device, name, payload, weakref.ref(tensor))
            self._by_id[id(tensor)] = saved
            self.stats[f"{name}_bytes"] += tensor.numel() * tensor.element_size()
        saved.readers += 1
        ref = _Ref(saved, chunk)
        self._chunks[chunk].append(ref)
        return ref

    def _unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Ref):
            return packed
        chunk = packed.chunk
        if chunk not in self._started:
            self._started.add(chunk)
            self._start_chunk(chunk)
        saved = packed.saved
        if saved.tensor is None:
            self.stats["late_fetches"] += 1
            self._fetch(saved)
        if saved.ready is not None:
            torch.cuda.current_stream(self._device).wait_event(saved.ready)
        assert saved.tensor is not None
        return saved.tensor

    def _start_chunk(self, chunk: ChunkKey) -> None:
        # Backward reaches this layer: the one after it is done, the one before it comes next.
        stage, mb, layer = chunk
        layers = self._layers.get(stage, [])
        if layer in layers:
            pos = layers.index(layer)
            if pos + 1 < len(layers):
                self._release((stage, mb, layers[pos + 1]))
            if pos > 0:
                self.prefetch((stage, mb, layers[pos - 1]))

    def prefetch(self, chunk: ChunkKey) -> None:
        """Start reading ``chunk``'s saves back onto the device."""
        self._drain()
        for ref in self._chunks.get(chunk, []):
            if ref.saved.tensor is None:
                self._fetch(ref.saved)

    def prefetch_first(self, stage: int, mb: int) -> None:
        """Start reading back the saves of the layer ``stage``'s backward of ``mb`` needs first."""
        layers = self._layers.get(stage)
        if layers:
            self.prefetch((stage, mb, layers[-1]))

    def _fetch(self, saved: _Saved) -> None:
        out = torch.empty(saved.shape, dtype=saved.dtype, device=saved.device)
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream(self._device))
        self._backends[saved.backend].get(saved.payload, out, self._stream)
        if self._stream is not None:
            saved.ready = torch.cuda.Event()
            saved.ready.record(self._stream)
        saved.tensor = out

    def _release(self, chunk: ChunkKey) -> None:
        for ref in self._chunks.pop(chunk, []):
            saved = ref.saved
            saved.readers -= 1
            if saved.readers == 0:
                saved.tensor = None
                self._backends[saved.backend].free(saved.payload)
        self._started.discard(chunk)

    def finish(self, stage: int, mb: int) -> None:
        """Drop what ``stage``'s backward of ``mb`` read back."""
        for chunk in [c for c in self._chunks if c[0] == stage and c[1] == mb]:
            self._release(chunk)

    def _drain(self) -> None:
        # A source copied out is freed on the compute stream once its copy has finished.
        self._outgoing = [(e, t) for e, t in self._outgoing if e is not None and not e.query()]

