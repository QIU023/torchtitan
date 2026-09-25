# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which stage micro-batches a pipeline rank moves off the device, and when their saves come back."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from torch.distributed.pipelining.schedules import _ComputationType

_BACKWARD = {_ComputationType.FULL_BACKWARD, _ComputationType.BACKWARD_INPUT}


@dataclass(kw_only=True)
class PPOffloadKnobs:
    """Park the activations a pipeline rank holds longest on host memory."""

    microbatches: int = 0
    """Stage micro-batches per rank whose saves go to host, the longest held first; 0 is off."""

    lead: int = 1
    """Actions before a stage's backward at which the saves of its last layer start coming back."""

    min_tensor_mib: int = 1
    """Saves smaller than this stay on the device."""


class ActivationPlan:
    """A rank's moved stage micro-batches, where each goes, and the action that brings it back."""

    def __init__(
        self,
        order: list[Any],
        stage_layers: dict[int, int],
        moves: list[tuple[str, int, int]],
    ) -> None:
        """``moves`` lists (backend, count, lead): each backend takes the next ``count`` longest-held
        stage micro-batches, whose saves come back ``lead`` actions before their backward."""
        actions = []
        for a in order:
            if a is None:
                continue
            if a.computation_type == _ComputationType.FORWARD:
                actions.append(("F", a.stage_index, a.microbatch_index))
            elif a.computation_type in _BACKWARD:
                actions.append(("B", a.stage_index, a.microbatch_index))
        index = {key: i for i, key in enumerate(actions)}
        forwards = [key[1:] for key in index if key[0] == "F"]

        def held(key: tuple[int, int]) -> int:
            # a stage saves one input per layer beyond its first, held from its forward to its backward
            s, mb = key
            return max(stage_layers.get(s, 1) - 1, 0) * (index[("B", s, mb)] - index[("F", s, mb)])

        ranked = sorted((k for k in forwards if ("B", *k) in index), key=held, reverse=True)
        self._backend: dict[tuple[int, int], str] = {}
        self._due: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
        for backend, count, lead in moves:
            # a backward too close to its forward leaves no window: the copy back would overlap the copy out
            eligible = [k for k in ranked if index[("B", *k)] - index[("F", *k)] > lead + 1]
            for key in [k for k in eligible if k not in self._backend][:count]:
                self._backend[key] = backend
                trigger = max(index[("B", *key)] - lead, index[("F", *key)] + 1)
                self._due[actions[trigger]].append(key)

    def backend(self, stage: int, mb: int) -> str | None:
        return self._backend.get((stage, mb))

    def due(self, kind: str, stage: int, mb: int) -> list[tuple[int, int]]:
        """Stage micro-batches whose first saves start coming back when this action starts."""
        return self._due.get((kind, stage, mb), [])

