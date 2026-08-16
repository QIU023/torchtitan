# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Defer the vision tower's backward so it can run in a pipeline bubble.

Report sec 5.2.3: "the backward passes are handled analogously". The forward half only
had to decide WHEN to call the encode, because nothing else consumes it. The backward
has no such freedom by default: the tower's output is spliced into the text embedding
and the two share one autograd graph, so the tower's backward happens inside the
splicing stage's ``backward_one_chunk``, inline, wherever the schedule put that action.

Moving it means cutting the graph at the seam. :func:`cut_for_deferred_backward`
detaches the tower's output, splices the detached stand-in, and captures the gradient
that arrives on it with a tensor hook; the tower's own backward is then replayed later,
explicitly, at a planned slot.

## The invariant that matters more than the placement

Every deferred backward MUST run before the optimizer step. A gradient that was cut off
and never re-run is not a slow step, it is silently wrong training: the tower's
parameters simply do not get that micro-batch's contribution, and nothing raises.
:meth:`GradQueue.drain` is therefore called unconditionally at step end for whatever the
plan did not place, and :meth:`GradQueue.assert_empty` exists so a caller can turn a
leak into an exception rather than a quiet accuracy loss.

Ordering does not matter for correctness -- parameter gradients accumulate -- so a
deferred backward is free to run in any bubble after its gradient arrives. What it costs
is memory: the tower's forward graph for that micro-batch has to stay alive from the
encode until the deferred backward runs, which is a longer window than the forward
prefetch's and is the real bound on how much of the backward can be moved.
"""

from __future__ import annotations

import torch

from torchtitan.tools.logging import logger


def cut_for_deferred_backward(
    features: torch.Tensor, queue: "GradQueue", microbatch: int
) -> torch.Tensor:
    """Return a stand-in for ``features`` whose gradient is queued, not propagated.

    Splice the RESULT into the text embedding. The tower's graph stays alive and
    untouched until :meth:`GradQueue.run_one` or :meth:`GradQueue.drain` replays the
    captured gradient into it.

    A detached leaf plus a tensor hook, and both halves of that are load-bearing.

    The detach makes the tower's graph unreachable from the text's, which is what keeps
    the text's ``.backward()`` from freeing it. An ``autograd.Function`` wrapping the
    tower's output does NOT achieve that even when its backward returns ``None``:
    measured, the deferred pass then dies with "Trying to backward through the graph a
    second time", and it only survives if the text backward is given
    ``retain_graph=True`` -- which would mean holding the whole text graph for the sake
    of the tower, and the pipeline calls that backward itself.

    The hook, rather than a Function, is what fires at the right moment without putting
    anything back in the graph: ``detached`` is a leaf of the text graph, so autograd
    computes its gradient and calls the hook there. Returning ``None`` from the hook
    leaves that gradient as it is; returning anything else would rewrite it.
    """
    detached = features.detach().requires_grad_(True)

    def _capture(grad: torch.Tensor):
        queue.stash(microbatch, features, grad)
        return None

    detached.register_hook(_capture)
    return detached


class GradQueue:
    """Vision backwards whose gradient has arrived but which have not run yet."""

    def __init__(self) -> None:
        self._pending: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.ran = 0
        self.drained = 0

    def stash(self, microbatch: int, output: torch.Tensor, grad: torch.Tensor) -> None:
        self._pending.setdefault(microbatch, []).append((output, grad))

    def has(self, microbatch: int) -> bool:
        return bool(self._pending.get(microbatch))

    def run_one(self, microbatch: int) -> bool:
        """Run the tower's backward for ``microbatch`` if its gradient has arrived.

        False when it has not: the plan is derived from the schedule's shape, so a slot
        can come up before the gradient does, and that is not an error -- the entry stays
        pending and the step-end drain will take it.
        """
        entries = self._pending.pop(microbatch, None)
        if not entries:
            return False
        for output, grad in entries:
            torch.autograd.backward(output, grad)
            self.ran += 1
        return True

    def drain(self) -> int:
        """Run everything still pending. Called unconditionally at step end.

        This is not a fallback for tidiness. A deferred backward that never runs means
        the tower silently misses that micro-batch's gradient, with no error anywhere,
        so the drain is the correctness guarantee and the placement is only the
        optimisation.
        """
        count = 0
        for microbatch in sorted(self._pending):
            for output, grad in self._pending[microbatch]:
                torch.autograd.backward(output, grad)
                count += 1
        self._pending.clear()
        self.drained += count
        return count

    def assert_empty(self, where: str) -> None:
        if self._pending:
            raise AssertionError(
                f"{where}: {sum(len(v) for v in self._pending.values())} vision "
                f"backward(s) still pending for micro-batches "
                f"{sorted(self._pending)}. Running the optimizer now would train the "
                f"tower on incomplete gradients."
            )

    def report(self, placed: int) -> None:
        level = logger.info if self.drained == 0 else logger.warning
        level(
            "DEP bubble backward: %d ran at a planned slot, %d drained at step end "
            "(%d slots planned)",
            self.ran,
            self.drained,
            placed,
        )
        self.ran = 0
        self.drained = 0
