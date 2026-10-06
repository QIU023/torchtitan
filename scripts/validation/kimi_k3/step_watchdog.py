# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump stalled Python steps using normal, GIL-held frame inspection."""
import sys
import threading
import traceback
from contextlib import contextmanager


@contextmanager
def watch_step(seconds=90, *, stream=None):
    """Report this thread's stack periodically until the step finishes.

    Unlike faulthandler's native timer, this uses normal Python APIs under the
    interpreter lock. It cannot diagnose an extension that holds the GIL forever.
    """
    stream = sys.stderr if stream is None else stream
    target = threading.get_ident()
    done = threading.Event()

    def report():
        while not done.wait(seconds):
            frame = sys._current_frames().get(target)
            if frame is not None:
                print(
                    f"STEP_WATCHDOG: step exceeds {seconds}s; thread={target}",
                    file=stream,
                    flush=True,
                )
                traceback.print_stack(frame, file=stream)
                stream.flush()
                del frame

    thread = threading.Thread(target=report, name="step-watchdog", daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
