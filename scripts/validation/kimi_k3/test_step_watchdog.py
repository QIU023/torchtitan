# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import threading
import time
import unittest

from step_watchdog import watch_step


class WatchdogTests(unittest.TestCase):
    def test_reports_target_and_stops(self):
        stream = io.StringIO()
        with watch_step(0.01, stream=stream):
            time.sleep(0.05)
        captured = stream.getvalue()
        self.assertIn("STEP_WATCHDOG:", captured)
        self.assertIn("test_reports_target_and_stops", captured)
        time.sleep(0.03)
        self.assertEqual(captured, stream.getvalue())
        self.assertFalse(any(t.name == "step-watchdog" for t in threading.enumerate()))

    def test_fast_step_and_exception_cleanup(self):
        stream = io.StringIO()
        with self.assertRaisesRegex(ValueError, "expected"):
            with watch_step(60, stream=stream):
                raise ValueError("expected")
        self.assertEqual("", stream.getvalue())
        self.assertFalse(any(t.name == "step-watchdog" for t in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
