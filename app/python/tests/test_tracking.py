"""The worker's tracking backends: btrack when it is installed and its
compiled core loads, and a clear refusal when it is not."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import tracking  # noqa: E402

AVAILABLE, WHY = tracking.available()


def moving_labels(t=5, z=4, y=40, x=40):
    """Three objects crossing the field at constant velocity."""
    out = np.zeros((t, z, y, x), np.uint32)
    for frame in range(t):
        for k, (y0, x0, vy, vx) in enumerate([(8, 8, 2, 1), (20, 30, -1, 2), (30, 10, 0, -2)], start=1):
            yc, xc = int(y0 + vy * frame), int(x0 + vx * frame)
            out[frame, 1:3, yc - 2:yc + 3, xc - 2:xc + 3] = k
    return out


class TestAvailability(unittest.TestCase):
    def test_reports_why_it_cannot_run(self):
        ok, why = tracking.available()
        self.assertIsInstance(ok, bool)
        if ok:
            self.assertEqual(why, "")
        else:
            # the reason has to be actionable, not just "no"
            self.assertTrue(why)
            self.assertTrue(any(w in why for w in ("pip install", "libstdc++", "will not load")), why)

    @unittest.skipIf(AVAILABLE, "btrack loads here")
    def test_refuses_clearly_when_unavailable(self):
        with self.assertRaises(tracking.NotAvailable) as cm:
            tracking.run_btrack(moving_labels(), (1.0, 1.0, 1.0), {})
        self.assertTrue(str(cm.exception))


@unittest.skipUnless(AVAILABLE, f"btrack unavailable: {WHY}")
class TestBtrack(unittest.TestCase):
    def test_follows_three_objects(self):
        labels = moving_labels()
        out, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"max_distance": 8.0, "min_length": 2})
        self.assertEqual(out.shape, labels.shape)
        self.assertEqual(info["objects"], 15)
        self.assertEqual(info["tracks"], 3)
        self.assertEqual(info["longest"], 5)
        # every object voxel keeps a track id, and one object keeps one id
        for t in range(labels.shape[0]):
            for k in (1, 2, 3):
                ids = set(np.unique(out[t][labels[t] == k]).tolist())
                self.assertEqual(len(ids), 1, f"frame {t} object {k} split across {ids}")

    def test_short_tracks_are_dropped(self):
        labels = moving_labels()
        labels[1:, ...][labels[1:, ...] == 3] = 0   # object 3 exists in one frame only
        _, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"max_distance": 8.0, "min_length": 3})
        self.assertEqual(info["tracks"], 2)

    def test_a_missing_configuration_is_named(self):
        with self.assertRaises(tracking.NotAvailable) as cm:
            tracking.run_btrack(moving_labels(), (1.0, 1.0, 1.0), {"config": "/no/such/config.json"})
        self.assertIn("config", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
