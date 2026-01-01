import unittest

from fridge.data.windowing import compute_window_starts, build_windows


class WindowingTests(unittest.TestCase):
    def test_compute_window_starts(self):
        starts = compute_window_starts(
            duration_s=20.0,
            window_s=2.0,
            hop_s=1.0,
            trim_start_s=2.0,
            trim_end_s=2.0,
        )
        self.assertEqual(len(starts), 15)
        self.assertAlmostEqual(starts[0], 2.0)
        self.assertAlmostEqual(starts[-1], 16.0)

    def test_build_windows(self):
        records = [
            {
                "id": "rec1",
                "path": "dummy.wav",
                "duration_s": 10.0,
                "fridge_on": 1,
                "ac_on": 0,
                "environment": "day",
            }
        ]
        windows = build_windows(records, 2.0, 1.0, 1.0, 1.0)
        self.assertTrue(windows)
        self.assertEqual(windows[0].recording_id, "rec1")


if __name__ == "__main__":
    unittest.main()
