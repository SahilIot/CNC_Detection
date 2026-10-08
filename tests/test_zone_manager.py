import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cnc_backend"))

from zones.zone_manager import ZoneManager


class ZoneManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.manager = ZoneManager(Path(self.temp_dir.name), margin_px=12)

    def test_saved_normalized_zones_round_trip_to_camera_pixels(self):
        polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        self.manager.save_normalized("camera_test", [polygon])

        zones = self.manager.load("camera_test", width=200, height=100)

        np.testing.assert_array_equal(
            zones[0],
            np.array([[20, 10], [180, 10], [180, 90], [20, 90]]),
        )

    def test_save_rejects_out_of_range_and_degenerate_polygons(self):
        with self.assertRaisesRegex(ValueError, "normalized finite values"):
            self.manager.save_normalized(
                "camera_test",
                [[[0, 0], [1.1, 0], [0, 1]]],
            )
        with self.assertRaisesRegex(ValueError, "non-zero area"):
            self.manager.save_normalized(
                "camera_test",
                [[[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]]],
            )

    def test_person_assignment_uses_floor_point_and_boundary_tolerance(self):
        zone = np.array(
            [[20, 20], [80, 20], [80, 80], [20, 80]],
            dtype=np.int32,
        )

        self.assertEqual(
            ZoneManager.person_zone([zone], 40, 0, 60, 78),
            0,
        )
        self.assertIsNone(
            ZoneManager.person_zone([zone], 40, 0, 60, 94),
        )
        self.assertEqual(
            ZoneManager.person_zone([zone], 40, 0, 60, 94, margin_px=15),
            0,
        )
        self.assertIsNone(
            ZoneManager.person_zone([zone], 40, 0, 60, 95, margin_px=12),
        )

    def test_overlapping_zones_choose_polygon_closest_to_foot_point(self):
        first = np.array(
            [[0, 0], [60, 0], [60, 60], [0, 60]],
            dtype=np.int32,
        )
        second = np.array(
            [[40, 0], [100, 0], [100, 60], [40, 60]],
            dtype=np.int32,
        )

        self.assertEqual(
            ZoneManager.person_zone([first, second], 65, 0, 85, 50),
            1,
        )


if __name__ == "__main__":
    unittest.main()
