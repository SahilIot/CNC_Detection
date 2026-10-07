import csv
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "cnc_backend"),
)

from detection.pose_detector import PoseDetector
from events.event_manager import EventManager
from safety.rules import SafetyRules


class FakeEventManager:
    def __init__(self):
        self.records = []

    def log(self, camera_id, video_time, event_name, track_ids, details):
        self.records.append((event_name, track_ids, details))
        return "event-timestamp"


def make_config():
    return SimpleNamespace(
        multiple_person_limit_seconds=120,
        allowed_people_per_zone=1,
        default_allowed_people=1,
        multiple_person_cooldown_seconds=1800,
        absence_limit_seconds=300,
        head_down_violation_seconds=120,
        track_grace_seconds=1.5,
        inside_grace_seconds=2.5,
        track_id_switch_max_seconds=10,
        track_id_switch_center_ratio=1.0,
        track_id_switch_iou_threshold=0.1,
        duplicate_iou_threshold=0.65,
        duplicate_center_ratio=0.25,
    )


def make_person(head_down=True):
    return {
        "id": 1,
        "x1": 10,
        "y1": 10,
        "x2": 110,
        "y2": 210,
        "zone": 0,
        "head_down": head_down,
    }


class HeadDownPostureTests(unittest.TestCase):
    def test_pose_keypoints_distinguish_head_down_from_upright(self):
        confidences = [0.9] * 5
        upright = [
            [50, 58], [45, 50], [55, 50], [40, 51], [60, 51],
        ]
        head_down = [
            [50, 66], [45, 50], [55, 50], [40, 51], [60, 51],
        ]

        self.assertFalse(
            PoseDetector.is_head_down(upright, confidences, 200, 0.25, 0.05)
        )
        self.assertTrue(
            PoseDetector.is_head_down(head_down, confidences, 200, 0.25, 0.05)
        )

    def test_violation_is_logged_once_after_two_minutes(self):
        event_manager = FakeEventManager()
        rules = SafetyRules("camera_01", 1, event_manager, make_config())

        self.assertEqual(rules.update([make_person()], 0, 1)[2], [])
        for second in range(1, 120):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 119.9, 1)[2], [])

        events = rules.update([make_person()], 120, 1)[2]
        self.assertEqual([event["event_type"] for event in events], [
            "HEAD_DOWN_VIOLATION",
        ])
        self.assertEqual(len(event_manager.records), 1)
        self.assertEqual(rules.update([make_person()], 121, 1)[2], [])

    def test_posture_or_long_tracking_gap_restarts_timer(self):
        event_manager = FakeEventManager()
        rules = SafetyRules("camera_01", 1, event_manager, make_config())

        rules.update([make_person()], 0, 1)
        rules.update([make_person(False)], 30, 1)
        rules.update([make_person()], 31, 1)
        for second in range(32, 151):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 150.9, 1)[2], [])
        self.assertEqual(
            rules.update([make_person()], 151, 1)[2][0]["event_type"],
            "HEAD_DOWN_VIOLATION",
        )

        rules.update([make_person()], 200, 1)
        rules.update([], 202, 1)
        rules.update([make_person()], 203, 1)
        for second in range(204, 323):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 322.9, 1)[2], [])
        self.assertEqual(
            rules.update([make_person()], 323, 1)[2][0]["event_type"],
            "HEAD_DOWN_VIOLATION",
        )

    def test_violation_is_counted_in_csv_and_has_screenshot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            events = EventManager(output_dir)
            timestamp = events.log(
                "camera_01",
                120,
                "HEAD_DOWN_VIOLATION",
                1,
                "Head-down posture sustained for 120 seconds",
            )
            screenshot = events.screenshot(
                "camera_01",
                np.zeros((80, 80, 3), dtype=np.uint8),
                120,
                "HEAD_DOWN_VIOLATION",
                {1},
                zone=0,
                timestamp=timestamp,
            )

            csv_path = Path(output_dir) / "cameras" / "camera_01" / "events.csv"
            with csv_path.open(encoding="utf-8", newline="") as event_file:
                rows = list(csv.DictReader(event_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event"], "HEAD_DOWN_VIOLATION")
            self.assertTrue(screenshot.is_file())
            self.assertEqual(
                events.list_events()[0]["screenshot"],
                screenshot.name,
            )


if __name__ == "__main__":
    unittest.main()
