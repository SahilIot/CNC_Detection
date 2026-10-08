import csv
import tempfile
import threading
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "cnc_backend"),)

from camera.worker import CameraWorker
from detection.pose_detector import PoseDetector
from display.renderer import Renderer
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
        head_down_violation_seconds=300,
        track_grace_seconds=1.5,
        inside_grace_seconds=2.5,
        track_id_switch_max_seconds=10,
        track_id_switch_center_ratio=1.0,
        track_id_switch_iou_threshold=0.1,
        duplicate_iou_threshold=0.65,
        duplicate_center_ratio=0.25,
    )

def make_person(head_down=True, phone_detected=True):
    return {
        "id": 1,
        "x1": 10,
        "y1": 10,
        "x2": 110,
        "y2": 210,
        "zone": 0,
        "head_down": head_down,
        "phone_detected": phone_detected,
    }

class HeadDownPostureTests(unittest.TestCase):
    def test_pose_model_tracks_once_on_full_frame(self):
        class FakePoseModel:
            def track(self, frame, **kwargs):
                self.frame = frame
                self.kwargs = kwargs
                return ["tracked-result"]

        detector = PoseDetector.__new__(PoseDetector)
        detector.model = FakePoseModel()
        detector.image_size = 640
        detector.confidence = 0.35
        detector.device = "cpu"
        detector.half = False
        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        result = detector.track(frame, "botsort.yaml", augment=True)

        self.assertEqual(result, ["tracked-result"])
        self.assertIs(detector.model.frame, frame)
        self.assertTrue(detector.model.kwargs["persist"])
        self.assertEqual(detector.model.kwargs["tracker"], "botsort.yaml")
        self.assertEqual(detector.model.kwargs["classes"], [0])
        self.assertTrue(detector.model.kwargs["augment"])

    def test_person_box_and_pose_are_extracted_from_same_track_result(self):
        class CpuValues:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def numpy(self):
                return self.values

            def int(self):
                return self

            def tolist(self):
                return self.values

            def __len__(self):
                return len(self.values)

        class Boxes:
            def __init__(self, xyxy, ids, confidences):
                self.xyxy = CpuValues(xyxy)
                self.id = CpuValues(ids)
                self.conf = CpuValues(confidences)

            def __len__(self):
                return len(self.xyxy)

        points = np.zeros((1, 17, 2), dtype=float)
        points[0, :5] = [
            [50, 35], [45, 20], [55, 20], [40, 20], [60, 20],
        ]
        points[0, 5:7] = [[40, 50], [60, 50]]
        points[0, 11:13] = [[40, 150], [60, 150]]
        point_confidences = np.full((1, 17), 0.9)
        box = Boxes(
            np.array([[10, 10, 110, 210]], dtype=float),
            [7],
            np.array([0.9]),
        )
        keypoints = SimpleNamespace(
            xy=CpuValues(points),
            conf=CpuValues(point_confidences),
        )
        worker = CameraWorker.__new__(CameraWorker)
        worker.camera_id = "camera_test"
        worker.zone_manager = SimpleNamespace(
            person_zone=lambda zones, x1, y1, x2, y2, margin_px: 0,
        )
        worker.zones = []
        worker.zone_masks = None
        worker.pose_detector = PoseDetector
        worker.config = SimpleNamespace(
            head_pose_keypoint_confidence=0.25,
            head_pose_keypoint_delta=0.05,
            head_pose_hold_seconds=1.5,
            track_grace_seconds=1.5,
            zone_margin_px=60,
        )
        worker.head_down_states = {}
        result = SimpleNamespace(
            boxes=box,
            keypoints=keypoints,
        )

        person = worker._extract_persons([result], 0)[0]

        self.assertEqual(person["id"], 7)
        self.assertEqual((person["x1"], person["y1"], person["x2"], person["y2"]),
                         (10, 10, 110, 210))
        self.assertTrue(person["head_down"])
        self.assertEqual(person["pose_keypoints"][0], (50.0, 35.0, 0.9))

    def test_phone_presence_is_refreshed_at_detection_interval(self):
        phone_box = {"x1": 50, "y1": 100, "x2": 70, "y2": 120}

        class FakePhoneDetector:
            def __init__(self):
                self.results = [[phone_box], []]

            def detect_in_person(self, frame, x1, y1, x2, y2):
                return self.results.pop(0)

        worker = CameraWorker.__new__(CameraWorker)
        worker.config = SimpleNamespace(
            phone_detection_interval=3,
            track_grace_seconds=1.5,
        )
        worker.phone_detector = FakePhoneDetector()
        worker.phone_states = {}
        worker.processed_sequence = 0
        person = {
            "id": 1,
            "x1": 10,
            "y1": 10,
            "x2": 110,
            "y2": 210,
            "zone": 0,
            "head_down": True,
        }

        worker._check_phones(None, [person], 0.0, 1)
        self.assertTrue(person["phone_detected"])
        self.assertEqual(person["phone_boxes"], [phone_box])

        worker._check_phones(None, [person], 0.1, 2)
        self.assertTrue(person["phone_detected"])

        worker._check_phones(None, [person], 0.2, 3)
        self.assertTrue(person["phone_detected"])

        worker._check_phones(None, [person], 0.3, 4)
        self.assertFalse(person["phone_detected"])
        self.assertEqual(person["phone_boxes"], [])

        person["zone"] = None
        worker._check_phones(None, [person], 0.4, 5)
        self.assertFalse(person["phone_detected"])
        self.assertNotIn(1, worker.phone_states)

    def test_phone_detection_is_not_run_without_head_down_posture(self):
        class FakePhoneDetector:
            def detect_in_person(self, *args):
                raise AssertionError("Phone-only detection should not run")

        worker = CameraWorker.__new__(CameraWorker)
        worker.config = SimpleNamespace(
            phone_detection_interval=3,
            track_grace_seconds=1.5,
        )
        worker.phone_detector = FakePhoneDetector()
        worker.phone_states = {}
        worker.processed_sequence = 0
        person = {
            "id": 1,
            "x1": 10,
            "y1": 10,
            "x2": 110,
            "y2": 210,
            "zone": 0,
            "head_down": False,
        }

        worker._check_phones(None, [person], 0.0, 1)

        self.assertFalse(person["phone_detected"])
        self.assertEqual(person["phone_boxes"], [])
        self.assertNotIn(1, worker.phone_states)

    def test_display_uses_the_frame_that_was_processed(self):
        processed_frame = np.full((2, 2, 3), 17, dtype=np.uint8)

        class FakeRenderer:
            def render(self, frame, camera_id, video_time, zones, persons,
                       inside, safety, processing_fps):
                return frame, video_time

        worker = CameraWorker.__new__(CameraWorker)
        worker.camera_id = "camera_test"
        worker.state_lock = threading.Lock()
        worker.latest_raw = processed_frame
        worker.latest_video_time = 12.5
        worker.latest_persons = []
        worker.latest_inside = {}
        worker.processing_fps = 10.0
        worker.zones = []
        worker.safety = None
        worker.renderer = FakeRenderer()

        frame, video_time = worker.get_display()

        np.testing.assert_array_equal(frame, processed_frame)
        self.assertEqual(video_time, 12.5)

    def test_pose_skeleton_draws_keypoints_and_connections(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        keypoints = [(0.0, 0.0, 0.0) for _ in range(17)]
        keypoints[5] = (30.0, 30.0, 0.9)
        keypoints[7] = (50.0, 50.0, 0.9)

        Renderer.draw_pose(frame, keypoints, 0.25)

        self.assertTrue(np.any(frame[:, :, 1] > 0))
        self.assertTrue(np.any(frame[:, :, 0] > 0))
        self.assertTrue(np.max(frame) < 255)

    def test_pose_keypoints_distinguish_head_down_from_upright(self):
        confidences = [0.9] * 17

        def make_pose(nose):
            points = [[0, 0] for _ in range(17)]
            points[:5] = [
                nose, [45, 50], [55, 50], [40, 58], [60, 58],
            ]
            points[5:7] = [[40, 100], [60, 100]]
            points[11:13] = [[40, 160], [60, 160]]
            return points

        self.assertFalse(PoseDetector.is_head_down(
            make_pose([50, 58]), confidences, 200, 0.25, 0.05,
        ))
        self.assertTrue(PoseDetector.is_head_down(
            make_pose([50, 66]), confidences, 200, 0.25, 0.05,
        ))
        low_confidence = confidences.copy()
        low_confidence[1:5] = [0.1] * 4
        self.assertIsNone(PoseDetector.is_head_down(
            make_pose([50, 66]), low_confidence, 200, 0.25, 0.05,
        ))

        tilted_torso = make_pose([62, 60])
        tilted_torso[11:13] = [[80, 160], [100, 160]]
        self.assertTrue(PoseDetector.is_head_down(
            tilted_torso, confidences, 200, 0.25, 0.05,
        ))

        low_confidence_shoulders = confidences.copy()
        low_confidence_shoulders[5] = 0.1
        self.assertIsNone(PoseDetector.is_head_down(
            make_pose([50, 66]), low_confidence_shoulders, 200, 0.25, 0.05,
        ))

        degenerate_torso = make_pose([50, 66])
        degenerate_torso[11:13] = [[40, 100], [60, 100]]
        self.assertIsNone(PoseDetector.is_head_down(
            degenerate_torso, confidences, 200, 0.25, 0.05,
        ))

    def test_violation_requires_head_down_and_phone_for_five_minutes(self):
        event_manager = FakeEventManager()
        rules = SafetyRules("camera_01", 1, event_manager, make_config())

        self.assertEqual(rules.update(
            [make_person(phone_detected=False)], 0, 1,
        )[2], [])
        for second in range(1, 601):
            self.assertEqual(rules.update(
                [make_person(phone_detected=False)], second, 1,
            )[2], [])
        self.assertEqual(rules.update([make_person()], 610, 1)[2], [])
        for second in range(611, 910):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 909.9, 1)[2], [])

        events = rules.update([make_person()], 910, 1)[2]
        self.assertEqual([event["event_type"] for event in events], [
            "HEAD_DOWN_PHONE_VIOLATION",
        ])
        self.assertEqual(len(event_manager.records), 1)
        self.assertEqual(rules.update([make_person()], 911, 1)[2], [])

    def test_phone_or_posture_interruptions_restart_combined_timer(self):
        rules = SafetyRules("camera_01", 1, FakeEventManager(), make_config())

        rules.update([make_person()], 0, 1)
        rules.update([make_person(phone_detected=False)], 30, 1)
        rules.update([make_person()], 31, 1)
        for second in range(32, 331):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 330.9, 1)[2], [])
        self.assertEqual(
            rules.update([make_person()], 331, 1)[2][0]["event_type"],
            "HEAD_DOWN_PHONE_VIOLATION",
        )

    def test_missing_detection_does_not_leave_a_frozen_display_box(self):
        rules = SafetyRules("camera_01", 1, FakeEventManager(), make_config())
        rules.zone_locks[0].append({
            "track_id": 1,
            "person_key": 1,
            "inside": True,
            "last_seen": 10.0,
            "box": (10, 10, 110, 210),
        })
        rules.track_states[1] = {}

        self.assertEqual(len(rules.get_display_persons(10.0)), 1)
        self.assertEqual(rules.get_display_persons(10.1), [])

    def test_posture_or_long_tracking_gap_restarts_timer(self):
        event_manager = FakeEventManager()
        rules = SafetyRules("camera_01", 1, event_manager, make_config())

        rules.update([make_person()], 0, 1)
        rules.update([make_person(False)], 30, 1)
        rules.update([make_person()], 31, 1)
        for second in range(32, 331):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 330.9, 1)[2], [])
        self.assertEqual(
            rules.update([make_person()], 331, 1)[2][0]["event_type"],
            "HEAD_DOWN_PHONE_VIOLATION",
        )

        rules.update([make_person()], 400, 1)
        rules.update([], 402, 1)
        rules.update([make_person()], 403, 1)
        for second in range(404, 703):
            self.assertEqual(rules.update([make_person()], second, 1)[2], [])
        self.assertEqual(rules.update([make_person()], 702.9, 1)[2], [])
        self.assertEqual(
            rules.update([make_person()], 703, 1)[2][0]["event_type"],
            "HEAD_DOWN_PHONE_VIOLATION",
        )

    def test_violation_is_counted_in_csv_and_has_screenshot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            events = EventManager(output_dir)
            timestamp = events.log(
                "camera_01",
                300,
                "HEAD_DOWN_PHONE_VIOLATION",
                1,
                "Head-down posture with phone sustained for 300 seconds",
            )
            screenshot = events.screenshot(
                "camera_01",
                np.zeros((80, 80, 3), dtype=np.uint8),
                300,
                "HEAD_DOWN_PHONE_VIOLATION",
                {1},
                zone=0,
                timestamp=timestamp,
            )

            csv_path = Path(output_dir) / "cameras" / "camera_01" / "events.csv"
            with csv_path.open(encoding="utf-8", newline="") as event_file:
                rows = list(csv.DictReader(event_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event"], "HEAD_DOWN_PHONE_VIOLATION")
            self.assertTrue(screenshot.is_file())
            self.assertEqual(
                events.list_events()[0]["screenshot"],
                screenshot.name,
            )


if __name__ == "__main__":
    unittest.main()