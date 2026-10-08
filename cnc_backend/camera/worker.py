import threading # Allows each camers worker to run independently
import time # Measures processing FPS and add small waits
import cv2 # draws bounding boxes and event annotations

from detection.phone_detector import PhoneDetector
from detection.pose_detector import PoseDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager
from safety.rules import SafetyRules
from display.renderer import Renderer

# brain for one camera ,processes one camera
# takes frames provided by reader.py and performs the processing pipeline
class CameraWorker(threading.Thread):
    def __init__(self, camera_id, source, config, device, half, phone_detector, pose_detector, event_manager, zone_manager):
        super().__init__(daemon=True, name=f"worker-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.device = device
        self.half = half
        self.phone_detector = phone_detector
        self.pose_detector = pose_detector
        self.event_manager = event_manager
        self.zone_manager = zone_manager
        self.renderer = Renderer(config.display_width)
        self.reader = None
        self.zones = None
        self.zone_masks = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.safety = None
        self.running = True
        self.latest_raw = None
        self.latest_frame_number = -1
        self.latest_video_time = 0.0
        self.latest_persons = []
        self.latest_inside = {}
        self.head_down_states = {}
        self.phone_states = {}
        self.processed_sequence = 0
        self.state_lock = threading.Lock()
        self.fps_counter_start = time.perf_counter()
        self.processed_frames = 0
        self.processing_fps = 0.0
        self.startup_ready = threading.Event()
        self.error = None

    def prepare(self): #prepares one camera before the worker starts processing frames
        from camera.reader import CameraReader
        self.reader = CameraReader(self.camera_id, self.source)
        self.width, self.height, self.fps = self.reader.width, self.reader.height, self.reader.fps
        self.zones = self.zone_manager.load(self.camera_id, self.width, self.height)
        expected_zones = getattr(self.config, "zone_count", None)
        if self.zones is not None and expected_zones is not None and len(self.zones) != expected_zones:
            self.reader.stop()
            raise RuntimeError(
                f"Saved zone count ({len(self.zones)}) does not match "
                f"configured zone count ({expected_zones})"
            )
        self.reader.start()
        frame = None
        for _ in range(300):
            frame, _, _ = self.reader.get_latest()
            if frame is not None:
                break
            time.sleep(0.01)
        if frame is None:
            self.reader.stop()
            raise RuntimeError(f"{self.camera_id} did not produce a frame")
        if self.zones is None:
            self.reader.stop()
            raise RuntimeError("Zones must be defined in the dashboard before monitoring starts")
        self.zone_masks = self.zone_manager.masks(self.zones, self.width, self.height)
        self.safety = SafetyRules(self.camera_id, len(self.zones), self.event_manager, self.config)
        self.latest_raw = frame
        print(f"{self.camera_id}: {self.width}x{self.height} @ {self.fps:.2f} FPS | {len(self.zones)} zone(s)")
        return True

    def get_display(self):
        with self.state_lock:
            if self.latest_raw is None:
                return None
            frame = self.latest_raw.copy()
            video_time = self.latest_video_time
            persons = [p.copy() for p in self.latest_persons]
            inside = {k: set(v) for k, v in self.latest_inside.items()}
            processing_fps = self.processing_fps
        return self.renderer.render(
            frame, self.camera_id, video_time, self.zones,
            persons, inside, self.safety, processing_fps,
        )

    def redraw_zones(self):
        frame, _, _ = self.reader.get_latest()
        if frame is None:
            return
        zones = self.zone_manager.setup_interactive(self.camera_id, frame)
        if zones is not None:
            self.zones = zones
            self.zone_masks = self.zone_manager.masks(zones, self.width, self.height)
            self.safety.reset_zones(len(zones))

    def _extract_persons(self, results, video_time):
        persons = []
        if not results or len(results) == 0:
            return persons
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or len(boxes) == 0 or keypoints is None:
            return persons
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
        pose_points = keypoints.xy.cpu().numpy()
        pose_confidences = (
            keypoints.conf.cpu().numpy()
            if keypoints.conf is not None
            else None
        )
        for index, (box, track_id, confidence) in enumerate(zip(xyxy, ids, confs)):
            # Track IDs can be absent briefly while BoT-SORT initializes.
            # Negative temporary IDs let SafetyRules retain the detection and
            # transfer its state when a real tracker ID appears.
            if track_id is None:
                track_id = -(index + 1)
            x1, y1, x2, y2 = map(int, box)
            point_confidences = (
                pose_confidences[index]
                if pose_confidences is not None
                else [0.0] * len(pose_points[index])
            )
            pose_keypoints = [
                (float(point[0]), float(point[1]), float(point_confidence))
                for point, point_confidence in zip(
                    pose_points[index], point_confidences
                )
            ]
            head_down = self.pose_detector.head_down_from_keypoints(
                pose_points[index],
                point_confidences,
                y2 - y1,
                self.config.head_pose_keypoint_confidence,
                self.config.head_pose_keypoint_delta,
            )
            if head_down is not None:
                self.head_down_states[track_id] = (head_down, video_time)
            last_head_down, observed_at = self.head_down_states.get(
                track_id, (False, video_time)
            )
            persons.append({
                "id": int(track_id),
                "x1": x1,"y1": y1,
                "x2": x2,"y2": y2,
                "zone": self.zone_manager.person_zone(self.zone_masks, x1, y1, x2, y2),
                "confidence": float(confidence),
                "head_down": (
                    last_head_down
                    and video_time - observed_at
                    <= self.config.head_pose_hold_seconds
                ),
                "pose_keypoints": pose_keypoints,
            })
        active_ids = {person["id"] for person in persons}
        for track_id, (_, observed_at) in list(self.head_down_states.items()):
            if (
                track_id not in active_ids
                and video_time - observed_at > self.config.track_grace_seconds
            ):
                self.head_down_states.pop(track_id, None)
        return persons

    def _check_phones(self, frame, persons, video_time, frame_number):
        self.processed_sequence += 1
        detection_interval = max(1, self.config.phone_detection_interval)
        should_detect = (
            (self.processed_sequence - 1) % detection_interval == 0
        )
        active_ids = set()

        for person in persons:
            track_id = person["id"]
            zone = person["zone"]
            active_ids.add(track_id)
            if zone is None:
                self.phone_states.pop(track_id, None)
                person["phone_boxes"] = []
                person["phone_detected"] = False
                continue

            if should_detect:
                boxes = self.phone_detector.detect_in_person(
                    frame,
                    person["x1"],
                    person["y1"],
                    person["x2"],
                    person["y2"],
                )
                boxes = [
                    box for box in boxes
                    if self._phone_belongs_to_person(box, person)
                ]
                self.phone_states[track_id] = (
                    boxes,
                    self.processed_sequence,
                    video_time,
                )

            boxes, checked_sequence, _ = self.phone_states.get(
                track_id, ([], -1, video_time)
            )
            person["phone_boxes"] = boxes
            person["phone_detected"] = (
                bool(boxes)
                and self.processed_sequence - checked_sequence
                < detection_interval
            )

        for track_id, (_, _, checked_at) in list(self.phone_states.items()):
            if (
                track_id not in active_ids
                and video_time - checked_at > self.config.track_grace_seconds
            ):
                self.phone_states.pop(track_id, None)

    @staticmethod
    def _phone_belongs_to_person(phone_box, person):
        phone_center_x = (phone_box["x1"] + phone_box["x2"]) / 2
        phone_center_y = (phone_box["y1"] + phone_box["y2"]) / 2
        return (
            person["x1"] <= phone_center_x <= person["x2"]
            and person["y1"] <= phone_center_y <= person["y2"]
        )

    def _process_one(self, frame, frame_number, video_time):
        try:
            results = self.pose_detector.track(
                frame,
                self.config.person_tracker_config,
                self.config.person_use_augment,
            )
            persons = self._extract_persons(results, video_time)
            self._check_phones(frame, persons, video_time, frame_number)
            persons, inside, events = self.safety.update(persons,video_time,len(self.zones))

            for event in events:
                screenshot_frame = frame
                if event["event_type"] == "HEAD_DOWN_PHONE_VIOLATION":
                    screenshot_frame = ZoneManager.draw(frame.copy(), self.zones)
                    event_track_ids = event["track_ids"]
                    for person in persons:
                        if person["id"] not in event_track_ids:
                            continue
                        cv2.rectangle(
                            screenshot_frame,
                            (person["x1"], person["y1"]),
                            (person["x2"], person["y2"]),
                            (0, 0, 255),
                            3,
                        )
                        cv2.putText(
                            screenshot_frame,
                            f"HEAD DOWN + PHONE VIOLATION | ID {person['id']}",
                            (person["x1"], max(25, person["y1"] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        Renderer.draw_pose(
                            screenshot_frame,
                            person.get("pose_keypoints"),
                            self.config.head_pose_keypoint_confidence,
                        )
                        for phone_box in person.get("phone_boxes", []):
                            cv2.rectangle(
                                screenshot_frame,
                                (phone_box["x1"], phone_box["y1"]),
                                (phone_box["x2"], phone_box["y2"]),
                                (0, 0, 255),
                                2,
                            )
                self.event_manager.screenshot(
                    self.camera_id,
                    screenshot_frame,
                    video_time,
                    event["event_type"],
                    event["track_ids"],
                    event.get("zone"),
                    timestamp=event.get("timestamp"),
                )
            display_persons = self.safety.get_display_persons(video_time)

            with self.state_lock:
                self.latest_persons = [p.copy()
                    for p in display_persons
                ]
                self.latest_inside = { k: set(v)
                    for k, v in inside.items()
                }
                self.latest_raw = frame.copy()
                self.latest_video_time = video_time
                self.latest_frame_number = frame_number

                self.processed_frames += 1
                elapsed = time.perf_counter() - self.fps_counter_start

                if elapsed >= 1.0:
                    self.processing_fps = self.processed_frames / elapsed

                    self.processed_frames = 0
                    self.fps_counter_start = time.perf_counter()

        except Exception as exc:
            self.error = exc
            self.running = False

            print(f"[{self.camera_id}] processing error: {exc}")

    def run(self):
        try:
            if self.safety is None:
                return
            last_frame = -1
            while self.running and not self.reader.stop_event.is_set():
                frame, frame_number, video_time = self.reader.get_latest()
                if frame is None:
                    if self.reader.finished:
                        break
                    time.sleep(0.002)
                    continue
                if frame_number == last_frame:
                    if self.reader.finished:
                        break
                    time.sleep(0.001)
                    continue
                last_frame = frame_number
                self._process_one(frame, frame_number, video_time)
                if self.reader.finished and frame_number >= self.reader.current_frame_number - 1:
                    break
        finally:
            self.running = False
            if self.reader is not None:
                self.reader.stop()

#  Worker gets the frame from Reader and check frame number has changed