import json
import math
from pathlib import Path
import cv2
import numpy as np

class ZoneManager:
    def __init__(self, output_dir, margin_px=60):
        self.zone_dir = Path(output_dir) / "camera_zones"
        self.zone_dir.mkdir(parents=True, exist_ok=True)
        self.margin_px = margin_px

    def file_path(self, camera_id):
        return self.zone_dir / f"{camera_id}_zones.json"

    def save(self, camera_id, zones, width, height):
        data = {"zones": [[[float(x) / width, float(y) / height] for x, y in zone] for zone in zones], "margin_pixels": self.margin_px}
        with open(self.file_path(camera_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, camera_id, width, height):
        path = self.file_path(camera_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            normalized_zones = data.get("zones", [])
            if not normalized_zones:
                return None
            self.validate_normalized(normalized_zones)
            return [
                np.array(
                    [[int(x * width), int(y * height)] for x, y in zone],
                    dtype=np.int32,
                )
                for zone in normalized_zones
            ]
        except Exception as exc:
            print(f"Could not load zones for {camera_id}:", exc)
            return None

    def save_normalized(self, camera_id, zones):
        self.validate_normalized(zones)
        path = self.file_path(camera_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"zones": zones, "margin_pixels": self.margin_px},
                f,
                indent=4,
            )

    @staticmethod
    def validate_normalized(zones):
        if not isinstance(zones, list) or not 1 <= len(zones) <= 20:
            raise ValueError("Number of zones must be between 1 and 20")
        for index, zone in enumerate(zones, start=1):
            if not isinstance(zone, list) or len(zone) < 3:
                raise ValueError(
                    f"Zone {index} must contain at least three points"
                )
            for point in zone:
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        or value > 1
                        for value in point
                    )
                ):
                    raise ValueError(
                        f"Zone {index} coordinates must be normalized "
                        "finite values between 0 and 1"
                    )
            polygon = np.asarray(zone, dtype=np.float32).reshape(-1, 1, 2)
            if abs(cv2.contourArea(polygon)) <= 1e-8:
                raise ValueError(f"Zone {index} must have a non-zero area")

    def masks(self, zones, width, height):
        result = []
        for zone in zones:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [zone], 255)
            result.append(mask)
        return result

    @staticmethod
    def point_inside(mask, x, y):
        h, w = mask.shape
        x = int(x)
        y = int(y)
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return mask[y, x] > 0

    @staticmethod
    def person_zone(zones, x1, y1, x2, y2, margin_px=0):
        # The bottom-center point approximates where the person's feet touch
        # the floor. A person's upper body can overlap a zone while they are
        # standing outside it, so bounding-box overlap is not occupancy.
        # Pick the closest polygon when zones overlap or the footpoint is
        # slightly outside a boundary due to box/keypoint jitter.
        footpoint = ((x1 + x2) / 2, y2)
        closest_zone = None
        closest_distance = -float("inf")
        for index, zone in enumerate(zones or []):
            polygon = np.asarray(zone, dtype=np.int32).reshape(-1, 1, 2)
            distance = cv2.pointPolygonTest(
                polygon, footpoint, measureDist=True
            )
            if distance > closest_distance:
                closest_distance = distance
                closest_zone = index
        if closest_distance >= -max(0, margin_px):
            return closest_zone
        return None

    @staticmethod
    def draw(frame, zones):
        output = frame.copy()
        for index, zone in enumerate(zones):
            overlay = output.copy()
            cv2.fillPoly(overlay, [zone], (255, 255, 0))
            output = cv2.addWeighted(overlay, 0.12, output, 0.88, 0)
            cv2.polylines(output, [zone], True, (255, 255, 0), 3)
            x, y, _, _ = cv2.boundingRect(zone)
            cv2.putText(output, f"ZONE {index + 1}", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return output

    def setup_interactive(self, camera_id, frame, zone_count=None):
        zones = []
        points = []
        window = f"CNC - {camera_id} - Zone Setup"
        state = {"points": points}

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["points"].append((x, y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
        while True:
            display = frame.copy()
            for i, zone in enumerate(zones):
                cv2.polylines(display, [zone], True, (255, 255, 0), 2)
                x, y, _, _ = cv2.boundingRect(zone)
                cv2.putText(display, f"ZONE {i + 1}", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if len(points) > 1:
                cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
            for px, py in points:
                cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(display, f"{camera_id} - ZONE SETUP", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, "Left click = point | ENTER = finish zone", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, "S = save | C = clear | D = delete | Q = cancel", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 13:
                if len(points) >= 3:
                    zones.append(np.array(points, dtype=np.int32))
                    points.clear()
            elif key == ord("c"):
                points.clear()
            elif key == ord("d") and zones:
                zones.pop()
            elif key == ord("s"):
                if len(points) >= 3:
                    zones.append(np.array(points, dtype=np.int32))
                    points.clear()
                if zones and (zone_count is None or len(zones) == zone_count):
                    self.save(camera_id, zones, frame.shape[1], frame.shape[0])
                    cv2.destroyWindow(window)
                    return zones
            elif key == ord("q"):
                cv2.destroyWindow(window)
                return None
