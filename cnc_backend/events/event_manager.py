from pathlib import Path
import csv
import cv2
from datetime import datetime
import hashlib
import time

def format_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    return (f"{int(seconds // 3600):02d}:"
        f"{int((seconds % 3600) // 60):02d}:"
        f"{int(seconds % 60):02d}"
    )
class EventManager:
    def __init__(self, output_dir):
        self.base = Path(output_dir) / "cameras"
        self.base.mkdir(parents=True, exist_ok=True)

    def camera_dir(self, camera_id):
        path = self.base / camera_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self,camera_id,video_seconds,event_name,track_ids="",details="" ):
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M:%S %p")

        print(f"\nSAFETY EVENT\n Camera    : {camera_id}\n Timestamp : {timestamp}\n Event     : {event_name}")

        if track_ids:
            print("Track IDs :", track_ids)
        if details:
            print("Details   :", details)
        path = self.camera_dir(camera_id) / "events.csv"

        if not path.exists():
            path.write_text("timestamp,video_time,event,track_ids,details\n",encoding="utf-8")

        def safe(value):
            return str(value).replace(",", ";")

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{safe(timestamp)},"
                f"{video_seconds:.2f},"
                f"{safe(event_name)},"
                f"{safe(track_ids)},"
                f"{safe(details)}\n"
            )
        return timestamp

    def screenshot(self, camera_id, frame, video_time, event_type, track_ids, zone=None,timestamp=None,):
        # Use the same directory structure as camera_dir()
        camera_dir = self.camera_dir(camera_id)

        screenshot_dir = camera_dir / "screenshots"
        screenshot_dir.mkdir(parents=True,exist_ok=True)
        # format_timestamp() is a standalone function
        timestamp = (timestamp or format_timestamp(video_time)).replace(":", "-")
        # Make sure track_ids is iterable
        if isinstance(track_ids, (int, str)):
            track_ids = {track_ids}

        ids_text = "_".join(str(i) for i in sorted(track_ids, key=str))

        filename = f"{timestamp}_{event_type}_IDs_{ids_text}"
        if zone is not None:
            filename += f"_Zone_{zone + 1}"

        filename += ".jpg"
        path = screenshot_dir / filename

        success = cv2.imwrite(str(path),frame)

        if not success:
            raise RuntimeError(f"cv2.imwrite failed. Could not save screenshot: {path}")

        print(f"SCREENSHOT SAVED: {path}")
        return path

    def cleanup_old_screenshots(self, retention_days=15, now=None):
        cutoff = (time.time() if now is None else now) - retention_days * 86400
        deleted = 0
        for screenshot_dir in self.base.glob("*/screenshots"):
            if not screenshot_dir.is_dir():
                continue
            for screenshot in screenshot_dir.glob("*.jpg"):
                if screenshot.stat().st_mtime < cutoff:
                    screenshot.unlink()
                    deleted += 1
        return deleted

    @staticmethod
    def _event_id(camera_id, row_number, row):
        value = f"{camera_id}:{row_number}:{row.get('timestamp', '')}:"
        value += f"{row.get('video_time', '')}:{row.get('event', '')}:"
        value += f"{row.get('track_ids', '')}:{row.get('details', '')}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def list_events(self, limit=100):
        records = []
        for camera_dir in self.base.glob("*"):
            events_path = camera_dir / "events.csv"
            if not events_path.is_file():
                continue
            with events_path.open("r", encoding="utf-8", newline="") as file:
                for row_number, row in enumerate(csv.DictReader(file)):
                    timestamp = row.get("timestamp", "")
                    event_type = row.get("event", "")
                    screenshot_prefix = timestamp.replace(":", "-")
                    screenshots = sorted(
                        (camera_dir / "screenshots").glob(
                            f"{screenshot_prefix}_{event_type}_*.jpg"
                        )
                    )
                    records.append({
                        "camera_id": camera_dir.name,
                        "event_id": self._event_id(camera_dir.name, row_number, row),
                        "timestamp": timestamp,
                        "video_time": float(row.get("video_time") or 0),
                        "event_type": event_type,
                        "track_ids": row.get("track_ids", ""),
                        "details": row.get("details", ""),
                        "screenshot": screenshots[-1].name if screenshots else None,
                    })
        records.sort(key=lambda event: (event["camera_id"], event["video_time"]), reverse=True)
        return records[:limit]

    def delete_event(self, event_id):
        for camera_dir in self.base.glob("*"):
            events_path = camera_dir / "events.csv"
            if not events_path.is_file():
                continue
            with events_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
            kept = []
            removed = None
            for row_number, row in enumerate(rows):
                if removed is None and self._event_id(camera_dir.name, row_number, row) == event_id:
                    removed = row
                else:
                    kept.append(row)
            if removed is None:
                continue
            with events_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept)
            timestamp = removed.get("timestamp", "").replace(":", "-")
            event_type = removed.get("event", "")
            for screenshot in (camera_dir / "screenshots").glob(f"{timestamp}_{event_type}_*.jpg"):
                screenshot.unlink()
            return True
        return False

    # Event logging, Screenshot capture, camera organization, event retrieval, event deletion
