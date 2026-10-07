import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent
load_dotenv(project_root.parent / ".env")
load_dotenv(project_root / ".env", override=True)

data_dir = project_root / "data"
videos_dir = data_dir / "videos"
models_dir = project_root / "models"
outputs_dir = data_dir / "outputs"

for folder in (videos_dir, models_dir, outputs_dir):
    folder.mkdir(parents=True, exist_ok=True)

def resolve_path(value, default):
    raw = os.getenv(value, default)

    path = Path(raw).expanduser()

    if path.is_absolute():
        return path

    return project_root / path

model_path = resolve_path("MODEL_PATH",str(models_dir / "yolo11n.pt"))
phone_model_path = resolve_path("PHONE_MODEL_PATH",str(model_path))

head_pose_model_path = resolve_path(
    "HEAD_POSE_MODEL_PATH",
    str(models_dir / "yolo11n-pose.pt"),
)

def resolve_camera_source(value):
    value = value.strip()
    if not value:
        return ""
    # Preserve network URLs without converting them into Path objects.
    if value.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return value
    # Resolve local video paths.
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(project_root / path)

cameras = {}

for index in range(1, 5):
    key = f"CAMERA_{index:02d}"
    value = os.getenv(key, "").strip()
    if value:
        source = resolve_camera_source(value)
        cameras[f"camera_{index:02d}"] = source

if not cameras:
    for index in range(1, 5):
        key = f"CAMERA_SOURCE_{index:02d}"
        value = os.getenv(key, "").strip()

        if value:
            source = resolve_camera_source(value)
            cameras[f"camera_{index:02d}"] = source

person_model_path = str(model_path)

person_confidence = float(os.getenv("PERSON_CONFIDENCE", "0.40"))

phone_confidence = float(os.getenv("PHONE_CONFIDENCE", "0.50"))

head_pose_confidence = float(os.getenv("HEAD_POSE_CONFIDENCE", "0.35"))

head_pose_keypoint_confidence = float(
    os.getenv("HEAD_POSE_KEYPOINT_CONFIDENCE", "0.25")
)

head_pose_keypoint_delta = float(
    os.getenv("HEAD_POSE_KEYPOINT_DELTA", "0.05")
)

person_image_size = int(os.getenv("PERSON_IMAGE_SIZE", "640"))

phone_image_size = int(os.getenv("PHONE_IMAGE_SIZE", "512"))

head_pose_image_size = int(os.getenv("HEAD_POSE_IMAGE_SIZE", "640"))

person_use_augment = (os.getenv("PERSON_USE_AUGMENT", "false").strip().lower()== "true")

person_detection_interval = 1

phone_detection_interval = int(os.getenv("PHONE_DETECTION_INTERVAL", "3"))

head_pose_detection_interval = int(
    os.getenv("HEAD_POSE_DETECTION_INTERVAL", "3")
)

head_pose_hold_seconds = float(
    os.getenv("HEAD_POSE_HOLD_SECONDS", "1.5")
)

head_down_violation_seconds = float(
    os.getenv("HEAD_DOWN_VIOLATION_SECONDS", "120")
)

phone_class_id = int(os.getenv("PHONE_CLASS_ID", "67"))

person_tracker_config = os.getenv("PERSON_TRACKER_CONFIG","botsort.yaml")

_tracker_path = Path(person_tracker_config)

if (not _tracker_path.is_absolute() and (project_root / person_tracker_config).exists()):
    person_tracker_config = str(project_root / person_tracker_config)

multiple_person_limit_seconds = 120

multiple_person_cooldown_seconds = 1800

track_id_switch_max_seconds = 10.0

track_id_switch_center_ratio = 1.0

track_id_switch_iou_threshold = 0.10

absence_limit_seconds = 300

zone_margin_px = 60

track_grace_seconds = float(os.getenv("TRACK_GRACE_SECONDS", "1.5"))

inside_grace_seconds = float(os.getenv("INSIDE_GRACE_SECONDS", "2.5"))

display_width = 900

multiple_limit_seconds = 120

duplicate_iou_threshold = 0.65

duplicate_center_ratio = 0.25

model_path = str(model_path)

phone_model_path = str(phone_model_path)

person_model_path = str(person_model_path)

outputs_dir = Path(outputs_dir)
print("\nCAMERA CONFIGURATION")
for camera_id, source in cameras.items():
    print(camera_id, "=>", source)