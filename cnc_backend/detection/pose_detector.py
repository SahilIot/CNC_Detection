from pathlib import Path
from statistics import median
from ultralytics import YOLO

class PoseDetector:
    def __init__(self,
        model_path,device,half,confidence,
        image_size,keypoint_confidence,keypoint_delta,):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO pose model weights not found: {model_path}")
        self.model = YOLO(model_path)
        if self.model.task != "pose":
            raise ValueError(f"Head posture detection requires a YOLO pose model; "
                f"{model_path} is task={self.model.task!r}"
            )
        self.device = device
        self.half = half
        self.confidence = confidence
        self.image_size = image_size
        self.keypoint_confidence = keypoint_confidence
        self.keypoint_delta = keypoint_delta

    def track(self, frame, tracker_config, augment=False):
        return self.model.track(frame,persist=True,
            tracker=tracker_config,classes=[0],
            conf=self.confidence,imgsz=self.image_size,
            device=self.device,half=self.half,
            augment=augment,verbose=False,
        )

    @staticmethod
    def head_down_from_keypoints(keypoints, confidences, person_height, confidence_threshold, keypoint_delta,):
        return PoseDetector.is_head_down(keypoints,
            confidences,
            person_height,confidence_threshold,
            keypoint_delta,
        )

    @staticmethod
    def is_head_down(keypoints,confidences,person_height,confidence_threshold,keypoint_delta,):
        # COCO pose indices: nose 0, eyes 1-2, ears 3-4.
        if person_height <= 0 or len(keypoints) < 5 or len(confidences) < 5:
            return None
        if confidences[0] < confidence_threshold:
            return None
        visible_head_points = [keypoints[index][1]
            for index in range(1, 5)
            if confidences[index] >= confidence_threshold]
        if not visible_head_points:
            return None
        return (keypoints[0][1] - median(visible_head_points) >= person_height * keypoint_delta)
