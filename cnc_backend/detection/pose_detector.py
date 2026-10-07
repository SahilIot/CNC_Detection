from pathlib import Path
from statistics import median
from ultralytics import YOLO

class PoseDetector:
    def __init__(self,
        model_path,device,
        half,confidence,
        image_size,
        keypoint_confidence,
        keypoint_delta,
    ):
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

    def detect_head_down(self, frame, person_box, padding=0.12):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = person_box
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame_width, x2 + pad_x)
        crop_y2 = min(frame_height, y2 + pad_y)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None

        results = self.model.predict(crop,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        box_confidences = result.boxes.conf
        if box_confidences is not None:
            best_index = int(box_confidences.argmax().item())
        else:
            best_index = 0
        if best_index >= len(boxes):
            return None

        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf
        if confidences is None:
            return None
        confidences = confidences.cpu().numpy()
        px1, py1, px2, py2 = boxes[best_index]
        return self.is_head_down(
            keypoints[best_index],
            confidences[best_index],
            py2 - py1,
            self.keypoint_confidence,
            self.keypoint_delta,
        )

    @staticmethod
    def is_head_down(keypoints,confidences,
        person_height,confidence_threshold,
        keypoint_delta,
    ):
        # COCO pose indices: nose 0, eyes 1-2, ears 3-4.
        if person_height <= 0 or len(keypoints) < 5 or len(confidences) < 5:
            return None
        if confidences[0] < confidence_threshold:
            return None

        visible_head_points = [keypoints[index][1]
            for index in range(1, 5)
            if confidences[index] >= confidence_threshold
        ]
        if not visible_head_points:
            return None

        return (keypoints[0][1] - median(visible_head_points) >= person_height * keypoint_delta)
