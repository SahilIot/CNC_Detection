from statistics import median

from ultralytics import YOLO


class PoseDetector:
    def __init__(
        self,
        model_path,
        device,
        half,
        confidence,
        image_size,
        keypoint_confidence,
        keypoint_delta,
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.half = half
        self.confidence = confidence
        self.image_size = image_size
        self.keypoint_confidence = keypoint_confidence
        self.keypoint_delta = keypoint_delta

    def detect(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf
        if confidences is None:
            return []
        confidences = confidences.cpu().numpy()

        poses = []
        for box, points, point_confidences in zip(boxes, keypoints, confidences):
            x1, y1, x2, y2 = map(int, box)
            poses.append({
                "box": (x1, y1, x2, y2),
                "head_down": self.is_head_down(
                    points,
                    point_confidences,
                    y2 - y1,
                    self.keypoint_confidence,
                    self.keypoint_delta,
                ),
            })
        return poses

    @staticmethod
    def is_head_down(
        keypoints,
        confidences,
        person_height,
        confidence_threshold,
        keypoint_delta,
    ):
        # COCO pose indices: nose 0, eyes 1-2, ears 3-4.
        if person_height <= 0 or len(keypoints) < 5 or len(confidences) < 5:
            return False
        if confidences[0] < confidence_threshold:
            return False

        visible_head_points = [
            keypoints[index][1]
            for index in range(1, 5)
            if confidences[index] >= confidence_threshold
        ]
        if not visible_head_points:
            return False

        return (
            keypoints[0][1] - median(visible_head_points)
            >= person_height * keypoint_delta
        )
