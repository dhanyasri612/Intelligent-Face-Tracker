from ultralytics import YOLO


class FaceDetector:

    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence
                })

        return detections