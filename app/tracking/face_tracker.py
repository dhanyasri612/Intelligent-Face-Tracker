from ultralytics import YOLO


class FaceTracker:

    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            verbose=False
        )

        tracks = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for box in boxes:

                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                tracks.append({
                    "track_id": int(box.id[0]),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0])
                })

        return tracks