import cv2

from app.recognition.face_recognizer import FaceRecognizer
from app.utils.image_utils import crop_face
from app.tracking.face_tracker import FaceTracker


VIDEO_PATH = "data/videos/video_sample1.mp4"
MODEL_PATH = "models/yolov11n-face.pt"


recognizer = FaceRecognizer()
tracker = FaceTracker(MODEL_PATH, confidence_threshold=0.5)

capture = cv2.VideoCapture(VIDEO_PATH)

if not capture.isOpened():
    raise RuntimeError("Could not open video")

print("Video opened successfully")

frame_number = 0
embeddings = {}

while True:
    success, frame = capture.read()

    if not success:
        break

    # Process every 10th frame for this experiment
    if frame_number % 10 != 0:
        frame_number += 1
        continue

    tracks = tracker.track(frame)

    for track in tracks:
        track_id = track["track_id"]
        bbox = track["bbox"]

        face_crop = crop_face(frame, bbox)

        if face_crop is None:
            continue

        embedding = recognizer.get_embedding(face_crop)

        if embedding is None:
            continue

        embeddings.setdefault(track_id, []).append(embedding)

    frame_number += 1

capture.release()

print("\nCollected embeddings:")

for track_id, values in embeddings.items():
    print(
        f"Track {track_id}: "
        f"{len(values)} embeddings"
    )

print("\nSame-track similarity:")

for track_id, values in embeddings.items():

    if len(values) < 2:
        continue

    similarities = []

    for i in range(len(values)):
        for j in range(i + 1, len(values)):

            similarity = recognizer.similarity(
                values[i],
                values[j]
            )

            similarities.append(similarity)

    average = sum(similarities) / len(similarities)
    minimum = min(similarities)
    maximum = max(similarities)

    print(
        f"Track {track_id}: "
        f"min={minimum:.4f}, "
        f"avg={average:.4f}, "
        f"max={maximum:.4f}"
    )