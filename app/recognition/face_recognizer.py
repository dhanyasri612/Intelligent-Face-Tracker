import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        self.recognition_model = self.app.models["recognition"]

    def get_embedding(self, face_image):

        if face_image is None or face_image.size == 0:
            return None

        # ArcFace expects a reasonably sized face image.
        face_image = cv2.resize(
            face_image,
            (112, 112)
        )

        # Generate embedding directly using ArcFace.
        embedding = self.recognition_model.get_feat(
            face_image
        )

        embedding = embedding.flatten()

        # Normalize embedding.
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def similarity(self, embedding1, embedding2):

        embedding1 = np.asarray(embedding1)
        embedding2 = np.asarray(embedding2)

        return float(
            np.dot(embedding1, embedding2)
        )