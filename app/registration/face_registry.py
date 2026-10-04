import uuid
import numpy as np


class FaceRegistry:

    def __init__(self, repository, similarity_threshold=0.5):
        self.repository = repository
        self.similarity_threshold = similarity_threshold

    def register(self, embedding, image_path=None):

        face_id = f"FACE_{uuid.uuid4().hex[:8].upper()}"

        self.repository.save_face(
            face_id=face_id,
            embedding=embedding,
            image_path=image_path
        )

        return face_id

    def find_match(self, embedding):

        faces = self.repository.get_all_faces()

        if not faces:
            return None, 0.0

        best_face_id = None
        best_similarity = -1.0

        embedding = np.asarray(embedding)

        for face in faces:

            stored_embedding = np.asarray(
                face["embedding"]
            )

            similarity = float(
                np.dot(embedding, stored_embedding)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_face_id = face["face_id"]

        if best_similarity >= self.similarity_threshold:
            return best_face_id, best_similarity

        return None, best_similarity

    def identify(self, embedding, image_path=None):

        face_id, similarity = self.find_match(
            embedding
        )

        if face_id is not None:
            return face_id, False, similarity

        face_id = self.register(
            embedding,
            image_path
        )

        return face_id, True, 1.0