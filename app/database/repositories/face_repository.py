from datetime import datetime, timezone
from pymongo import ASCENDING, IndexModel


class FaceRepository:

    def __init__(self, database):
        self.collection = database["faces"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        # Unique index on face_id — one document per registered person.
        self.collection.create_indexes([
            IndexModel([("face_id", ASCENDING)], unique=True, name="face_id_unique"),
        ])

    def save_face(self, face_id, embedding, image_path=None):
        document = {
            "face_id": face_id,
            "embedding": embedding.tolist(),
            "image_path": image_path,
            "created_at": datetime.now(timezone.utc),
        }
        self.collection.insert_one(document)

    def get_all_faces(self):
        return list(self.collection.find({}, {"_id": 0}))

    def get_face(self, face_id):
        return self.collection.find_one({"face_id": face_id}, {"_id": 0})
