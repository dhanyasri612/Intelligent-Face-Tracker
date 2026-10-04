from datetime import datetime, timezone
from pymongo import ASCENDING, IndexModel


class EventRepository:

    def __init__(self, database):
        self.collection = database["events"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        # Index on face_id for quick per-person event lookup.
        # Index on timestamp for chronological queries.
        self.collection.create_indexes([
            IndexModel([("face_id", ASCENDING)], name="events_face_id"),
            IndexModel([("timestamp", ASCENDING)], name="events_timestamp"),
        ])

    def save_event(
        self,
        face_id,
        event_type,
        timestamp=None,
        image_path=None,
        track_id=None,
        video_filename=None,
    ):
        document = {
            "face_id": face_id,
            "event_type": event_type,
            "timestamp": timestamp or datetime.now(timezone.utc),
            "image_path": image_path,
            "track_id": track_id,
            "video_filename": video_filename,
        }
        self.collection.insert_one(document)

    def get_events(self, face_id=None):
        query = {}
        if face_id:
            query["face_id"] = face_id
        return list(
            self.collection.find(query, {"_id": 0}).sort("timestamp", 1)
        )
