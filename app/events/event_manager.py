from datetime import datetime, timezone


class EventManager:

    def __init__(self, missing_threshold=15):
        self.active_faces = {}
        self.missing_threshold = missing_threshold
        self.unique_visitors = set()

    def process_seen_face(self, face_id, track_id):

        now = datetime.now(timezone.utc)

        if face_id not in self.active_faces:

            self.active_faces[face_id] = {
                "track_id": track_id,
                "last_seen": now,
                "missing_frames": 0,
                "inside": True
            }

            is_new_visitor = face_id not in self.unique_visitors

            self.unique_visitors.add(face_id)

            return {
                "event": "entry",
                "face_id": face_id,
                "is_new_visitor": is_new_visitor,
                "timestamp": now
            }

        state = self.active_faces[face_id]

        state["track_id"] = track_id
        state["last_seen"] = now
        state["missing_frames"] = 0

        return {
            "event": "recognition",
            "face_id": face_id,
            "is_new_visitor": False,
            "timestamp": now
        }

    def process_missing_faces(self, visible_face_ids):

        exit_events = []

        for face_id, state in list(self.active_faces.items()):

            if face_id in visible_face_ids:
                continue

            state["missing_frames"] += 1

            if state["missing_frames"] >= self.missing_threshold:

                now = datetime.now(timezone.utc)

                exit_events.append({
                    "event": "exit",
                    "face_id": face_id,
                    "timestamp": now
                })

                del self.active_faces[face_id]

        return exit_events

    def get_unique_count(self):
        return len(self.unique_visitors)