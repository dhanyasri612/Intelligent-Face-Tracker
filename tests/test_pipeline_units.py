"""
Unit tests for the Intelligent Face Tracker pipeline.

All tests are isolated — no video files, no YOLO model, no MongoDB
connection required. Heavy dependencies are replaced with lightweight
fakes or unittest.mock stubs so the suite runs in < 1 second.

Tests covered
─────────────
1.  crop_face returns a valid numpy array for a good bbox
2.  crop_face returns None when the bbox is out-of-bounds / zero-area
3.  FaceRegistry.identify registers a new face when the DB is empty
4.  FaceRegistry.identify returns the existing face_id on a match
5.  EventManager fires an entry event on first observation
6.  EventManager fires a recognition event on repeated observation
7.  EventManager fires an exit event after missing_threshold frames
8.  Re-entry does NOT increase the unique visitor count
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────
from app.utils.image_utils import crop_face
from app.registration.face_registry import FaceRegistry
from app.events.event_manager import EventManager


# ═══════════════════════════════════════════════════════════════════════
# 1 & 2 — crop_face
# ═══════════════════════════════════════════════════════════════════════

class TestCropFace(unittest.TestCase):

    def setUp(self):
        # 100 × 100 black image
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_valid_crop_returns_array(self):
        """A bbox fully inside the frame produces a non-empty numpy array."""
        crop = crop_face(self.frame, [10, 10, 50, 50])
        self.assertIsNotNone(crop)
        self.assertIsInstance(crop, np.ndarray)
        self.assertGreater(crop.size, 0)

    def test_crop_shape_matches_bbox(self):
        """The returned crop dimensions match the requested bbox."""
        crop = crop_face(self.frame, [10, 20, 40, 60])
        # height = 60-20=40, width = 40-10=30
        self.assertEqual(crop.shape[:2], (40, 30))

    def test_zero_area_bbox_returns_none(self):
        """x1 == x2 (zero width) → None."""
        crop = crop_face(self.frame, [30, 30, 30, 60])
        self.assertIsNone(crop)

    def test_out_of_bounds_bbox_clipped_correctly(self):
        """A bbox that extends beyond the frame edges is clipped, not None."""
        # bbox wider than the 100px frame
        crop = crop_face(self.frame, [80, 80, 150, 150])
        self.assertIsNotNone(crop)
        # clipped to 100×100 frame → 20×20
        self.assertEqual(crop.shape[:2], (20, 20))


# ═══════════════════════════════════════════════════════════════════════
# 3 & 4 — FaceRegistry
# ═══════════════════════════════════════════════════════════════════════

def _unit_vector(seed=42, dim=512):
    """Reproducible unit-norm embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestFaceRegistry(unittest.TestCase):

    def _make_registry(self, existing_faces=None, threshold=0.6):
        """Build a FaceRegistry backed by a mock repository."""
        repo = MagicMock()
        repo.get_all_faces.return_value = existing_faces or []
        registry = FaceRegistry(repository=repo, similarity_threshold=threshold)
        return registry, repo

    def test_registers_new_face_when_db_empty(self):
        """With an empty database every embedding creates a new face."""
        registry, repo = self._make_registry(existing_faces=[])
        embedding = _unit_vector(seed=1)

        face_id, is_new, similarity = registry.identify(embedding)

        self.assertTrue(is_new)
        self.assertTrue(face_id.startswith("FACE_"))
        repo.save_face.assert_called_once()

    def test_returns_existing_face_on_match(self):
        """An embedding very close to a stored one is recognised, not re-registered."""
        stored_embedding = _unit_vector(seed=7)
        existing = [{"face_id": "FACE_EXISTING", "embedding": stored_embedding.tolist()}]
        registry, repo = self._make_registry(existing_faces=existing, threshold=0.6)

        # Slightly perturb the embedding — cosine similarity will still be > 0.6
        noise = np.random.default_rng(99).standard_normal(512).astype(np.float32) * 0.05
        query = stored_embedding + noise
        query /= np.linalg.norm(query)

        face_id, is_new, similarity = registry.identify(query)

        self.assertFalse(is_new)
        self.assertEqual(face_id, "FACE_EXISTING")
        self.assertGreaterEqual(similarity, 0.6)
        repo.save_face.assert_not_called()

    def test_registers_new_face_when_similarity_below_threshold(self):
        """An embedding orthogonal to all stored ones triggers registration."""
        stored = _unit_vector(seed=1)
        existing = [{"face_id": "FACE_A", "embedding": stored.tolist()}]
        registry, repo = self._make_registry(existing_faces=existing, threshold=0.6)

        # Build a vector with near-zero dot product against `stored`
        orthogonal = np.zeros(512, dtype=np.float32)
        orthogonal[0] = stored[1]
        orthogonal[1] = -stored[0]
        orthogonal /= np.linalg.norm(orthogonal)

        face_id, is_new, _ = registry.identify(orthogonal)

        self.assertTrue(is_new)
        self.assertNotEqual(face_id, "FACE_A")
        repo.save_face.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# 5, 6, 7, 8 — EventManager
# ═══════════════════════════════════════════════════════════════════════

class TestEventManager(unittest.TestCase):

    def setUp(self):
        # Use a short threshold so exit tests don't need many loop iterations
        self.manager = EventManager(missing_threshold=3)

    def test_entry_event_on_first_observation(self):
        """First time a face_id is seen → entry event."""
        event = self.manager.process_seen_face("FACE_001", track_id=1)
        self.assertEqual(event["event"], "entry")
        self.assertEqual(event["face_id"], "FACE_001")
        self.assertTrue(event["is_new_visitor"])

    def test_recognition_event_on_repeated_observation(self):
        """Subsequent frames with the same face_id → recognition event."""
        self.manager.process_seen_face("FACE_001", track_id=1)  # entry
        event = self.manager.process_seen_face("FACE_001", track_id=1)
        self.assertEqual(event["event"], "recognition")
        self.assertFalse(event["is_new_visitor"])

    def test_exit_event_after_missing_threshold_frames(self):
        """Face absent for missing_threshold processed frames → exit event."""
        self.manager.process_seen_face("FACE_001", track_id=1)

        exit_events = []
        # Simulate the face being absent for exactly `missing_threshold` cycles
        for _ in range(self.manager.missing_threshold):
            exit_events = self.manager.process_missing_faces(visible_face_ids=set())

        self.assertEqual(len(exit_events), 1)
        self.assertEqual(exit_events[0]["event"], "exit")
        self.assertEqual(exit_events[0]["face_id"], "FACE_001")
        # Face should no longer be active
        self.assertNotIn("FACE_001", self.manager.active_faces)

    def test_reentry_does_not_increase_unique_count(self):
        """A face that exits and re-enters must not inflate unique visitor count."""
        # Entry
        self.manager.process_seen_face("FACE_001", track_id=1)
        self.assertEqual(self.manager.get_unique_count(), 1)

        # Exit
        for _ in range(self.manager.missing_threshold):
            self.manager.process_missing_faces(visible_face_ids=set())

        self.assertEqual(self.manager.get_unique_count(), 1)

        # Re-entry with a NEW track_id (simulates a new ByteTrack session)
        event = self.manager.process_seen_face("FACE_001", track_id=99)
        self.assertEqual(event["event"], "entry")
        self.assertFalse(event["is_new_visitor"])  # not a NEW visitor
        self.assertEqual(self.manager.get_unique_count(), 1)  # count still 1

    def test_two_different_faces_counted_separately(self):
        """Two distinct face_ids each increment the unique visitor count."""
        self.manager.process_seen_face("FACE_001", track_id=1)
        self.manager.process_seen_face("FACE_002", track_id=2)
        self.assertEqual(self.manager.get_unique_count(), 2)

    def test_missing_faces_not_exited_before_threshold(self):
        """Faces missing for fewer than threshold frames must NOT generate exit events."""
        self.manager.process_seen_face("FACE_001", track_id=1)

        # One frame short of the threshold
        for _ in range(self.manager.missing_threshold - 1):
            exits = self.manager.process_missing_faces(visible_face_ids=set())
            self.assertEqual(exits, [])

        # Face should still be active
        self.assertIn("FACE_001", self.manager.active_faces)


if __name__ == "__main__":
    unittest.main(verbosity=2)
