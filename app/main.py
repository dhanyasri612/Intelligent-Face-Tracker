"""
Main pipeline orchestrator.

Responsibilities:
  - iterate videos
  - for each frame: track → embed → identify → event
  - delegate persistence to repositories
  - delegate local logging to EventLogger
  - print final summary

Nothing business-logic lives here; every concern is handled by its module.
"""

from app.config.settings import Settings
from app.video.video_reader import VideoReader
from app.tracking.face_tracker import FaceTracker
from app.recognition.face_recognizer import FaceRecognizer
from app.registration.face_registry import FaceRegistry
from app.events.event_manager import EventManager
from app.logging.logger import EventLogger

from app.database.mongodb import MongoDB
from app.database.repositories.face_repository import FaceRepository
from app.database.repositories.event_repository import EventRepository

from app.utils.image_utils import crop_face


class FaceTrackingPipeline:

    def __init__(self):
        self.settings = Settings()

        # ── Video source ──────────────────────────────────────────────────
        self.video_reader = VideoReader(self.settings.video_directory)

        # ── Detection + tracking (single YOLO model) ─────────────────────
        # FaceDetector is intentionally NOT initialised here.
        # FaceTracker runs YOLO.track() internally, so loading a second
        # YOLO instance via FaceDetector would waste memory and time.
        self.tracker = FaceTracker(
            model_path="models/yolov11n-face.pt",
            confidence_threshold=self.settings.confidence_threshold,
        )

        # ── Embedding generation ──────────────────────────────────────────
        self.recognizer = FaceRecognizer()

        # ── Database ──────────────────────────────────────────────────────
        self.database = MongoDB()
        self.face_repository = FaceRepository(self.database.database)
        self.event_repository = EventRepository(self.database.database)

        # ── Identity registry ─────────────────────────────────────────────
        self.face_registry = FaceRegistry(
            repository=self.face_repository,
            similarity_threshold=self.settings.similarity_threshold,
        )

        # ── Business logic ────────────────────────────────────────────────
        self.event_manager = EventManager(
            missing_threshold=self.settings.missing_frames_threshold
        )

        # ── Local file logging ────────────────────────────────────────────
        self.event_logger = EventLogger(log_directory="logs")

        # ── Run-wide accumulators ─────────────────────────────────────────
        self._videos_processed = 0
        self._entry_events = 0
        self._exit_events = 0
        self._faces_registered = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        videos = self.video_reader.get_videos()

        if not videos:
            print("No MP4 videos found in", self.settings.video_directory)
            return

        print(f"Found {len(videos)} video(s) to process\n")

        for video_path in videos:
            self._process_video(video_path)

        self._print_summary()
        self.database.close()

    # ──────────────────────────────────────────────────────────────────────
    # Per-video processing
    # ──────────────────────────────────────────────────────────────────────

    def _process_video(self, video_path):
        print(f"Processing: {video_path.name}")

        # track_id → face_id mapping is LOCAL to this video.
        # ByteTrack resets its counter for each new call to YOLO.track()
        # with persist=False between videos, so the same integer track_id
        # in two different videos does NOT mean the same person.
        track_to_face = {}

        # face_id → latest face crop (numpy array).
        # Kept so we have an image to attach to the exit event even though
        # the person is no longer visible in the current frame.
        last_crop: dict = {}

        frame_number = 0
        processed_frames = 0

        for frame in self.video_reader.read(video_path):

            # ── Frame-skipping ────────────────────────────────────────────
            # skip_frames=2 means process 1 in every 3 frames (0, 3, 6 …)
            if frame_number % (self.settings.skip_frames + 1) != 0:
                frame_number += 1
                continue

            processed_frames += 1

            # ── Detection + tracking ──────────────────────────────────────
            tracks = self.tracker.track(frame)

            # face IDs visible in this processed frame
            visible_face_ids: set = set()

            for track in tracks:
                track_id = track["track_id"]
                bbox = track["bbox"]

                # ── Resolve face identity ─────────────────────────────────
                if track_id in track_to_face:
                    # Already resolved for this video session — reuse.
                    face_id = track_to_face[track_id]
                else:
                    # New track: crop → embed → match or register.
                    face_id = self._resolve_identity(
                        frame, bbox, track_id, video_path.name, track_to_face
                    )
                    if face_id is None:
                        # Crop or embedding failed; skip this track.
                        continue

                visible_face_ids.add(face_id)

                # ── Cache latest crop for exit image ──────────────────────
                face_crop = crop_face(frame, bbox)
                if face_crop is not None:
                    last_crop[face_id] = face_crop

                # ── Notify EventManager ───────────────────────────────────
                event = self.event_manager.process_seen_face(face_id, track_id)
                self._handle_event(event, face_crop, track_id, video_path.name)

            # ── Check for exits ───────────────────────────────────────────
            exit_events = self.event_manager.process_missing_faces(visible_face_ids)
            for exit_event in exit_events:
                face_id = exit_event["face_id"]
                # Use the cached crop — there is no current frame for this face.
                cached_crop = last_crop.pop(face_id, None)
                self._handle_event(exit_event, cached_crop, None, video_path.name)

            frame_number += 1

        self._videos_processed += 1
        print(
            f"  → {processed_frames} frames processed, "
            f"{len(track_to_face)} tracks seen\n"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Identity resolution
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_identity(self, frame, bbox, track_id, video_filename, track_to_face):
        """
        Crop → embed → registry lookup.
        Returns face_id on success, None if the crop or embedding fails.
        """
        face_crop = crop_face(frame, bbox)
        if face_crop is None:
            return None

        embedding = self.recognizer.get_embedding(face_crop)
        if embedding is None:
            return None

        face_id, is_new, similarity = self.face_registry.identify(embedding)

        track_to_face[track_id] = face_id

        # Log embedding generation and tracking association.
        self.event_logger.log(
            "embedding_generation",
            face_id=face_id,
            message=f"similarity={similarity:.3f}",
        )
        self.event_logger.log(
            "tracking",
            face_id=face_id,
            message=f"track_id={track_id}",
        )

        if is_new:
            self._faces_registered += 1
            self.event_logger.log("registration", face_id=face_id)
            # Save registration event to MongoDB.
            self.event_repository.save_event(
                face_id=face_id,
                event_type="registration",
                track_id=track_id,
                video_filename=video_filename,
            )

        return face_id

    # ──────────────────────────────────────────────────────────────────────
    # Event handling
    # ──────────────────────────────────────────────────────────────────────

    def _handle_event(self, event, face_crop, track_id, video_filename):
        """
        Dispatch an event dict produced by EventManager to:
          1. EventLogger  (events.log + image file)
          2. EventRepository (MongoDB)
        """
        event_type = event["event"]
        face_id = event["face_id"]
        timestamp = event["timestamp"]

        image_path = None

        # Save face image for entry and exit events.
        if event_type in ("entry", "exit") and face_crop is not None:
            try:
                image_path = self.event_logger.save_image(
                    face_crop, event_type, face_id
                )
            except Exception as exc:
                print(f"  [WARN] Could not save {event_type} image for {face_id}: {exc}")

        # Write to events.log.
        self.event_logger.log(
            event_type,
            face_id=face_id,
            message=f"track_id={track_id}" if track_id is not None else "",
        )

        # Persist to MongoDB.
        try:
            self.event_repository.save_event(
                face_id=face_id,
                event_type=event_type,
                timestamp=timestamp,
                image_path=image_path,
                track_id=track_id,
                video_filename=video_filename,
            )
        except Exception as exc:
            print(f"  [WARN] MongoDB write failed for {event_type}/{face_id}: {exc}")

        # Update run-wide counters.
        if event_type == "entry":
            self._entry_events += 1
        elif event_type == "exit":
            self._exit_events += 1

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────

    def _print_summary(self):
        unique = self.event_manager.get_unique_count()
        print("=" * 44)
        print("       FACE TRACKING COMPLETE")
        print("=" * 44)
        print(f"  Videos processed : {self._videos_processed}")
        print(f"  Unique visitors  : {unique}")
        print(f"  Faces registered : {self._faces_registered}")
        print(f"  Entry events     : {self._entry_events}")
        print(f"  Exit events      : {self._exit_events}")
        print("=" * 44)
