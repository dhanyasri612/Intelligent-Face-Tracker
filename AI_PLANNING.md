# AI Planning Document — Intelligent Face Tracker

## Problem Decomposition

The system must answer three questions for every processed frame:

1. **Who is this?** — identity matching via face embeddings
2. **Have I seen them before?** — registry lookup
3. **What just happened?** — entry, recognition, or exit event

These map directly to the three core modules: `FaceRecognizer`, `FaceRegistry`, and `EventManager`.

---

## Detection Strategy

**Model:** YOLOv11n-face (`models/yolov11n-face.pt`)

- Chosen for speed on CPU — the "nano" variant minimises inference time without sacrificing recall for frontal faces.
- Confidence threshold (`0.5`) filters low-quality detections before any embedding is attempted.
- Frame skipping (`skip_frames=2`) means only every 3rd frame is processed, reducing CPU load by ~67% with minimal tracking quality loss at typical walking speeds.

**Why not a separate FaceDetector in the pipeline?**
`FaceTracker` already calls `YOLO.track()` internally, which performs detection as part of tracking. Running a second `FaceDetector` instance would load the same model twice and run inference twice per frame — wasted work eliminated by design.

---

## Tracking Strategy

**Algorithm:** ByteTrack (via Ultralytics `bytetrack.yaml`)

- ByteTrack maintains tracklets using Kalman filtering + IoU association.
- `persist=True` keeps tracklet state across consecutive `track()` calls on the same model instance.
- Each `track_id` is a **temporary** integer assigned by ByteTrack within a single video session.
- **Critical:** `track_id` is reset between videos. The pipeline resets `track_to_face` at the start of every video to prevent cross-video track ID collisions.

---

## Recognition Strategy

**Model:** InsightFace `buffalo_l` → ArcFace recognition head

- Input: 112×112 RGB face crop.
- Output: 512-dimensional embedding vector.
- Embeddings are L2-normalised so cosine similarity reduces to a dot product.
- Similarity = `np.dot(embedding_a, embedding_b)` — values range from -1 to 1; > 0.6 is treated as the same person.

**Why ArcFace?**
ArcFace uses additive angular margin loss, producing highly discriminative embeddings with tight intra-class and wide inter-class angular separation. It consistently outperforms standard softmax-trained models on face verification benchmarks.

---

## Registration Strategy

**Trigger:** `FaceRegistry.identify()` is called for every new `track_id`.

1. Load all stored embeddings from MongoDB (`faces` collection).
2. Compute cosine similarity against every stored embedding.
3. If best similarity ≥ threshold → return existing `face_id`, `is_new=False`.
4. Otherwise → generate a new `FACE_<8-char hex>` ID, persist embedding to MongoDB, return `is_new=True`.

**Identity persistence:**
Once a `track_id` is resolved to a `face_id`, the mapping is cached in `track_to_face` for the rest of the video. Embedding inference only happens once per new track — not on every frame.

---

## Event Strategy

Events are managed by `EventManager` which tracks active faces in memory:

| Trigger | Event | Action |
|---|---|---|
| `face_id` seen, not in `active_faces` | `entry` | Add to `active_faces`, add to `unique_visitors` set |
| `face_id` seen, already in `active_faces` | `recognition` | Reset `missing_frames` counter |
| `face_id` absent for ≥ `missing_threshold` processed frames | `exit` | Remove from `active_faces` |

**Missing-frame logic:**
`process_missing_faces(visible_face_ids)` is called once per processed frame. Any active face not in `visible_face_ids` has its `missing_frames` counter incremented. This gracefully handles momentary occlusion without premature exit events.

---

## Re-identification Strategy

When a person re-enters the scene:

1. ByteTrack assigns a **new** `track_id` (the old one is gone).
2. The new track is not in `track_to_face` → embedding is generated.
3. The embedding matches the stored one → same `face_id` returned, `is_new=False`.
4. `EventManager.process_seen_face()` fires an `entry` event (person is back) but `is_new_visitor=False` because `face_id` is already in `unique_visitors`.
5. Unique visitor count **does not increase**.

---

## Database Strategy

**MongoDB Atlas** (free M0 tier sufficient for MVP)

Collections:
- `faces` — one document per registered person; stores `face_id`, `embedding` (list), `image_path`, `created_at`
- `events` — one document per event; stores `face_id`, `event_type`, `timestamp`, `image_path`, `track_id`, `video_filename`

Indexes created on startup (`_ensure_indexes()`):
- `faces.face_id` — unique, prevents duplicate registrations
- `events.face_id` — fast per-person event queries
- `events.timestamp` — chronological event queries

**Scale consideration:** For hundreds of faces, loading all embeddings for every new track is acceptable. For thousands, switch to a vector index (MongoDB Atlas Vector Search, Pinecone, or pgvector).

---

## Error Handling

| Failure | Handling |
|---|---|
| Invalid/corrupt video | `RuntimeError` from `VideoReader`, caught by pipeline, skips video |
| Failed frame read | Generator ends naturally; no crash |
| Invalid face crop (zero area) | `crop_face()` returns `None`; pipeline skips the track |
| Failed embedding | `get_embedding()` returns `None`; pipeline skips the track |
| MongoDB write failure | `try/except` in `_handle_event()`; logs a warning, continues |
| MongoDB connection failure | `RuntimeError` raised at startup before any video processing |
| Empty face database | `FaceRegistry.find_match()` returns `(None, 0.0)`; registration proceeds normally |
| No detections in a frame | `tracker.track()` returns `[]`; frame is skipped cleanly |

The guiding principle: **never crash because one face crop is bad**. The pipeline must continue processing remaining tracks and frames.

---

## Performance Considerations

- **Frame skipping** (`skip_frames=2`) — primary lever for throughput vs. accuracy trade-off.
- **Single YOLO instance** — `FaceTracker` owns the only loaded YOLO model.
- **Embedding cache via `track_to_face`** — ArcFace inference runs once per new track, not once per frame.
- **Generator-based frame reading** — frames are decoded one at a time; the full video is never loaded into RAM.
- **InsightFace CPU mode** — on M1, InsightFace runs on CPU. YOLO can use MPS (`device="mps"`) for faster detection if needed.
