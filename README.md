# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven system that processes video files, detects and tracks faces, recognises previously-seen people, auto-registers new arrivals, counts unique visitors, and records entry/exit events to local logs and MongoDB Atlas.

---

## Project Overview

The system implements the following pipeline end-to-end:

```
Video file
  ↓
YOLO face detection + ByteTrack tracking
  ↓
Face crop
  ↓
ArcFace (InsightFace buffalo_l) embedding — 512-dim normalised vector
  ↓
Identity matching via cosine similarity
  ↓
Auto-registration of unknown faces
  ↓
Entry / Recognition / Exit event management
  ↓
Local image logs + events.log + MongoDB Atlas
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full component diagram.

| Module | Responsibility |
|---|---|
| `VideoReader` | Opens MP4 files, yields frames |
| `FaceTracker` | YOLO detection + ByteTrack tracking, returns `track_id` + `bbox` |
| `FaceRecognizer` | Generates 512-dim ArcFace embeddings |
| `FaceRegistry` | Cosine similarity matching, auto-registration |
| `EventManager` | Entry / recognition / exit logic, unique visitor set |
| `FaceRepository` | MongoDB `faces` collection |
| `EventRepository` | MongoDB `events` collection |
| `EventLogger` | `logs/events.log` + entry/exit image files |
| `FaceTrackingPipeline` | Orchestrates all components in `app/main.py` |

---

## Setup

### Requirements

- macOS (tested on M1)
- Python 3.11
- MongoDB Atlas account

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd intelligent-face-tracker
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the YOLO face model

Place the model at:

```
models/yolov11n-face.pt
```

The InsightFace `buffalo_l` pack downloads automatically on first run (stored in `~/.insightface/models/`).

### 4. Configure `.env`

Create `.env` in the project root:

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=face_tracker
```

Never commit this file — it is listed in `.gitignore`.

### 5. MongoDB Atlas setup

- Create a free M0 cluster.
- Add your IP address to the Network Access list.
- Create a database user with read/write permissions.
- Copy the connection string into `MONGODB_URI`.

The collections `faces` and `events` are created automatically on first run, along with their indexes.

### 6. Add video files

Place MP4 files in:

```
data/videos/
```

---

## Configuration

All tuneable parameters live in `config.json`:

```json
{
    "video": {
        "input_type": "file",
        "directory": "data/videos"
    },
    "detection": {
        "skip_frames": 2,
        "confidence_threshold": 0.5
    },
    "recognition": {
        "similarity_threshold": 0.6
    },
    "tracking": {
        "missing_frames_threshold": 15
    }
}
```

| Key | Meaning |
|---|---|
| `video.directory` | Folder scanned for `.mp4` files |
| `detection.skip_frames` | Process 1 in every `N+1` frames (2 → process every 3rd frame) |
| `detection.confidence_threshold` | Minimum YOLO detection confidence |
| `recognition.similarity_threshold` | Cosine similarity required to match an existing face (0–1) |
| `tracking.missing_frames_threshold` | Processed frames a face must be absent before an exit event fires |

---

## How to Run

```bash
source .venv/bin/activate
python run.py
```

---

## Output

### Terminal summary

```
Found 3 video(s) to process

Processing: record_20250620_183903.mp4
  → 214 frames processed, 8 tracks seen

Processing: record_20250620_184049.mp4
  → 198 frames processed, 6 tracks seen

============================================
       FACE TRACKING COMPLETE
============================================
  Videos processed : 3
  Unique visitors  : 25
  Faces registered : 25
  Entry events     : 31
  Exit events      : 28
============================================
```

### events.log (sample lines)

```
2025-06-20T18:39:14.221 | registration      | face_id=FACE_3A1B2C3D |
2025-06-20T18:39:14.222 | embedding_generation | face_id=FACE_3A1B2C3D | similarity=1.000
2025-06-20T18:39:14.223 | tracking          | face_id=FACE_3A1B2C3D | track_id=1
2025-06-20T18:39:14.224 | entry             | face_id=FACE_3A1B2C3D | track_id=1
2025-06-20T18:39:47.901 | recognition       | face_id=FACE_3A1B2C3D | track_id=1
2025-06-20T18:40:02.114 | exit              | face_id=FACE_3A1B2C3D |
```

### Entry/exit images

```
logs/
├── entries/
│   └── 2025-06-20/
│       └── FACE_3A1B2C3D_183914_123456.jpg
├── exits/
│   └── 2025-06-20/
│       └── FACE_3A1B2C3D_184002_654321.jpg
└── events.log
```

### Sample MongoDB documents

**faces collection**
```json
{
  "face_id": "FACE_3A1B2C3D",
  "embedding": [0.023, -0.114, ...],
  "image_path": null,
  "created_at": "2025-06-20T18:39:14.221Z"
}
```

**events collection**
```json
{
  "face_id": "FACE_3A1B2C3D",
  "event_type": "entry",
  "timestamp": "2025-06-20T18:39:14.224Z",
  "image_path": "logs/entries/2025-06-20/FACE_3A1B2C3D_183914_123456.jpg",
  "track_id": 1,
  "video_filename": "record_20250620_183903.mp4"
}
```

---

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/test_pipeline_units.py -v
```

The unit tests are self-contained — no GPU, no video, no MongoDB connection required.

---

## Assumptions

- Every input file is a valid H.264 MP4 readable by OpenCV.
- The face model file exists at `models/yolov11n-face.pt`.
- MongoDB Atlas is reachable from the host network.
- A similarity threshold of 0.6 is sufficient for development purposes; it has not been scientifically tuned.
- Track IDs produced by ByteTrack are local to a single video session. They are never treated as persistent identities across videos.

---

## Limitations

- **Single-process, CPU-bound** — no GPU batching. Expect ~3–8 FPS on M1 CPU with `skip_frames=2`.
- **In-memory similarity search** — all face embeddings are loaded from MongoDB into RAM on every new track. Works fine for hundreds of faces; a vector DB (e.g. Pinecone, pgvector) would be needed at scale.
- **No real-time stream support** — designed for offline MP4 files, not RTSP/webcam streams.
- **Threshold sensitivity** — at 0.6, partial occlusion or extreme lighting changes may cause false re-registrations.
- **Exit image quality** — the exit image is the last cached crop before the face disappeared, which may not be the clearest frame.

---

## CPU / GPU Considerations

- The YOLO model runs on CPU by default on M1 (`ctx_id=0` in InsightFace maps to CPU when CUDA is absent).
- For faster inference, MPS (Apple Metal) can be enabled in the YOLO call: `device="mps"`.
- InsightFace does not yet support MPS; it always uses CPU on Apple Silicon.

---

This project is a part of a hackathon run by https://katomaran.com
