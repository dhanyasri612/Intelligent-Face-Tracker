from pathlib import Path
import cv2


class VideoReader:

    def __init__(self, video_directory):
        self.video_directory = Path(video_directory)

    def get_videos(self):
        """Return sorted list of Path objects for every .mp4 in the directory."""
        return sorted(self.video_directory.glob("*.mp4"))

    def get_metadata(self, video_path):
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            return {
                "filename": Path(video_path).name,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
            }
        finally:
            capture.release()

    def read(self, video_path):
        """
        Generator that yields frames one at a time.
        Accepts both str and Path objects.
        Releases the capture when exhausted or if an error occurs.
        """
        capture = cv2.VideoCapture(str(video_path))  # str() handles Path objects

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame
        finally:
            capture.release()
