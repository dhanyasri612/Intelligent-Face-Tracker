from pathlib import Path
from datetime import datetime


class EventLogger:

    def __init__(self, log_directory="logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.events_file = self.log_directory / "events.log"

    def log(self, event_type, face_id=None, message=""):

        timestamp = datetime.now().isoformat()

        log_message = (
            f"{timestamp} | "
            f"{event_type} | "
            f"face_id={face_id} | "
            f"{message}\n"
        )

        with open(self.events_file, "a") as file:
            file.write(log_message)

    def save_image(self, image, event_type, face_id):

        date_folder = datetime.now().strftime("%Y-%m-%d")

        if event_type == "entry":
            folder = self.log_directory / "entries" / date_folder
        elif event_type == "exit":
            folder = self.log_directory / "exits" / date_folder
        else:
            folder = self.log_directory / event_type / date_folder

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%H%M%S_%f"
        )

        filename = (
            f"{face_id}_{timestamp}.jpg"
        )

        image_path = folder / filename

        # OpenCV will be imported only when needed
        import cv2

        success = cv2.imwrite(
            str(image_path),
            image
        )

        if not success:
            raise RuntimeError(
                f"Failed to save image: {image_path}"
            )

        return str(image_path)