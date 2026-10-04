import json


class Settings:

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as file:
            self.config = json.load(file)

    @property
    def video_directory(self):
        return self.config["video"]["directory"]

    @property
    def skip_frames(self):
        return self.config["detection"]["skip_frames"]

    @property
    def confidence_threshold(self):
        return self.config["detection"]["confidence_threshold"]

    @property
    def similarity_threshold(self):
        return self.config["recognition"]["similarity_threshold"]

    @property
    def missing_frames_threshold(self):
        # How many processed frames a face must be absent before an exit is fired.
        return self.config.get("tracking", {}).get("missing_frames_threshold", 15)
