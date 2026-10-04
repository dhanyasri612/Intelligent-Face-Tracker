import os

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


class MongoDB:

    def __init__(self):
        uri = os.getenv("MONGODB_URI")
        database_name = os.getenv("MONGODB_DATABASE")

        if not uri:
            raise RuntimeError("MONGODB_URI is not configured")

        if not database_name:
            raise RuntimeError("MONGODB_DATABASE is not configured")

        self.client = MongoClient(uri)

        self.client.admin.command("ping")

        self.database = self.client[database_name]

    def close(self):
        self.client.close()