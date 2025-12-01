import os

from pymongo.asynchronous.mongo_client import AsyncMongoClient


class DBContext:
    """Singleton wrapper around the async MongoDB client."""

    _instance: "DBContext | None" = None

    def __new__(cls) -> "DBContext":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            mongo_url = os.getenv("MONGODB_URL")
            if not mongo_url:
                raise RuntimeError("MONGODB_URL environment variable is not set")
            cls._instance._client = AsyncMongoClient(mongo_url)
            cls._instance._db = cls._instance._client.robot_db
        return cls._instance

    @property
    def database(self):
        return self._db

