from pymongo.asynchronous.mongo_client import AsyncMongoClient
import os

class DBContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = AsyncMongoClient(os.getenv("MONGODB_URL"))
            cls._instance.db = cls._instance.client.robot_db
        return cls._instance

    @property
    def database(self):
        return self.db
