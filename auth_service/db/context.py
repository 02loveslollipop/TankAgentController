from motor.motor_asyncio import AsyncIOMotorClient
import os

class DBContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
            cls._instance.db = cls._instance.client.robot_db
        return cls._instance

    @property
    def database(self):
        return self.db