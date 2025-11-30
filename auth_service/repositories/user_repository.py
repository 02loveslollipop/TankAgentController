from auth_service.db.context import DBContext


class UserRepository:
    """Data access layer for the users collection."""

    def __init__(self, db_context: DBContext | None = None):
        context = db_context or DBContext()
        self.collection = context.database.users

    async def get_by_username(self, username: str):
        return await self.collection.find_one({"username": username})

    async def create(self, username: str, hashed_password: str):
        return await self.collection.insert_one(
            {
                "username": username,
                "hashed_password": hashed_password,
                "refresh_token": None,
            }
        )

    async def update_refresh_token(self, username: str, refresh_token: str):
        return await self.collection.update_one(
            {"username": username},
            {"$set": {"refresh_token": refresh_token}},
        )
