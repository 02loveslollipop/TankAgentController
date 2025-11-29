from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from auth_service.db.context import DBContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.db = DBContext().database
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        if self.jwt_algorithm == "HS256":
            self.secret_key = os.getenv("JWT_SECRET")
        else:
            # For RSA-based algorithms
            self.private_key = os.getenv("JWT_KEY")
            self.public_key = os.getenv("JWT_CERTIFICATE")

    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password):
        return pwd_context.hash(password)

    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        if self.jwt_algorithm == "HS256":
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.jwt_algorithm)
        else:
            encoded_jwt = jwt.encode(to_encode, self.private_key, algorithm=self.jwt_algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire})
        if self.jwt_algorithm == "HS256":
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.jwt_algorithm)
        else:
            encoded_jwt = jwt.encode(to_encode, self.private_key, algorithm=self.jwt_algorithm)
        return encoded_jwt

    async def authenticate_user(self, username: str, password: str):
        user = await self.db.users.find_one({"username": username})
        if not user or not self.verify_password(password, user["hashed_password"]):
            return False
        return user

    async def get_user(self, username: str):
        return await self.db.users.find_one({"username": username})

    async def create_user(self, username: str, password: str):
        hashed_password = self.get_password_hash(password)
        await self.db.users.insert_one({"username": username, "hashed_password": hashed_password})

    async def update_refresh_token(self, username: str, refresh_token: str):
        await self.db.users.update_one({"username": username}, {"$set": {"refresh_token": refresh_token}})