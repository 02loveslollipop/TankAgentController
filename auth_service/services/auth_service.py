from datetime import datetime, timedelta
import os

from jose import jwt
from passlib.context import CryptContext

from auth_service.repositories.user_repository import UserRepository

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    def __init__(self, user_repository: UserRepository | None = None):
        self.user_repository = user_repository or UserRepository()
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

    def decode_token(self, token: str):
        """Decode a JWT token and return its payload."""
        key = self.secret_key if self.jwt_algorithm == "HS256" else self.public_key
        return jwt.decode(token, key, algorithms=[self.jwt_algorithm])

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
        user = await self.user_repository.get_by_username(username)
        if not user or not self.verify_password(password, user["hashed_password"]):
            return False
        return user

    async def get_user(self, username: str):
        return await self.user_repository.get_by_username(username)

    async def update_refresh_token(self, username: str, refresh_token: str):
        await self.user_repository.update_refresh_token(username, refresh_token)

    async def login_user(self, username: str, password: str):
        user = await self.authenticate_user(username, password)
        if not user:
            return None, None
        access_token = self.create_access_token(data={"sub": username})
        refresh_token = self.create_refresh_token(data={"sub": username})
        await self.update_refresh_token(username, refresh_token)
        return access_token, refresh_token

    async def refresh_access_token(self, refresh_token: str):
        payload = self.decode_token(refresh_token)
        username: str | None = payload.get("sub")
        if not username:
            return None
        user = await self.get_user(username)
        if not user or user.get("refresh_token") != refresh_token:
            return None
        return self.create_access_token(data={"sub": username})
