from datetime import datetime, timedelta
import os

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from auth_service.main import app
from auth_service.routes import auth as auth_routes


class FakeAuthService:
    def __init__(self):
        self.users = {}
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.secret_key = os.getenv("JWT_SECRET", "test-secret")

    def decode_token(self, token: str):
        return jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])

    async def get_user(self, username: str):
        return self.users.get(username)

    async def authenticate_user(self, username: str, password: str):
        user = self.users.get(username)
        if not user or user["password"] != password:
            return False
        return user

    async def update_refresh_token(self, username: str, refresh_token: str):
        if username in self.users:
            self.users[username]["refresh_token"] = refresh_token

    async def login_user(self, username: str, password: str):
        user = await self.authenticate_user(username, password)
        if not user:
            return None, None
        access_token = self.create_access_token({"sub": username})
        refresh_token = self.create_refresh_token({"sub": username})
        await self.update_refresh_token(username, refresh_token)
        return access_token, refresh_token

    async def refresh_access_token(self, refresh_token: str):
        payload = self.decode_token(refresh_token)
        username = payload.get("sub")
        if not username:
            return None
        user = await self.get_user(username)
        if not user or user.get("refresh_token") != refresh_token:
            return None
        return self.create_access_token({"sub": username})

    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.jwt_algorithm)


@pytest.fixture
def client_and_service(monkeypatch):
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", "test-secret")

    fake_service = FakeAuthService()
    # Replace the real AuthService instance used by the routes
    auth_routes.auth_service = fake_service

    client = TestClient(app)
    return client, fake_service


def test_login_flow_with_existing_user(client_and_service):
    client, fake_service = client_and_service
    # Seed a user (registration endpoint removed)
    fake_service.users["alice"] = {
        "username": "alice",
        "password": "secret",
        "refresh_token": None,
    }

    login_response = client.post(
        "/login", json={"username": "alice", "password": "secret"}
    )
    assert login_response.status_code == 200
    login_data = login_response.json()
    assert "access_token" in login_data
    assert login_data["token_type"] == "bearer"

    # The login token should decode to the correct subject
    payload = jwt.decode(
        login_data["access_token"],
        "test-secret",
        algorithms=["HS256"],
    )
    assert payload["sub"] == "alice"


def test_verify_endpoint_with_valid_token(client_and_service):
    client, fake_service = client_and_service

    # Prepare a user directly in the fake service
    username = "bob"
    fake_service.users[username] = {
        "username": username,
        "password": "secret",
        "refresh_token": None,
    }

    access_token = fake_service.create_access_token({"sub": username})

    response = client.get(
        "/verify",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == username


def test_refresh_token_flow(client_and_service):
    client, fake_service = client_and_service

    username = "carol"
    fake_service.users[username] = {
        "username": username,
        "password": "secret",
        "refresh_token": None,
    }

    refresh_token = fake_service.create_refresh_token({"sub": username})
    fake_service.users[username]["refresh_token"] = refresh_token

    response = client.post(
        "/refresh",
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    payload = jwt.decode(
        data["access_token"],
        "test-secret",
        algorithms=["HS256"],
    )
    assert payload["sub"] == username
