from datetime import datetime, timedelta
import os

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from ..main import app
from ..routes import auth as auth_routes


class FakeAuthService:
    def __init__(self):
        self.users = {}
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.secret_key = os.getenv("JWT_SECRET", "test-secret")

    async def get_user(self, username: str):
        return self.users.get(username)

    async def create_user(self, username: str, password: str):
        self.users[username] = {
            "username": username,
            "password": password,
            "refresh_token": None,
        }

    async def authenticate_user(self, username: str, password: str):
        user = self.users.get(username)
        if not user or user["password"] != password:
            return False
        return user

    async def update_refresh_token(self, username: str, refresh_token: str):
        if username in self.users:
            self.users[username]["refresh_token"] = refresh_token

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


def test_register_and_login_flow(client_and_service):
    client, _ = client_and_service

    # Register a new user
    register_response = client.post(
        "/auth/register", json={"username": "alice", "password": "secret"}
    )
    assert register_response.status_code == 200
    register_data = register_response.json()
    assert "access_token" in register_data
    assert register_data["token_type"] == "bearer"

    # Login with the same user
    login_response = client.post(
        "/auth/login", json={"username": "alice", "password": "secret"}
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
        "/auth/verify",
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
        "/auth/refresh",
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

