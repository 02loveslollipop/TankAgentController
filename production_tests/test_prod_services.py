import os

import pytest
import requests


AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "https://auth.kanade.02labs.me")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://control.kanade.02labs.me")
PROD_LOGIN_USER = os.getenv("PROD_LOGIN_USER")
PROD_LOGIN_PASSWORD = os.getenv("PROD_LOGIN_PASSWORD")
PROD_DB_USER = os.getenv("PROD_TEST_USER")
PROD_DB_PASSWORD = os.getenv("PROD_TEST_PASSWORD")
PROD_MONGODB_URI = os.getenv("PROD_MONGODB_URI")
AUTH_LOGIN_PATH = os.getenv("AUTH_LOGIN_PATH", "/login")


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_auth_health():
    resp = requests.get(_url(AUTH_BASE_URL, "/health"), timeout=10)
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_backend_health():
    resp = requests.get(_url(BACKEND_BASE_URL, "/health"), timeout=10)
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


@pytest.mark.skipif(not PROD_LOGIN_USER or not PROD_LOGIN_PASSWORD, reason="PROD_LOGIN_USER/PASSWORD not set")
def test_auth_login():
    payload = {"username": PROD_LOGIN_USER, "password": PROD_LOGIN_PASSWORD}
    url = _url(AUTH_BASE_URL, AUTH_LOGIN_PATH)
    resp = requests.post(url, json=payload, timeout=10)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code} from {url}: {resp.text}"
    body = resp.json()
    assert "access_token" in body
    assert body.get("token_type") == "bearer"


@pytest.mark.skipif(
    not (PROD_MONGODB_URI and PROD_DB_USER and PROD_DB_PASSWORD and PROD_LOGIN_USER),
    reason="DB credentials/URI not set",
)
def test_user_exists_in_db():
    """Validate login user exists in MongoDB using DB credentials."""
    from pymongo import MongoClient

    client = MongoClient(PROD_MONGODB_URI, username=PROD_DB_USER, password=PROD_DB_PASSWORD)
    db = client.get_default_database("robot_db") or client.get_database("robot_db")
    doc = db.users.find_one({"username": PROD_LOGIN_USER})
    assert doc is not None, f"User {PROD_LOGIN_USER} not found in users collection"
