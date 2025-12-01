import os

import pytest
import requests


AUTH_BASE_URL = os.getenv("AUTH_BASE_URL", "https://auth.kanade.02labs.me")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://control.kanade.02labs.me")
PROD_USER = os.getenv("PROD_TEST_USER")
PROD_PASSWORD = os.getenv("PROD_TEST_PASSWORD")


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


@pytest.mark.skipif(not PROD_USER or not PROD_PASSWORD, reason="PROD_TEST_USER/PASSWORD not set")
def test_auth_login():
    payload = {"username": PROD_USER, "password": PROD_PASSWORD}
    resp = requests.post(_url(AUTH_BASE_URL, "/auth/login"), json=payload, timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body.get("token_type") == "bearer"
