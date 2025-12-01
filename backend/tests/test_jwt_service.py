import os

from jose import jwt

from backend.services.jwt_service import JWTAuthService


def test_jwt_service_decodes_valid_token(monkeypatch):
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    token = jwt.encode({"sub": "robot-1"}, "test-secret", algorithm="HS256")

    service = JWTAuthService()
    payload = service.decode_token(token)

    assert payload["sub"] == "robot-1"


def test_jwt_service_try_decode_returns_none_on_invalid(monkeypatch):
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", "test-secret")

    service = JWTAuthService()
    # Wrong secret
    bad_token = jwt.encode({"sub": "robot-1"}, "other-secret", algorithm="HS256")

    assert service.try_decode(bad_token) is None
