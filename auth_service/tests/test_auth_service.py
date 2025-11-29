from datetime import datetime
from types import SimpleNamespace

from jose import jwt
import services.auth_service as auth_module


def _make_auth_service(monkeypatch):
    class FakeDBContext:
        def __init__(self):
            self.database = SimpleNamespace()

    monkeypatch.setattr(auth_module, "DBContext", FakeDBContext)
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", "test-secret")

    return auth_module.AuthService()


def test_create_access_token_contains_subject_and_exp(monkeypatch):
    service = _make_auth_service(monkeypatch)
    data = {"sub": "test-user"}

    # Act
    token = service.create_access_token(data)
    payload = jwt.decode(token, "test-secret", algorithms=["HS256"])

    # Assert
    assert payload["sub"] == "test-user"
    # exp should be in the future relative to now
    assert payload["exp"] > int(datetime.utcnow().timestamp())


def test_create_refresh_token_has_longer_expiration(monkeypatch):
    service = _make_auth_service(monkeypatch)
    data = {"sub": "test-user"}

    access_token = service.create_access_token(data)
    refresh_token = service.create_refresh_token(data)

    access_payload = jwt.decode(access_token, "test-secret", algorithms=["HS256"])
    refresh_payload = jwt.decode(refresh_token, "test-secret", algorithms=["HS256"])

    assert refresh_payload["exp"] > access_payload["exp"]
