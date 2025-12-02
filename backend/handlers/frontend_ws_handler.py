import json
from typing import Any, Dict

import tornado.websocket
from jose import JWTError
from jose.exceptions import ExpiredSignatureError

from backend.models import FrontendSubscribeMessage
from backend.services.jwt_service import JWTAuthService


class FrontendWebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, state: Dict[str, Any], jwt_service: JWTAuthService):
        self.state = state
        self.jwt_service = jwt_service
        self.jwt_payload: Dict[str, Any] | None = None

    def check_origin(self, origin: str) -> bool:
        # Allow cross-origin WebSocket connections (lock down in production).
        return True

    def open(self):
        if not self._authenticate():
            return
        self.state["frontend_clients"].add(self)

    async def on_message(self, message: str):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        if payload.get("type") == "subscribe":
            try:
                FrontendSubscribeMessage.model_validate(payload)
            except Exception:
                return
            # Subscription filtering placeholder; for now, accept all.

    def on_close(self):
        self.state["frontend_clients"].discard(self)

    def _authenticate(self) -> bool:
        token = self._extract_token()
        if not token:
            self.close(code=4001, reason="missing token")
            return False
        try:
            self.jwt_payload = self.jwt_service.decode_token(token)
            return True
        except ExpiredSignatureError:
            self.close(code=4001, reason="token expired")
            return False
        except JWTError:
            self.close(code=4003, reason="invalid token")
            return False
        except RuntimeError as exc:
            self.close(code=4003, reason=str(exc))
            return False

    def _extract_token(self) -> str | None:
        auth_header = self.request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header.split(" ", 1)[1].strip()
        return self.get_argument("token", default=None)
