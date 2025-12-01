import json
import os
from typing import Any, Dict, Set

import logging

import tornado.ioloop
import tornado.web
import tornado.websocket
from jose import JWTError
from jose.exceptions import ExpiredSignatureError

from backend.models import (
    FrontendSubscribeMessage,
    PlanMessage,
    RobotFrameMessage,
    RobotTelemetryMessage,
    SchemaDocument,
)
from backend.repositories import PlanRepository, TelemetryRepository
from backend.services.gemini_service import GeminiService
from backend.services.jwt_service import JWTAuthService


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": "ok"})


class RobotWebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(
        self,
        gemini_service: GeminiService,
        telemetry_repo: TelemetryRepository,
        plan_repo: PlanRepository,
        state: Dict[str, Any],
        jwt_service: JWTAuthService,
    ):
        self.gemini_service = gemini_service
        self.telemetry_repo = telemetry_repo
        self.plan_repo = plan_repo
        self.state = state
        self.jwt_service = jwt_service
        self.jwt_payload: Dict[str, Any] | None = None

    def check_origin(self, origin: str) -> bool:
        # Allow cross-origin WebSocket connections (lock down in production).
        return True

    def open(self):
        if not self._authenticate():
            return
        self.state["robot_clients"].add(self)

    async def on_message(self, message: str):
        parsed = self._parse_robot_message(message)
        if isinstance(parsed, RobotTelemetryMessage):
            await self._handle_telemetry(parsed)
        elif isinstance(parsed, RobotFrameMessage):
            await self._handle_frame(parsed)

    def _parse_robot_message(self, message: str):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return None

        msg_type = payload.get("type")
        if msg_type == "telemetry":
            try:
                return RobotTelemetryMessage.model_validate(payload)
            except Exception:
                return None
        if msg_type == "frame":
            try:
                return RobotFrameMessage.model_validate(payload)
            except Exception:
                return None
        return None

    async def _handle_telemetry(self, payload: RobotTelemetryMessage):
        # For now, just forward telemetry to all frontend clients.
        for client in list(self.state["frontend_clients"]):
            try:
                await client.write_message(
                    json.dumps(
                        {
                            "type": "telemetry",
                            "robot_id": payload.robot_id,
                            "telemetry": payload.telemetry.model_dump(),
                        }
                    )
                )
            except tornado.websocket.WebSocketClosedError:
                self.state["frontend_clients"].discard(client)
        await self.telemetry_repo.insert(
            robot_id=payload.robot_id,
            telemetry=payload.telemetry.model_dump(),
        )

    async def _handle_frame(self, payload: RobotFrameMessage):
        plan = await self.gemini_service.generate_plan(image=payload.image, instruction=payload.instruction)

        plan_msg = PlanMessage(
            robot_id=payload.robot_id,
            frame_id=payload.frame_id,
            plan=plan,
        )
        message = plan_msg.model_dump_json()
        for client in list(self.state["frontend_clients"]):
            try:
                await client.write_message(message)
            except tornado.websocket.WebSocketClosedError:
                self.state["frontend_clients"].discard(client)
        await self.plan_repo.insert(
            robot_id=payload.robot_id,
            plan=plan,
            frame_id=payload.frame_id,
        )

    def on_close(self):
        self.state["robot_clients"].discard(self)

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


class DocsHandler(tornado.web.RequestHandler):
    def get(self):
        schema = SchemaDocument(
            websocket_endpoints={
                "robot": "/ws/robot",
                "frontend": "/ws/frontend",
            },
            inbound_messages={
                "RobotTelemetryMessage": RobotTelemetryMessage.model_json_schema(),
                "RobotFrameMessage": RobotFrameMessage.model_json_schema(),
            },
            outbound_messages={
                "PlanMessage": PlanMessage.model_json_schema(),
                "TelemetryForward": {
                    "type": "telemetry",
                    "robot_id": "string",
                    "telemetry": RobotTelemetryMessage.model_json_schema()["properties"]["telemetry"],
                },
            },
            notes=[
                "All WebSocket messages are JSON.",
                "Images are expected as base64-encoded strings in RobotFrameMessage.image.",
                "PlanMessage is broadcast to all connected frontend clients.",
                "Authentication: provide Bearer token via 'Authorization: Bearer <token>' header or '?token=' query param on WebSocket connect.",
            ],
        )
        self.set_header("Content-Type", "application/json")
        self.write(schema.model_dump(mode="json"))


def make_app() -> tornado.web.Application:
    gemini_service = GeminiService()
    telemetry_repo = TelemetryRepository()
    plan_repo = PlanRepository()
    jwt_service = JWTAuthService()
    state: Dict[str, Set[tornado.websocket.WebSocketHandler]] = {
        "robot_clients": set(),
        "frontend_clients": set(),
    }

    return tornado.web.Application(
        [
            (r"/health", HealthHandler),
            (r"/docs", DocsHandler),
            (
                r"/ws/robot",
                RobotWebSocketHandler,
                dict(
                    gemini_service=gemini_service,
                    telemetry_repo=telemetry_repo,
                    plan_repo=plan_repo,
                    state=state,
                    jwt_service=jwt_service,
                ),
            ),
            (
                r"/ws/frontend",
                FrontendWebSocketHandler,
                dict(state=state, jwt_service=jwt_service),
            ),
        ]
    )
    
    
def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def main() -> None:
    logger = setup_logger(__name__)
    logger.info(f"Started server process {os.getpid()}")
    port = int(os.environ.get("PORT", "8000"))
    address = os.environ.get("ADDRESS", "0.0.0.0")
    app = make_app()
    logger.info(f"Waiting for application startup...")
    app.listen(port=port, address=address)
    logger.info(f"Application startup complete.")
    logger.info(f"Tornado running on http://{address}:{port} (Press Ctrl+C to quit)")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
