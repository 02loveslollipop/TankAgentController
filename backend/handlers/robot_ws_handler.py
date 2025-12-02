import json
from typing import Any, Dict

import tornado.websocket
from jose import JWTError
from jose.exceptions import ExpiredSignatureError

from backend.models import Plan, PlanMessage, PlannerContext, RobotFrameMessage, RobotTelemetryMessage
from backend.repositories import PlanRepository, TelemetryRepository
from backend.services.gemini_service import GeminiService
from backend.services.jwt_service import JWTAuthService


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
        planner_context = PlannerContext(
            image_width=payload.image_width,
            image_height=payload.image_height,
            user_instruction=payload.user_instruction or payload.instruction,
            robot_state=payload.robot_state,
            sensor_state=payload.sensor_state,
            task_context=payload.task_context,
        )

        plan = await self.gemini_service.generate_plan(
            image=payload.image,
            context=planner_context,
            instruction=payload.instruction,
        )

        plan_msg = PlanMessage(
            robot_id=payload.robot_id,
            frame_id=payload.frame_id,
            plan=Plan.model_validate(plan),
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
            plan_id=payload.robot_state.plan_id if payload.robot_state else None,
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
