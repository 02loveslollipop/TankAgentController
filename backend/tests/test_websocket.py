import asyncio
import json
import os
import sys
import types

import pytest
from jose import jwt
from tornado import websocket, httpserver, testing, httpclient
from tornado.platform.asyncio import AsyncIOMainLoop

# Provide a stub for google.genai before importing backend.main
google_module = types.ModuleType("google")
google_genai_module = types.ModuleType("genai")
google_genai_module.Client = lambda api_key=None: types.SimpleNamespace(
    aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=None))
)
google_module.genai = google_genai_module
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.genai", google_genai_module)


class FakeGeminiService:
    async def generate_plan(self, image: str, instruction: str | None = None):
        return {"goal_pixel": [10, 20], "confidence": 0.9, "instruction": instruction}


class FakeTelemetryRepository:
    def __init__(self):
        self.records = []

    async def insert(self, robot_id, telemetry, frame_id=None):
        self.records.append({"robot_id": robot_id, "telemetry": telemetry, "frame_id": frame_id})


class FakePlanRepository:
    def __init__(self):
        self.records = []

    async def insert(self, robot_id, plan, frame_id=None):
        self.records.append({"robot_id": robot_id, "plan": plan, "frame_id": frame_id})


@pytest.fixture(scope="module", autouse=True)
def install_asyncio_loop():
    # Ensure Tornado uses asyncio event loop for websocket_connect
    try:
        AsyncIOMainLoop().install()
    except RuntimeError:
        pass
    yield


def make_token(subject: str) -> str:
    return jwt.encode({"sub": subject}, "test-secret", algorithm="HS256")


@pytest.mark.asyncio
async def test_frame_broadcasts_plan_and_persists(monkeypatch):
    import backend.main as main

    os.environ["JWT_ALGORITHM"] = "HS256"
    os.environ["JWT_SECRET"] = "test-secret"

    fake_gemini = FakeGeminiService()
    fake_telemetry = FakeTelemetryRepository()
    fake_plan = FakePlanRepository()

    monkeypatch.setattr(main, "GeminiService", lambda: fake_gemini)
    monkeypatch.setattr(main, "TelemetryRepository", lambda: fake_telemetry)
    monkeypatch.setattr(main, "PlanRepository", lambda: fake_plan)

    app = main.make_app()
    server = httpserver.HTTPServer(app)
    sock, port = testing.bind_unused_port()
    server.add_socket(sock)

    token = make_token("robot-1")
    headers = {"Authorization": f"Bearer {token}"}
    base_ws = f"ws://127.0.0.1:{port}"

    try:
        frontend_ws = await websocket.websocket_connect(
            httpclient.HTTPRequest(f"{base_ws}/ws/frontend", headers=headers)
        )
        robot_ws = await websocket.websocket_connect(
            httpclient.HTTPRequest(f"{base_ws}/ws/robot", headers=headers)
        )

        frame_msg = {
            "type": "frame",
            "robot_id": "robot-1",
            "frame_id": "frame-123",
            "image": "base64",
            "instruction": "go forward",
        }
        await robot_ws.write_message(json.dumps(frame_msg))

        response_raw = await frontend_ws.read_message()
        response = json.loads(response_raw)

        assert response["type"] == "plan"
        assert response["robot_id"] == "robot-1"
        assert response["frame_id"] == "frame-123"
        assert response["plan"]["goal_pixel"] == [10, 20]

        assert len(fake_plan.records) == 1
        assert fake_plan.records[0]["plan"]["goal_pixel"] == [10, 20]
    finally:
        try:
            frontend_ws.close()
        except Exception:
            pass
        try:
            robot_ws.close()
        except Exception:
            pass
        server.stop()
