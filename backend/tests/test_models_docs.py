import json
import os
import sys
import types

import pytest
from tornado import httpclient, httpserver, testing
from tornado.platform.asyncio import AsyncIOMainLoop

# Stub google.genai to avoid import errors when loading backend.main
google_module = types.ModuleType("google")
google_genai_module = types.ModuleType("genai")
google_genai_module.Client = lambda api_key=None: types.SimpleNamespace(
    aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=None))
)
google_module.genai = google_genai_module
sys_modules = getattr(__import__("sys"), "modules")
sys_modules.setdefault("google", google_module)
sys_modules.setdefault("google.genai", google_genai_module)


@pytest.fixture(scope="module", autouse=True)
def install_asyncio_loop():
    try:
        AsyncIOMainLoop().install()
    except RuntimeError:
        pass
    yield


def test_plan_validation_errors():
    from backend.models import Plan
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Plan(goal_pixel=[1])
    with pytest.raises(pydantic.ValidationError):
        Plan(goal_bbox=[10, 10, 5, 5])


@pytest.mark.asyncio
async def test_docs_exposes_planner_fields(monkeypatch):
    import backend.main as main

    # Avoid real external dependencies during docs fetch
    class DummyGemini:
        pass

    class DummyRepo:
        def __init__(self, *args, **kwargs):
            pass

    os.environ["JWT_ALGORITHM"] = "HS256"
    os.environ["JWT_SECRET"] = "test-secret"

    monkeypatch.setattr(main, "GeminiService", lambda: DummyGemini())
    monkeypatch.setattr(main, "TelemetryRepository", lambda: DummyRepo())
    monkeypatch.setattr(main, "PlanRepository", lambda: DummyRepo())

    app = main.make_app()
    server = httpserver.HTTPServer(app)
    sock, port = testing.bind_unused_port()
    server.add_socket(sock)

    try:
        client = httpclient.AsyncHTTPClient()
        resp = await client.fetch(f"http://127.0.0.1:{port}/docs")
        body = json.loads(resp.body)

        props = body["planner_response_schema"]["properties"]
        assert set(["action_type", "status", "hazards", "explanation"]).issubset(props.keys())
        context = body["examples"]["planner_context"]
        assert context["image_width"] == 320
        assert "robot_state" in context and "sensor_state" in context and "task_context" in context
    finally:
        server.stop()
