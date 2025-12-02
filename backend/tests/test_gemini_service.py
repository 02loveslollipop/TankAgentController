import asyncio
import sys
import types

import pytest


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModelRunner:
    def __init__(self, response_text):
        self.response_text = response_text

    async def generate_content(self, model=None, contents=None):
        self.last_model = model
        self.last_contents = contents
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.runner = FakeModelRunner(response_text)
        self.aio = types.SimpleNamespace(models=self.runner)
        self.response_text = response_text


# Stub google.genai before importing the service (after FakeClient is defined)
google_module = types.ModuleType("google")
google_genai_module = types.ModuleType("genai")
google_genai_module.Client = FakeClient
google_module.genai = google_genai_module
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.genai", google_genai_module)


def make_service(response_text):
    from backend.services.gemini_service import GeminiService

    client = FakeClient(response_text)
    service = GeminiService(client=client, prompt_text="PROMPT")
    return service, client.runner


@pytest.mark.asyncio
async def test_hard_stop_bypasses_model():
    from backend.models import PlannerContext, SensorState

    service, runner = make_service("{}")
    ctx = PlannerContext(sensor_state=SensorState(hard_stop=True, hazard_flags=["hard_stop"]))

    plan = await service.generate_plan(image="img", context=ctx)
    assert plan["action_type"] == "stop"
    assert plan["status"] == "unsafe"
    assert "hard_stop" in plan["hazards"]
    assert not hasattr(runner, "last_contents")


@pytest.mark.asyncio
async def test_valid_plan_parses_and_clamps_confidence():
    from backend.models import PlannerContext

    response_text = '{"goal_pixel":[1,2],"goal_bbox":null,"action_type":"move","status":"ok","confidence":1.5,"hazards":["none"],"explanation":"ok"}'
    service, runner = make_service(response_text)
    ctx = PlannerContext(user_instruction="hi")

    plan = await service.generate_plan(image="img", context=ctx)
    assert plan["goal_pixel"] == [1, 2]
    assert plan["confidence"] == 1.0  # clamped
    assert runner.last_contents[1] == "PROMPT"
    assert "CONTEXT_JSON" in runner.last_contents[2]


@pytest.mark.asyncio
async def test_parse_failure_returns_fallback():
    from backend.models import PlannerContext

    service, _ = make_service("not-json")
    plan = await service.generate_plan(image="img", context=PlannerContext())
    assert plan["action_type"] == "stop"
    assert plan["status"] in ("ambiguous_instruction", "unsafe")
    assert "planner_error" in plan["hazards"]
