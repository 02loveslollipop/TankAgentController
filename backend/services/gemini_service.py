import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from pydantic import ValidationError

from backend.models import Plan, PlannerContext


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "ROBOT_SYSTEM_ROLE"
ROBOT_SYSTEM_ROLE_PROMPT = None


class GeminiService:
    """Service wrapper around the Gemini Robotics model for plan generation."""

    def __init__(self, client: Optional[genai.Client] = None, model_id: Optional[str] = None, prompt_text: Optional[str] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key and client is None:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        self.client = client or genai.Client(api_key=api_key)
        self.model_id = model_id or "gemini-robotics-er-1.5-preview"
        self.system_prompt = prompt_text or self._load_prompt()

    def _load_prompt(self) -> str:
        global ROBOT_SYSTEM_ROLE_PROMPT
        if ROBOT_SYSTEM_ROLE_PROMPT:
            return ROBOT_SYSTEM_ROLE_PROMPT
        path = Path(os.getenv("ROBOT_SYSTEM_ROLE_PATH") or DEFAULT_PROMPT_PATH)
        try:
            ROBOT_SYSTEM_ROLE_PROMPT = path.read_text(encoding="utf-8")
        except OSError:
            ROBOT_SYSTEM_ROLE_PROMPT = (
                "You are a high-level planner. Output ONLY JSON with keys: "
                "goal_pixel, goal_bbox, action_type, status, confidence, hazards, explanation."
            )
        return ROBOT_SYSTEM_ROLE_PROMPT

    def _fallback_response(self, status: str, action_type: str, hazards: list[str], explanation: str) -> Dict[str, Any]:
        return Plan(
            goal_pixel=None,
            goal_bbox=None,
            action_type=action_type,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            confidence=0.0,
            hazards=hazards,
            explanation=explanation,
        ).model_dump()

    async def generate_plan(
        self,
        image: str,
        context: Optional[PlannerContext] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call Gemini with system prompt + context and return validated planner response.
        """
        ctx = context or PlannerContext()
        # Guardrail: hard stop bypasses model call
        if ctx.sensor_state and ctx.sensor_state.hard_stop:
            return self._fallback_response(
                status="unsafe",
                action_type="stop",
                hazards=["hard_stop"],
                explanation="Hardware stop asserted",
            )

        context_json = ctx.model_dump(exclude_none=True)
        if instruction and not context_json.get("user_instruction"):
            context_json["user_instruction"] = instruction

        request_contents = [
            image,
            self.system_prompt,
            f"CONTEXT_JSON:\n{json.dumps(context_json)}",
        ]
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=request_contents,
            )
            parsed = json.loads(response.text)
            validated = Plan.model_validate(parsed)
            # Clamp confidence
            if validated.confidence is not None:
                validated.confidence = max(0.0, min(1.0, validated.confidence))
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError, AttributeError, TypeError) as exc:
            return self._fallback_response(
                status="ambiguous_instruction",
                action_type="stop",
                hazards=["planner_error"],
                explanation=f"Planner parse error: {exc}",
            )
