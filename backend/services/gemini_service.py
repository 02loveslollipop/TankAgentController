import json
import os
from typing import Any, Dict, Optional

from google import genai


class GeminiService:
    """Service wrapper around the Gemini Robotics model for plan generation."""

    def __init__(self, client: Optional[genai.Client] = None, model_id: Optional[str] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key and client is None:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        self.client = client or genai.Client(api_key=api_key)
        self.model_id = model_id or "gemini-robotics-er-1.5-preview"

    async def generate_plan(self, image: str, instruction: Optional[str] = None) -> Dict[str, Any]:
        prompt = f"Plan navigation: {instruction or 'find safe path'}"
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=[image, prompt],
        )
        return json.loads(response.text)

