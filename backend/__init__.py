"""
Backend main orchestration service package.

This service is responsible for:
- Managing WebSocket connections to the robot SBC and frontend.
- Calling the Gemini API to generate navigation plans.
- Broadcasting plans and telemetry between robot and UI.

The HTTP/WebSocket server is implemented with Tornado.
"""

