import json
import os
from typing import Any, Dict, Set

import logging

import tornado.ioloop
import tornado.web
import tornado.websocket

from backend.handlers import DocsHandler, FrontendWebSocketHandler, HealthHandler, RobotWebSocketHandler
from backend.repositories import PlanRepository, TelemetryRepository
from backend.services.gemini_service import GeminiService
from backend.services.jwt_service import JWTAuthService


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
