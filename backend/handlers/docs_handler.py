import json
import tornado.web

from backend.models import (
    Plan,
    PlanMessage,
    PlannerContext,
    RobotFrameMessage,
    RobotTelemetryMessage,
    SchemaDocument,
)


class DocsHandler(tornado.web.RequestHandler):
    def get(self):
        planner_context_example = {
            "image_width": 320,
            "image_height": 240,
            "user_instruction": "go to the trash can and stop",
            "robot_state": {
                "last_goal_pixel": [160, 200],
                "last_status": "moving",
                "plan_id": "abc123",
                "gps": {
                    "latitude": 12.345678,
                    "longitude": -98.765432,
                    "accuracy_m": 3.5,
                },
                "estimated_heading_deg": 72.0,
            },
            "sensor_state": {
                "hazard_flags": ["none"],
                "hard_stop": False,
                "front_distance_m": 0.72,
                "notes": "local safety reports free space",
            },
            "task_context": {
                "task_id": "task_001",
                "step_index": 2,
                "high_level_goal": "go to the trash can and stop",
            },
        }

        planner_response_example = {
            "goal_pixel": [210, 190],
            "goal_bbox": [190, 130, 240, 220],
            "action_type": "move",
            "status": "ok",
            "confidence": 0.92,
            "hazards": [],
            "explanation": "Trash can detected on the right; goal pixel placed at the base. GPS not required for this step.",
        }

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
            planner_context_schema=PlannerContext.model_json_schema(),
            planner_response_schema=Plan.model_json_schema(),
            examples={
                "planner_context": planner_context_example,
                "planner_response": planner_response_example,
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
