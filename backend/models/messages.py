from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class TelemetryData(BaseModel):
    model_config = ConfigDict(extra="allow")

    pose: Optional[Dict[str, float]] = Field(
        default=None, description="Robot pose estimate (e.g., {x, y, theta})."
    )
    velocity: Optional[Dict[str, float]] = Field(
        default=None, description="Robot velocity (e.g., {linear, angular})."
    )
    battery: Optional[float] = Field(default=None, description="Battery level in percent.")
    hazards: Optional[Dict[str, Any]] = Field(
        default=None, description="Hazard flags/metadata from sensors or perception."
    )
    extras: Optional[Dict[str, Any]] = Field(
        default=None, description="Any additional telemetry fields."
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp for the telemetry reading.",
    )


class RobotTelemetryMessage(BaseModel):
    """Inbound telemetry message from robot -> backend."""

    type: Literal["telemetry"] = "telemetry"
    robot_id: str = Field(..., description="Unique robot identifier.")
    telemetry: TelemetryData


class RobotFrameMessage(BaseModel):
    """Inbound frame+instruction from robot -> backend to request a plan."""

    type: Literal["frame"] = "frame"
    robot_id: str = Field(..., description="Unique robot identifier.")
    frame_id: Optional[str] = Field(default=None, description="Identifier for correlating responses.")
    image: str = Field(..., description="Base64-encoded image payload.")
    instruction: Optional[str] = Field(
        default=None, description="Optional user or system instruction to guide planning."
    )


class Plan(BaseModel):
    goal_pixel: Optional[List[int]] = Field(
        default=None,
        description="Target pixel [u, v] in image coordinates.",
        min_length=2,
        max_length=2,
    )
    goal_bbox: Optional[List[int]] = Field(
        default=None,
        description="Optional bounding box [x1, y1, x2, y2].",
        min_length=4,
        max_length=4,
    )
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    hazards: Optional[List[str]] = Field(default=None, description="Hazard labels detected/considered.")
    explanation: Optional[str] = Field(default=None, description="Natural language rationale.")


class PlanMessage(BaseModel):
    """Outbound plan from backend -> frontend/robot."""

    type: Literal["plan"] = "plan"
    robot_id: str
    frame_id: Optional[str] = None
    plan: Plan


class FrontendSubscribeMessage(BaseModel):
    """Placeholder for future frontend-subscribe semantics."""

    type: Literal["subscribe"] = "subscribe"
    robot_id: Optional[str] = Field(default=None, description="Filter updates for a specific robot.")


class SchemaDocument(BaseModel):
    """Documentation payload served at /docs for quick reference."""

    websocket_endpoints: Dict[str, str]
    inbound_messages: Dict[str, Dict[str, Any]]
    outbound_messages: Dict[str, Dict[str, Any]]
    notes: List[str] = []
