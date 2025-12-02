from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


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


class GPSState(BaseModel):
    latitude: Optional[float] = Field(default=None, description="Latitude in decimal degrees.")
    longitude: Optional[float] = Field(default=None, description="Longitude in decimal degrees.")
    accuracy_m: Optional[float] = Field(default=None, description="Estimated GPS accuracy (meters).")


class RobotState(BaseModel):
    model_config = ConfigDict(extra="allow")

    last_goal_pixel: Optional[List[int]] = Field(default=None, description="Previous goal pixel [u, v].")
    last_status: Optional[str] = Field(default=None, description="Previous planner status.")
    plan_id: Optional[str] = Field(default=None, description="Identifier for current/last plan.")
    gps: Optional[GPSState] = Field(default=None, description="Current GPS reading.")
    estimated_heading_deg: Optional[float] = Field(default=None, description="Estimated heading (deg).")


class SensorState(BaseModel):
    model_config = ConfigDict(extra="allow")

    hazard_flags: List[str] = Field(default_factory=list, description="Sensor hazards e.g. ['front_obstacle'].")
    hard_stop: bool = Field(default=False, description="Hardware or software stop asserted.")
    front_distance_m: Optional[float] = Field(
        default=None, description="Front obstacle distance (meters) from ultrasonic/fused estimate."
    )
    notes: Optional[str] = Field(default=None, description="Additional sensor notes.")


class TaskContext(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Task identifier.")
    step_index: Optional[int] = Field(default=None, description="Index within task steps.")
    high_level_goal: Optional[str] = Field(default=None, description="High-level goal description.")


class PlannerContext(BaseModel):
    """
    Structured context passed to the planner per ROBOT_SYSTEM_ROLE.
    """

    image_width: Optional[int] = Field(default=None, description="Width of the source image in pixels.")
    image_height: Optional[int] = Field(default=None, description="Height of the source image in pixels.")
    user_instruction: Optional[str] = Field(default=None, description="Operator instruction.")
    robot_state: Optional[RobotState] = Field(default=None, description="Robot state context.")
    sensor_state: Optional[SensorState] = Field(default=None, description="Sensor context.")
    task_context: Optional[TaskContext] = Field(default=None, description="Task metadata.")


class RobotFrameMessage(BaseModel):
    """Inbound frame+instruction from robot -> backend to request a plan."""

    type: Literal["frame"] = "frame"
    robot_id: str = Field(..., description="Unique robot identifier.")
    frame_id: Optional[str] = Field(default=None, description="Identifier for correlating responses.")
    image: str = Field(..., description="Base64-encoded image payload.")
    instruction: Optional[str] = Field(
        default=None, description="Optional user or system instruction to guide planning."
    )
    image_width: Optional[int] = Field(default=None, description="Width of the source image in pixels.")
    image_height: Optional[int] = Field(default=None, description="Height of the source image in pixels.")
    user_instruction: Optional[str] = Field(
        default=None, description="Operator instruction to include in planner context."
    )
    robot_state: Optional[RobotState] = Field(
        default=None,
        description="Robot state context: last goal/status/plan_id, gps (lat/lon/accuracy_m), estimated_heading_deg.",
    )
    sensor_state: Optional[SensorState] = Field(
        default=None,
        description="Sensor context: hazard_flags, hard_stop, front_distance_m, notes.",
    )
    task_context: Optional[TaskContext] = Field(
        default=None, description="Task metadata: task_id, step_index, high_level_goal."
    )


class Plan(BaseModel):
    """
    Planner response JSON as defined in ROBOT_SYSTEM_ROLE.
    """

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
    action_type: Optional[Literal["move", "stop", "wait", "ask_human", "cancel"]] = Field(
        default=None, description="Action hint for controller per prompt."
    )
    status: Optional[
        Literal["ok", "blocked", "task_complete", "ambiguous_instruction", "unsafe"]
    ] = Field(default=None, description="Planner status per prompt.")
    confidence: Optional[float] = Field(default=None, description="Planner confidence 0-1 (clamped).")
    hazards: List[str] = Field(default_factory=list, description="Hazard labels detected/considered.")
    explanation: Optional[str] = Field(default=None, description="Natural language rationale.")

    @field_validator("goal_pixel")
    @classmethod
    def validate_goal_pixel(cls, value):
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("goal_pixel must have length 2")
        if any(v < 0 for v in value):
            raise ValueError("goal_pixel values must be non-negative")
        return value

    @field_validator("goal_bbox")
    @classmethod
    def validate_goal_bbox(cls, value):
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("goal_bbox must have length 4")
        x1, y1, x2, y2 = value
        if x2 < x1 or y2 < y1:
            raise ValueError("goal_bbox coordinates must be ordered (x1<=x2, y1<=y2)")
        if any(v < 0 for v in value):
            raise ValueError("goal_bbox values must be non-negative")
        return value

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value):
        if value is None:
            return value
        return max(0.0, min(1.0, float(value)))


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
    planner_context_schema: Optional[Dict[str, Any]] = None
    planner_response_schema: Optional[Dict[str, Any]] = None
    examples: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = []
