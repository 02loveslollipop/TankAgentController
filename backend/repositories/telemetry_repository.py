from datetime import datetime
from typing import Any, Dict, Optional

from backend.db.context import DBContext


class TelemetryRepository:
    """Data access layer for telemetry documents."""

    def __init__(self, db_context: Optional[DBContext] = None):
        context = db_context or DBContext()
        self.collection = context.database.telemetry

    async def insert(self, robot_id: str, telemetry: Dict[str, Any], frame_id: Optional[str] = None):
        doc = {
            "robot_id": robot_id,
            "telemetry": telemetry,
            "frame_id": frame_id,
            "timestamp": datetime.utcnow(),
        }
        await self.collection.insert_one(doc)

