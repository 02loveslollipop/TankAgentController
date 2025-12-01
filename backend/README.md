# Backend Orchestration Service

This service manages robot ↔ backend ↔ frontend communication, calls the Gemini planner, and streams plans/telemetry over WebSockets. It is built with Tornado and uses MongoDB (PyMongo async client) for persistence.

## Project structure
- `main.py` - Tornado app with WebSocket endpoints for robot (`/ws/robot`) and frontend (`/ws/frontend`) plus `/health` and `/docs` (JSON schema for WS messages).
- `models/` - Pydantic models for WebSocket message schemas.
- `services/` - Domain services (`gemini_service.py`, `jwt_service.py`).
- `repositories/` - Data access for telemetry and plans.
- `db/` - MongoDB context (`context.py` singleton with PyMongo async client).
- `tests/` - Pytest/pytest-asyncio tests for JWT and WebSocket flows.

## Environment Variables
These configure the backend service. Supply them via env, CI, or Docker runtime.

- `MONGODB_URL` (required): MongoDB URI.
- `JWT_ALGORITHM` (optional): JWT algorithm; defaults to `HS256`. Set to RSA (e.g., `RS256`) to use public key.
- `JWT_SECRET` or `SECRET_KEY` (required for `HS256`): Secret used to verify HS256 JWTs.
- `JWT_CERTIFICATE` (required for RSA): Public key to verify RSA JWTs.
- `PORT` (optional): Port to listen on; defaults to `8000` if unset.
- `GEMINI_API_KEY` (required when calling Gemini): API key for `google-genai`.

## WebSocket Endpoints
- `/ws/robot`: Robot connects here. Sends:
  - `RobotTelemetryMessage` (`type=telemetry`, `robot_id`, `telemetry`)
  - `RobotFrameMessage` (`type=frame`, `robot_id`, `frame_id`, `image`, optional `instruction`)
  Backend responds with `PlanMessage` broadcasts to frontend clients.
- `/ws/frontend`: Frontend connects here to receive `PlanMessage` and forwarded telemetry.

Auth: Supply JWT via `Authorization: Bearer <token>` header or `?token=` query param when opening the WebSocket.

Docs: `GET /docs` returns JSON schema for the WebSocket message types.

## Docker
Build (from repo root):

```bash
docker build -t backend:test -f backend/Dockerfile ./backend
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e MONGODB_URL="mongodb://host.docker.internal:27017/robot_db" \
  -e JWT_ALGORITHM=HS256 -e JWT_SECRET=test-secret \
  -e GEMINI_API_KEY=your-key \
  backend:test
```

## Tests
Pytest covers JWT decoding and WebSocket plan flow (with fakes).

Run tests locally:

```bash
pytest backend/tests -q
```

Run tests in Docker (CI pattern):

```bash
docker build -t backend:test -f backend/Dockerfile ./backend
docker run --rm -w /app/backend -e PYTHONPATH=/app/backend backend:test pytest tests -q
```
