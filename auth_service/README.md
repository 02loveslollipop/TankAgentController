# Auth Service

This service handles user authentication for the TankAgentController system. It provides endpoints for user registration, login, token issuance/refresh, and token verification using JWT. The service is built with FastAPI and uses MongoDB for user storage.

## Project structure
- `main.py` - Application entrypoint and router registration
- `routes/` - FastAPI routes (`auth.py` for login/refresh/verify)
- `services/` - Application service logic (`auth_service.py` for auth logic)
- `db/` - Database context (`context.py` singleton Motor client)
- `models/` - Pydantic models for requests and responses
- `tests/` - Pytest-based unit tests for endpoints and service logic

## Environment Variables
These environment variables are used to configure the Auth Service. Fill them via environment, CI runner, or Docker substitution.

- `MONGODB_URL` (required in production): MongoDB URI
- `JWT_ALGORITHM` (optional): JWT algorithm to use. Defaults to `HS256`. Set to RSA algorithm (e.g., `RS256`) to use private/public key pair.
- `JWT_SECRET` (required if `JWT_ALGORITHM=HS256`): Secret string used to sign HS256 JWTs.
- `JWT_KEY` (required if using RSA algorithm): The private key for signing 
- `JWT_CERTIFICATE` (required if using RSA algorithm): The public key used for verifying RSA signatures.
- `PORT` (optional): Port to run Uvicorn. Default in Docker & run steps is `8001`.

## Endpoints
- `GET /health` -> {"status":"ok"}
- `POST /auth/login` -> login an existing user; returns access token and refresh token
- `POST /auth/refresh` -> send `{ "refresh_token": "..." }` returns a new access token
- `GET /auth/verify` -> requires Authorization header `Bearer <access_token>`; returns `{ "username": "..." }`

Example registration with curl:

```bash
n/a (registration endpoint removed)
```

## Docker
Build container image (from repo root):

```bash
docker build -t auth_service:test ./auth_service
```

Run container with required env vars and port mapping:

```bash
docker run --rm -p 8001:8001 -e JWT_ALGORITHM=HS256 -e JWT_SECRET=test-secret -e MONGODB_URL="mongodb://host.docker.internal:27017/" auth_service:test
```

## Tests
The project uses `pytest`. Unit tests exist for the service logic and routes.

- Run tests locally (inside the repository root):

```bash
# From auth_service/ directory (venv active):
pytest tests -q
```

- Run tests inside Docker (CI workflow expects this):

```bash
docker build -t auth_service:test ./auth_service
docker run --rm -w /app -e PYTHONPATH=/app auth_service:test pytest auth_service/tests -q
```
