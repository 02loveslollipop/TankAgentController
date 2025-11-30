from fastapi import FastAPI
from auth_service.routes.auth import router as auth_router

app = FastAPI(title="Auth Service")

app.add_api_route("/health", endpoint=lambda: {"status": "ok"}, methods=["GET"])

app.include_router(auth_router, prefix="/auth", tags=["auth"])