from fastapi import FastAPI
from auth_service.routes.auth import router as auth_router

app = FastAPI(title="Auth Service")

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(auth_router, tags=["auth"])