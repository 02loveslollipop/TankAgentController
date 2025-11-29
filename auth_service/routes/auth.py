from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError
from auth_service.models import User, Token, RefreshToken
from auth_service.services.auth_service import AuthService
import os

router = APIRouter()
auth_service = AuthService()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    try:
        if jwt_algorithm == "HS256":
            payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=[jwt_algorithm])
        else:
            payload = jwt.decode(token, os.getenv("JWT_CERTIFICATE"), algorithms=[jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await auth_service.get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@router.post("/register", response_model=Token)
async def register(user: User):
    db_user = await auth_service.get_user(user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    await auth_service.create_user(user.username, user.password)
    access_token = auth_service.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login(user: User):
    db_user = await auth_service.authenticate_user(user.username, user.password)
    if not db_user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = auth_service.create_access_token(data={"sub": user.username})
    refresh_token = auth_service.create_refresh_token(data={"sub": user.username})
    await auth_service.update_refresh_token(user.username, refresh_token)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/refresh", response_model=Token)
async def refresh_token(refresh: RefreshToken):
    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    try:
        if jwt_algorithm == "HS256":
            payload = jwt.decode(refresh.refresh_token, os.getenv("JWT_SECRET"), algorithms=[jwt_algorithm])
        else:
            payload = jwt.decode(refresh.refresh_token, os.getenv("JWT_CERTIFICATE"), algorithms=[jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    db_user = await auth_service.get_user(username)
    if not db_user or db_user.get("refresh_token") != refresh.refresh_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    access_token = auth_service.create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/verify")
async def verify_token(user=Depends(get_current_user)):
    return {"username": user["username"]}
