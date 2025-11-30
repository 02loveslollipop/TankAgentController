from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from jose.exceptions import ExpiredSignatureError

from auth_service.models import User, Token, RefreshToken
from auth_service.services.auth_service import AuthService

router = APIRouter()
auth_service = AuthService()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = auth_service.decode_token(token)
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

@router.post("/login", response_model=Token)
async def login(user: User):
    access_token, refresh_token = await auth_service.login_user(user.username, user.password)
    if not access_token:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/refresh", response_model=Token)
async def refresh_token(refresh: RefreshToken):
    try:
        access_token = await auth_service.refresh_access_token(refresh.refresh_token)
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/verify")
async def verify_token(user=Depends(get_current_user)):
    return {"username": user["username"]}
