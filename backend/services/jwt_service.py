import os
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError


class JWTAuthService:
    """Minimal JWT decoder for WebSocket authentication."""

    def __init__(self):
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        # Support both JWT_SECRET (HS256) and JWT_CERTIFICATE (RSA)
        self.secret = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
        self.public_key = os.getenv("JWT_CERTIFICATE")

    def decode_token(self, token: str) -> Dict[str, Any]:
        if self.algorithm == "HS256":
            if not self.secret:
                raise RuntimeError("JWT_SECRET/SECRET_KEY is not configured")
            key = self.secret
        else:
            if not self.public_key:
                raise RuntimeError("JWT_CERTIFICATE is not configured")
            key = self.public_key
        return jwt.decode(token, key, algorithms=[self.algorithm])

    def try_decode(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            return self.decode_token(token)
        except (JWTError, ExpiredSignatureError, RuntimeError):
            return None

