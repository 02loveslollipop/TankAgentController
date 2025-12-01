import base64
import os
from typing import Any, Dict, Optional
from binascii import Error as BinasciiError

from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError


class JWTAuthService:
    """Minimal JWT decoder for WebSocket authentication."""

    def __init__(self):
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        # Support both JWT_SECRET (HS256) and JWT_CERTIFICATE (RSA)
        self.secret = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
        self.public_key = self._load_key(os.getenv("JWT_CERTIFICATE"))

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

    def _load_key(self, raw_value: str | None) -> str | None:
        """
        Return a PEM key string, decoding base64 input when necessary.
        Accepts either raw PEM text or a base64-encoded PEM.
        """
        if not raw_value:
            return raw_value
        if "BEGIN" in raw_value and "END" in raw_value:
            return raw_value
        try:
            decoded = base64.b64decode(raw_value).decode("utf-8")
            return decoded
        except (BinasciiError, UnicodeDecodeError):
            return raw_value
