from datetime import datetime, timedelta

import jwt

from config import settings

_ALGO = "HS256"

def create_token(user_id: str, ttl_minutes: int = 60) -> str:
    exp = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    payload = {"sub": user_id, "exp": exp}
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGO)

def verify_token(token: str) -> str:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGO])
    return payload["sub"]
