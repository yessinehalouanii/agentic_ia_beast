# app/api/auth_routes.py
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from urllib.parse import urlparse

from app.services.abi_auth import login_and_get_token
import traceback

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    base_url: str
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str


def _sanitize_base_url(raw_url: str) -> str:
    """
    Basic validation/sanitization for base_url used during login.
    Prevents obvious garbage and enforces http/https scheme.
    """
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="Missing base_url")

    parsed = urlparse(raw_url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="base_url must start with http:// or https://")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="base_url must include a hostname")

    # Normalise (remove trailing slash from path)
    normalised = parsed._replace(path=parsed.path.rstrip("/")).geturl()
    return normalised


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, response: Response):
    try:
        # ✅ Do NOT log password
        print(
            "LOGIN payload:",
            {
                "base_url": payload.base_url,
                "email": payload.email,
            },
        )

        safe_base_url = _sanitize_base_url(payload.base_url)

        token = login_and_get_token(
            safe_base_url,
            payload.email,
            payload.password,
        )

        # ✅ For production, secure should be True.
        #    If you're testing on http://localhost, you may need to set secure=False temporarily.
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,   # JS can't read it
            secure=True,     # ✅ only sent over HTTPS in prod
            samesite="lax",
            max_age=3600,
            path="/",
        )

        return LoginResponse(token=token)

    except Exception as e:
        # Log full details server-side for debugging
        print("LOGIN ERROR:", repr(e))
        traceback.print_exc()

        # But don't leak internals to the client
        raise HTTPException(status_code=400, detail="Login failed")
