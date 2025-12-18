# app/api/auth_routes.py
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from app.services.abi_auth import login_and_get_token
import traceback

router = APIRouter(prefix="/auth", tags=["Auth"])

class LoginRequest(BaseModel):
    base_url: str
    email: str
    password: str

class LoginResponse(BaseModel):
    token: str

@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, response: Response):
    try:
        print("LOGIN payload:", payload.model_dump())  # ✅ see what arrives

        token = login_and_get_token(payload.base_url, payload.email, payload.password)

        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=3600,
            path="/",
        )
        return LoginResponse(token=token)

    except Exception as e:
        print("LOGIN ERROR:", repr(e))
        traceback.print_exc()  # ✅ prints full stack trace
        raise HTTPException(status_code=400, detail=str(e))
