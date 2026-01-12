from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from .models import User


@dataclass(frozen=True)
class SessionUser:
    id: int
    username: str
    is_admin: bool


def get_session_user(request: Request) -> SessionUser | None:
    user = request.session.get("user")
    if not user:
        return None
    return SessionUser(id=int(user["id"]), username=str(user["username"]), is_admin=bool(user.get("is_admin", False)))


def require_user(request: Request) -> SessionUser:
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user


def require_admin(request: Request) -> SessionUser:
    user = require_user(request)
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return user


def login_session(request: Request, user: User) -> None:
    request.session["user"] = {"id": user.id, "username": user.username, "is_admin": user.is_admin}


def logout_session(request: Request) -> None:
    request.session.pop("user", None)
