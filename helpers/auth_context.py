from fastapi import HTTPException, Request


def require_username(request: Request) -> str:
    username = request.state.user.get("sub") if hasattr(request.state, "user") else None
    if not username:
        raise HTTPException(status_code=401, detail="User is not authenticated")
    return username


def get_caller_id(request: Request, fallback: str = "demo-user") -> str:
    """
    Best-effort caller identity for public (no-auth) routes, used only to keep
    separate chat histories apart: JWT subject if present, else the
    ?username= query param or X-Demo-User header, else a shared fallback.
    """
    username = request.state.user.get("sub") if hasattr(request.state, "user") else None
    if username:
        return username

    candidate = request.query_params.get("username") or request.headers.get("x-demo-user")
    if candidate and candidate.strip():
        return candidate.strip().lower()

    return fallback
