from fastapi import HTTPException, Request


def require_username(request: Request) -> str:
    username = request.state.user.get("sub") if hasattr(request.state, "user") else None
    if not username:
        raise HTTPException(status_code=401, detail="User is not authenticated")
    return username
