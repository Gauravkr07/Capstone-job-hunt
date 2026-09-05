import base64
import hashlib
import hmac
import json
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

PUBLIC_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/login",
    "/register",
    "/upload",
    "/chatbot",
}

PUBLIC_PREFIXES = (
    "/chatbot",
    "/chat",
)


class JWTError(Exception):
    pass


def _base64url_decode(input_str: str) -> bytes:
    padding = 4 - (len(input_str) % 4)
    if padding and padding < 4:
        input_str += "=" * padding
    return base64.urlsafe_b64decode(input_str.encode("utf-8"))


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("utf-8")


def encode(payload: dict, key: str, algorithm: str = "HS256") -> str:
    header = {"alg": algorithm, "typ": "JWT"}
    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    if algorithm == "HS256":
        signature = hmac.new(key.encode("utf-8"), msg=f"{header_b64}.{payload_b64}".encode("utf-8"), digestmod=hashlib.sha256).digest()
    else:
        raise JWTError("Unsupported signing algorithm")
    signature_b64 = _base64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode(token: str, key: str, algorithms=None):
    if algorithms is None:
        algorithms = ["HS256"]

    parts = token.split(".")
    if len(parts) != 3:
        raise JWTError("Invalid token format")

    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_base64url_decode(header_b64))
        payload = json.loads(_base64url_decode(payload_b64))
        signature = _base64url_decode(signature_b64)
    except Exception as exc:
        raise JWTError("Invalid token encoding") from exc

    alg = header.get("alg")
    if alg not in algorithms:
        raise JWTError("Unsupported signing algorithm")

    if alg == "HS256":
        expected = hmac.new(key.encode("utf-8"), msg=f"{header_b64}.{payload_b64}".encode("utf-8"), digestmod=hashlib.sha256).digest()
    else:
        raise JWTError("Unsupported signing algorithm")

    if not hmac.compare_digest(expected, signature):
        raise JWTError("Invalid token signature")

    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_value = int(exp)
        except (TypeError, ValueError) as exc:
            raise JWTError("Invalid exp claim") from exc
        if time.time() > exp_value:
            raise JWTError("Token has expired")

    return payload



class AuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        # Skip authentication for public routes and public route prefixes
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if any(request.url.path.startswith(prefix) for prefix in PUBLIC_PREFIXES):
            return await call_next(request)

        authorization = request.headers.get("Authorization")

        if not authorization:
            return JSONResponse(
                status_code=401,
                content={"message": "Authorization header missing"}
            )

        if not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"message": "Invalid authorization header"}
            )

        token = authorization.split(" ")[1]

        try:
            payload = decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM]
            )
        except JWTError:
            return JSONResponse(
                status_code=401,
                content={"message": "Invalid or expired token"}
            )


        # Store user information for later use
        request.state.user = payload

        response = await call_next(request)

        return response