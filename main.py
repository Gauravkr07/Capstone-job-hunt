from fastapi import FastAPI

from db.connection import init_db
from middleware.auth import AuthMiddleware
from middleware.logging import LoggingMiddleware
from routers.api import router as jobs_router
from routers.auth import router as auth_router

app = FastAPI()
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.include_router(auth_router)
app.include_router(jobs_router)


@app.on_event("startup")
async def startup_event():
    await init_db()

@app.get("/")
async def first():
    return {"message": "Ai training is fun!"}

@app.get("/health")
async def health():
    return {
        "status": "OK"
    }