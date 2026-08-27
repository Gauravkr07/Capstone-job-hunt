from fastapi import FastAPI

from db.connection import init_db, seed_skills
from middleware.auth import AuthMiddleware
from middleware.logging import LoggingMiddleware
from routers.auth import router as auth_router
from routers.job import router as job_router
from routers.resume_opr import router as resume_router

app = FastAPI()
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(job_router)


@app.on_event("startup")
async def startup_event():
    await init_db()
    await seed_skills()

@app.get("/")
async def first():
    return {"message": "Welcome to the Job Hunt API. Visit /docs for API documentation."}

