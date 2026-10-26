import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db.connection import init_db, seed_skills
from helpers import groq_client
from helpers.embeddings import warm_up_model
from middleware.auth import AuthMiddleware
from middleware.logging import LoggingMiddleware
from routers.ats_computing import router as ats_router
from routers.auth import router as auth_router
from routers.chatbot_flow import router as chatbot_router
from routers.job import router as job_router
from routers.resume_opr import router as resume_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    await seed_skills()
    # Load the embedding model in the background: the app must start answering
    # immediately even if the model download is slow or the network is flaky.
    # Requests that need it before it's ready just wait on the same lazy load.
    warm_up = asyncio.create_task(asyncio.to_thread(warm_up_model))
    yield
    warm_up.cancel()
    await groq_client.aclose()


app = FastAPI(title="Capstone Job Hunt", version="1.2.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.include_router(auth_router)
app.include_router(chatbot_router)
app.include_router(resume_router)
app.include_router(job_router)
app.include_router(ats_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def home():
    """React sign-in / register app (static/index.html + app.js)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/upload")
async def upload_form():
    return FileResponse(STATIC_DIR / "upload.html")


@app.get("/chatbot")
async def chatbot_ui():
    return FileResponse(STATIC_DIR / "chat_ui.html")
