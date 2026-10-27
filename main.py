import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
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
# Browsers must re-check the app's files on every load (a cheap 304 when
# unchanged); without this they can keep running a stale app.js after a deploy.
NO_CACHE = {"Cache-Control": "no-cache"}


class RevalidatedStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


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


# docs_url=None: /docs is served below so it can reuse the web app's sign-in.
app = FastAPI(title="Capstone Job Hunt", version="1.2.0", lifespan=lifespan, docs_url=None)
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.include_router(auth_router)
app.include_router(chatbot_router)
app.include_router(resume_router)
app.include_router(job_router)
app.include_router(ats_router)
app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def home():
    """React sign-in / register app (static/index.html + app.js)."""
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


# If the visitor is signed in to the web app (JWT in localStorage), hand that
# token to Swagger UI so "Try it out" works without pasting it into Authorize.
_SWAGGER_AUTO_AUTH = """<script>
(function () {
  var jwt = null;
  try {
    var s = JSON.parse(localStorage.getItem('jobhunt_session') || 'null');
    if (s && s.jwt && s.expiresAt * 1000 > Date.now()) jwt = s.jwt;
  } catch (e) {}
  if (!jwt) return;
  var tries = 0;
  var timer = setInterval(function () {
    // Swagger UI is a page-level `const ui`, not window.ui.
    if (typeof ui !== 'undefined' && ui.preauthorizeApiKey) {
      clearInterval(timer);
      ui.preauthorizeApiKey('HTTPBearer', jwt);
    } else if (++tries > 150) {
      clearInterval(timer);
    }
  }, 100);
})();
</script>"""


@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    page = get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} - API docs")
    return HTMLResponse(page.body.decode().replace("</body>", _SWAGGER_AUTO_AUTH + "</body>"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/upload")
async def upload_form():
    return FileResponse(STATIC_DIR / "upload.html", headers=NO_CACHE)


@app.get("/chatbot")
async def chatbot_ui():
    return FileResponse(STATIC_DIR / "chat_ui.html", headers=NO_CACHE)
