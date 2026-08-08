# Capstone Job Hunt

FastAPI backend that scrapes resumes into structured profiles (skills, experience,
industry) and embeds them as vectors for future ATS (resume-vs-JD) matching.

- **Postgres** — source of truth: users, job posts, and resume profiles
  (skills/experience/industry + a copy of the embedding vector).
- **Qdrant** — searchable vector index of resume embeddings, kept in sync with
  Postgres on every resume update.
- **sentence-transformers** (`all-MiniLM-L6-v2`, local, free) — turns
  `skills + industry + experience` into a 384-dim embedding.

## Architecture at a glance

```
resume text
   -> helpers/resume_scraper.py   (skills, experience, industry_type)
   -> helpers/embeddings.py       (embed_profile -> 384-dim vector)
   -> db/models.py UserDetails    (Postgres: structured fields + vector copy)
   -> db/vector_store.py          (Qdrant: upsert_resume_vector, searchable)
```

Postgres and Qdrant are **not** redundant: Postgres is the auditable system of
record and supports structured filters (experience > N, industry = X); Qdrant
exists purely for similarity search once the ATS-matching endpoint is built
(embed a JD with `embed_text()`, compare against stored resume vectors).

---

## 1. First execution plan (what was built, and why)

1. **`helpers/resume_scraper.py`** — regex/keyword extraction of skills
   (`helpers/constant.py` keyword dictionary), years of experience, and an
   inferred industry from free-text resume content.
2. **`helpers/embeddings.py`** — composes those fields into one string
   (`"Skills: ... Industry: ... Experience: N years"`) and embeds it locally
   with `sentence-transformers`. Also exposes `embed_text()` for embedding a
   raw job description later, in the same vector space as resumes.
3. **`db/models.py`** — `UserDetails.vector` stores the real embedding (was
   previously just the skills list duplicated); `embedding_model` records
   which model produced it, so future model upgrades are traceable.
4. **`db/vector_store.py`** — Qdrant client wrapper. `upsert_resume_vector`
   writes/overwrites a point keyed by a deterministic UUID derived from the
   username, so re-scraping a resume always updates the same point (no stale
   duplicates). `search_similar_resumes` is ready for the future
   many-candidates search case, with optional industry filtering.
5. **`routers/api.py`** (`POST /jobs/resume`) — wires it together: scrape →
   embed → save to Postgres → upsert to Qdrant, on every call.
6. **Logging** — `helpers/logger.py` (shared logger factory),
   `middleware/logging.py` (per-request HTTP logging), and embedding/vector
   stage logging (durations, empty-input warnings) in `helpers/embeddings.py`
   and `db/vector_store.py`.
7. **Containerization** — `Dockerfile` (Python 3.12 slim) + `docker-compose.yml`
   running `postgres`, `qdrant`, and `fastapi` together. All config comes from
   `dev.env` (gitignored, not committed) via `env_file:` — no hardcoded
   credentials in the compose file or `config.py`.

This was verified end-to-end: registered a test user, logged in, called
`POST /jobs/resume` with sample resume text, and confirmed the extracted
skills/experience/industry, the 384-dim vector, and the `embedding_model` tag
all landed correctly in both Postgres and Qdrant.

---

## 2. How to proceed (running it, and what's next)

### Run locally (Docker)

```bash
docker compose up -d --build
```

Starts three containers:

| Service    | Port(s)          | Purpose                          |
|------------|-------------------|-----------------------------------|
| `postgres` | 5432               | structured data (source of truth) |
| `qdrant`   | 6333 (REST), 6334 (gRPC) | vector search index         |
| `fastapi`  | 8000               | the API                           |

Config lives in `dev.env` (not committed — copy the format from an existing
teammate's file or reconstruct from `config.py`'s `os.getenv` calls). Inside
Docker, `dev.env` must point `POSTGRES_HOST`/`DATABASE_URL`/`QDRANT_URL` at
the container service names (`postgres`, `qdrant`), not `localhost` — if you
run the app directly on your host machine instead of in Docker, flip those
back to `localhost`.

Quick smoke test:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPass123","age":25,"graduated":true}'

TOKEN=$(curl -s -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPass123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['jwt'])")

curl -X POST http://localhost:8000/jobs/resume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"resume_text":"5 years experience in Python, FastAPI, PostgreSQL, Docker, AWS."}'
```

### Next: the ATS matching endpoint (not yet built)

Planned shape: `POST /jobs/atsCheck` — takes a job description, embeds it with
`embed_text()` (same model/vector space as resumes), and compares it against
the calling user's stored resume vector via cosine similarity. For a single
user-vs-JD check, this doesn't need `search_similar_resumes` at all — that
function is reserved for a future "match many candidates against one JD"
feature.

### Known follow-ups / things to revisit

- **Hybrid search** (keyword/BM25 + vector) — deferred until the ATS endpoint
  exists and we know what's actually missing from pure vector similarity.
- **CPU-only torch** — the Docker image currently pulls the full CUDA-enabled
  `torch` build via `sentence-transformers`, downloading unused GPU/nvidia-*
  packages (~2GB extra, slow first build). Switching to the CPU-only wheel
  would shrink the image and speed up cold builds; not done yet.
- **`AuthMiddleware` `SECRET_KEY`** ([middleware/auth.py](middleware/auth.py))
  is currently a hardcoded placeholder string — should move to `dev.env`
  before this goes anywhere near production.
- **`.gitignore`** currently excludes `README.md` from version control — if
  this file should be committed, remove that line.
