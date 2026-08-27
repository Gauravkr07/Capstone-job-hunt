# Capstone Job Hunt

FastAPI backend for a resume-vs-job matching (ATS-style) system. It ingests
resumes and job postings, extracts structured data (skills, experience,
industry, location, etc.) from each, and embeds both into a shared vector
space so a future matching endpoint can compare them.

- **Postgres** — source of truth: users, resumes (+ section chunks),
  companies, jobs, skills, and job↔skill links.
- **Qdrant** — searchable vector index (resume profiles, resume section
  chunks, job postings), kept in sync with Postgres on every write.
- **sentence-transformers** (`all-MiniLM-L6-v2`, local, free) — embeds
  resume/job summaries and chunks into 384-dim vectors, all in one shared
  space so resumes and JDs are directly comparable.
- **Groq LLM (optional)** — structured extraction of title/company/location/
  skills/experience from raw, unstructured job postings, with an automatic
  regex/keyword fallback when no key is configured or the call fails.

---

## Architecture at a glance

```
Resume ingestion                          Job ingestion
  raw text / file upload / URL              structured fields OR raw text
     |                                         |
     v                                         v
helpers/resume_scraper.py                helpers/jd_scraper.py
  (skills, experience, industry)            -> tries helpers/llm_jd_extractor.py (Groq)
     |                                            first, falls back to regex/keyword
     v                                            extraction otherwise
helpers/resume_chunker.py                       |
  (splits into sections:                        v
   summary/experience/education/...)      services/job_service.py
     |                                      -> Company (get-or-create)
     v                                      -> Job + JobSkill rows (Postgres)
services/resume_service.py                  -> embed + upsert to Qdrant
  -> UserDetails row (Postgres)               job_postings collection
  -> ResumeChunk rows (Postgres)
  -> embed + upsert to Qdrant
     resume_profiles + resume_chunks collections
```

Postgres and Qdrant are **not** redundant: Postgres is the auditable system of
record and supports structured filters (experience > N, industry = X, skill =
Y); Qdrant exists purely for similarity search once the matching endpoint
compares embedded resumes against embedded job postings.

---

## Project layout

```
routers/          thin FastAPI route handlers (auth, resume, job)
services/         business logic: DB writes, dedup, Qdrant sync
helpers/          scraping/parsing/embedding building blocks
db/                SQLAlchemy models, Qdrant client wrapper, Pydantic schemas
middleware/        JWT auth, request logging
```

| Router | Endpoints |
|---|---|
| `routers/auth.py` | `POST /register`, `POST /login` |
| `routers/resume_opr.py` | `POST /jobs/resume` (text or URL), `POST /jobs/resume/upload` (PDF/DOCX/text file), `GET /jobs/resume/me` (own profile), `GET /jobs/resume/me/qdrant` (raw Qdrant points, debug) |
| `routers/job.py` | `POST /jobs/exportJobDesc` (structured fields), `POST /jobs/scrapeJobPosting` (raw text, auto-extracted), `GET /jobs` (list, paginated), `GET /jobs/{job_id}` (detail), `GET /jobs/{job_id}/qdrant` (raw Qdrant point, debug), `GET /jobs/user` (debug) |

Each router delegates to a matching service:

- `services/resume_service.py` — `save_resume_profile()`: scrape → embed
  profile summary → chunk into sections → embed each chunk → persist
  everything to Postgres (`UserDetails`, `ResumeChunk`) and Qdrant
  (`resume_profiles`, `resume_chunks`).
- `services/job_service.py` — `ingest_job()`: extract fields → dedupe by
  content hash → get-or-create `Company` → persist `Job` + `JobSkill` rows →
  embed → upsert to Qdrant (`job_postings`). `build_job_response()` shapes the
  API response.

## Key data model

```
users ──< user_details (resume profile: skills, experience, industry, vector)
  │        └──< resume_chunks (per-section text + vector)
  │
companies ──< jobs ──< job_skills >── skills
                │
                └── location is a JSON list (a job can have multiple locations)
```

- `jobs.content_hash` = `sha256(title + company + location + description)` —
  identical postings (even re-submitted from different sources) dedupe
  automatically; `POST /jobs/exportJobDesc`/`scrapeJobPosting` return the
  existing row with `is_duplicate: true` instead of inserting a new one.
- `job_skills.importance` is `"required"` or `"preferred"` — job JDs get their
  skills classified by proximity to "preferred/nice to have/bonus" markers
  (regex path) or directly by the LLM (Groq path).
- One company can have many jobs — `Company` is looked-up-or-created by name,
  never duplicated per job.

## Read endpoints (Postgres + Qdrant)

Every ingestion endpoint has a matching GET to inspect what was actually
stored, split into two layers:

| Endpoint | Source | Returns |
|---|---|---|
| `GET /jobs/resume/me` | Postgres (`UserDetails`) | Caller's own resume profile: skills, experience, industry, embedding vector. 404 if none. |
| `GET /jobs/resume/me/qdrant` | Qdrant | Direct point lookup (not similarity search) for the caller: the `resume_profiles` point + all `resume_chunks` points, each with its real stored vector and payload. |
| `GET /jobs` | Postgres (`jobs`) | Paginated list, newest first. Optional `?status=`/`?industry_type=` filters, `?limit=`/`?offset=`. |
| `GET /jobs/{job_id}` | Postgres (`jobs` + `job_skills`) | Single job detail, including joined required/preferred skills. 404 if not found. |
| `GET /jobs/{job_id}/qdrant` | Qdrant | Direct point lookup for that job: its real stored vector + compact payload. |

The `/qdrant` endpoints are for verification/debugging — they bypass
similarity search entirely and fetch the exact point by its deterministic ID
(same ID scheme the upsert functions use), so you can confirm a resume/job
actually made it into the vector store with the payload you expect.

`db/vector_store.py` backs these with `get_resume_vector_point()`,
`get_resume_chunk_points()`, and `get_job_vector_point()` — separate from the
similarity-search functions (`search_similar_resumes`, `search_resume_chunks`,
`search_similar_jobs`), which remain unused/scaffolded for the future ATS
matching endpoint.

## Job posting extraction: LLM-first, regex-fallback

`POST /jobs/scrapeJobPosting` accepts a single `raw_text` field (e.g. pasted
directly from LinkedIn/Naukri/a job board) and extracts everything from it:

1. **Tries Groq** (`helpers/llm_jd_extractor.py`) first if `GROQ_API_KEY` is
   set — asks for structured JSON (title, company, location, skills split
   required/preferred, experience range, industry, remote/employment type).
   Free tier at [console.groq.com](https://console.groq.com); current default
   model is `openai/gpt-oss-120b` (set via `GROQ_MODEL`).
2. **Falls back to regex/keyword heuristics** (`helpers/jd_scraper.py`) when
   no key is configured, the call fails, or the LLM doesn't find a
   title/company — label-line detection ("Title:", "Company:", "Location:"),
   common JD phrasing ("`<Company>` is a/hiring...", "join `<Company>`'s...",
   "hiring a `<Title>` to/for..."), and the same skill-keyword matching used
   for resumes (word-boundary-aware, so "developer" doesn't false-match "ev",
   "internal" doesn't false-match "intern", etc.).
3. **Hybrid merge**: if the LLM call succeeds but is missing title or company
   specifically, those two fields fall back to regex detection while
   everything else (skills, industry, experience, remote/employment type)
   still comes from the LLM — so a partial LLM result isn't thrown away
   entirely.

`POST /jobs/exportJobDesc` (structured fields: `jd`, `company_name`,
`position`, etc.) always uses the regex/keyword path only — no LLM call,
since the caller already supplies the fields the LLM would otherwise infer.

---

## Running it

```bash
docker compose up -d --build
```

Starts three containers:

| Service    | Port(s)                  | Purpose                            |
|------------|---------------------------|-------------------------------------|
| `postgres` | 5432                       | structured data (source of truth)   |
| `qdrant`   | 6333 (REST), 6334 (gRPC)   | vector search index                 |
| `fastapi`  | 8000                       | the API                             |

### Config (`dev.env`, gitignored — not committed)

```
POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB / POSTGRES_HOST / POSTGRES_PORT
DATABASE_URL          # postgresql+asyncpg://...
QDRANT_URL            # http://qdrant:6333 inside Docker
QDRANT_API_KEY        # blank for local/no-auth Qdrant
GROQ_API_KEY          # optional -- blank disables LLM extraction, falls back to regex
GROQ_MODEL            # default: openai/gpt-oss-120b
```

Inside Docker, `POSTGRES_HOST`/`DATABASE_URL`/`QDRANT_URL` must point at the
container service names (`postgres`, `qdrant`), not `localhost` — if you run
the app directly on your host instead of in Docker, flip those back to
`localhost`.

### Smoke test

```bash
curl http://localhost:8000/

curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPass123","age":25,"graduated":true}'

TOKEN=$(curl -s -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPass123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['jwt'])")

# Resume ingestion
curl -X POST http://localhost:8000/jobs/resume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"resume_text":"5 years experience in Python, FastAPI, PostgreSQL, Docker, AWS."}'

# Read it back
curl http://localhost:8000/jobs/resume/me -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/jobs/resume/me/qdrant -H "Authorization: Bearer $TOKEN"

# Job ingestion from raw text (no separate title/company fields needed)
python3 -c "
import json
payload = {'raw_text': open('jd.txt').read(), 'source': 'LinkedIn'}
json.dump(payload, open('payload.json', 'w'))
"
curl -X POST http://localhost:8000/jobs/scrapeJobPosting \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @payload.json

# List and inspect jobs
curl "http://localhost:8000/jobs?limit=10"
curl http://localhost:8000/jobs/1 -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/jobs/1/qdrant -H "Authorization: Bearer $TOKEN"
```

**Note on curl + multi-line JD text**: never inline multi-line text directly
in `-d '...'` — literal newlines inside a JSON string produce an "Invalid
control character" error. Always build the JSON payload through something
that escapes it properly (like the `json.dump` above), or paste the text into
Swagger UI's "Try it out" form at `http://localhost:8000/docs` instead.

---

## What's next

**The ATS matching endpoint (not yet built).** Planned shape:
`POST /jobs/atsCheck` — given a job (or its stored vector) and a user's
resume vector, compute cosine similarity for a quick match score, and/or
compare against resume section chunks (`resume_chunks` collection) for
explainable "why this matched" results using the required/preferred skill
split already stored per job.

- `db/vector_store.py` already has `search_similar_resumes`,
  `search_resume_chunks`, and `search_similar_jobs` ready for this — currently
  unused/scaffolded, intentionally kept for when this endpoint is built.
- Hybrid search (keyword/BM25 + vector) is deferred until this endpoint
  exists and reveals what pure vector similarity is missing.

## Known follow-ups

- **CPU-only torch** — the Docker image pulls the full CUDA-enabled `torch`
  build via `sentence-transformers`, downloading unused GPU/nvidia-* packages
  (~2GB extra, slow first/no-cache build). Switching to the CPU-only wheel
  would shrink the image and speed up cold builds; not done yet.
- **`AuthMiddleware` `SECRET_KEY`** ([middleware/auth.py](middleware/auth.py))
  is currently a hardcoded placeholder string — should move to `dev.env`
  before this goes anywhere near production.
- **Skill/industry keyword taxonomy** (`helpers/constant.py`) is manually
  curated and English-only; word-boundary matching prevents false-positive
  substring matches, but it will still miss skills/phrasing not in the list.
- **`GET /jobs/user`** is a leftover debug endpoint (echoes the JWT payload)
  — not job-related, kept for auth troubleshooting only.
- **`GET /health`** was removed from `main.py` at some point but is still
  listed in `middleware/auth.py`'s public-paths allowlist — orphaned config,
  not currently causing issues since the route just doesn't exist. Use
  `GET /` for a liveness check instead, or re-add `/health` if something
  external depends on it.
- **`GET /jobs` has no auth dependency** — it's a public, unauthenticated
  endpoint (unlike every other `/jobs/*` route). Confirm this is intentional
  before relying on it as an access-controlled listing.
