# Capstone Job Hunt

FastAPI backend for a resume-vs-job matching (ATS-style) system. It ingests
resumes and job postings, extracts structured data (skills, experience,
industry, location, etc.) from each, embeds both into a shared vector space,
and scores/ranks job matches against a candidate's resume.

---

## 1. Tools & Stack

| Layer | Tool | Purpose |
|---|---|---|
| API framework | **FastAPI** | HTTP routing, request/response validation (Pydantic), auto-generated docs at `/docs`. |
| ASGI server | **Uvicorn** | Runs the FastAPI app. |
| Structured storage | **PostgreSQL** (async via SQLAlchemy + asyncpg) | Source of truth: users, resumes, resume section chunks, companies, jobs, skills, job↔skill links. |
| Vector storage | **Qdrant** | Similarity search over embeddings: resume profiles, resume section chunks, job postings. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`, local, free, 384-dim) | Turns resume/job summaries and text chunks into vectors, all in one shared space so resumes and JDs are directly comparable. |
| Structured JD extraction | **Groq API** (optional, hosted LLM) | Extracts title/company/location/skills/experience from raw, unstructured job postings. Falls back to regex/keyword heuristics automatically if no key is configured or the call fails. |
| File parsing | **pypdf**, **python-docx** | Extracts text from uploaded PDF/DOCX resumes. |
| Auth | Custom JWT (HMAC-SHA256, no external auth library) | `middleware/auth.py` — stateless bearer-token auth. |
| Containerization | **Docker Compose** | Runs `postgres`, `qdrant`, and `fastapi` together. |

---

## 2. Getting Started

### Prerequisites

- Docker and Docker Compose installed.
- (Optional) A free [Groq](https://console.groq.com) API key, for higher-quality job-posting extraction.

### Configuration

Create `dev.env` in the project root (gitignored, never committed):

```
OPENAI_API_KEY=

GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b

POSTGRES_USER=postgres
POSTGRES_PASSWORD=root
POSTGRES_DB=capstone_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://postgres:root@postgres:5432/capstone_db

QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
```

Notes:
- `POSTGRES_HOST`/`DATABASE_URL`/`QDRANT_URL` must point at the Docker Compose
  service names (`postgres`, `qdrant`) when running inside Docker. If you run
  the app directly on your host machine instead, change these to `localhost`.
- `GROQ_API_KEY` is optional. Leave it blank to disable LLM-based extraction
  entirely — the system automatically falls back to regex/keyword-based
  extraction with no loss of functionality, just lower accuracy on
  unusually-formatted job postings.
- `QDRANT_API_KEY` is optional; leave blank for local/no-auth Qdrant.

### Start the stack

```bash
docker compose up -d --build
```

This builds and starts three containers:

| Service | Port(s) | Purpose |
|---|---|---|
| `postgres` | 5432 | Structured data |
| `qdrant` | 6333 (REST), 6334 (gRPC) | Vector search index |
| `fastapi` | 8000 | The API |

### Verify it's running

- `GET http://localhost:8000/` — basic liveness check.
- `http://localhost:8000/docs` — interactive Swagger UI, lists every endpoint
  with a "Try it out" form (handles JSON encoding for you — no manual escaping
  needed, unlike hand-built `curl` commands).
- `http://localhost:8000/upload` — a minimal HTML page for pasting raw job
  posting text and submitting it without touching JSON at all.

### Stopping / rebuilding

```bash
docker compose down            # stop containers, keep data volumes
docker compose up -d --build   # rebuild after a code change and restart
```

---

## 3. Authentication

Every endpoint except `/`, `/upload`, `/register`, `/login`, and the docs
routes requires a **Bearer JWT** in the `Authorization` header.

### `POST /register`

Creates a new user account.

| Field | Type | Required | Notes |
|---|---|---|---|
| `username` | string | yes | Stored lowercase. |
| `password` | string | yes | Hashed (SHA-256) before storage. |
| `age` | integer | yes | |
| `graduated` | boolean | yes | |
| `linkedin_url` | URL | no | |
| `leetcode_url` | URL | no | |
| `hackerrank_url` | URL | no | |

**Response:** `{"message": "User registered successfully"}`

### `POST /login`

| Field | Type | Required |
|---|---|---|
| `username` | string | yes |
| `password` | string | yes |

**Response:**

| Field | Type | Notes |
|---|---|---|
| `bearer_token` | string | `"Bearer <jwt>"`, ready to drop into an `Authorization` header. |
| `jwt` | string | The raw token. |
| `token_type` | string | Always `"bearer"`. |
| `refresh_token` | string | Opaque random token (30-day validity noted server-side; no refresh endpoint currently implemented). |
| `jwt_expires_at` | integer | Unix timestamp, 1 hour from login. |
| `refresh_expires_at` | integer | Unix timestamp, 30 days from login. |

Use the token as: `Authorization: Bearer <jwt>` on every subsequent request.

---

## 4. Resume Endpoints (`routers/resume_opr.py`)

### `POST /jobs/resume`

Ingest a resume from raw text or a URL (JSON body).

| Field | Type | Required | Notes |
|---|---|---|---|
| `resume_text` | string | one of these two | Raw resume text. |
| `resume_url` | string | one of these two | A URL to fetch; bytes are parsed as PDF/DOCX/plain text based on content-type/extension. |

Pipeline: scrape skills/experience/industry → embed a profile summary →
split into sections (summary/experience/education/etc.) → embed each
section → persist to Postgres (`UserDetails`, `ResumeChunk`) and Qdrant
(`resume_profiles`, `resume_chunks` collections). Overwrites any existing
resume profile for the same user — one profile per user, always reflecting
the latest submission.

**Response (`ResumeFetchResponse`):**

| Field | Type |
|---|---|
| `username` | string |
| `skills` | list[string] |
| `experience` | integer or null |
| `industry_type` | string or null |
| `vector` | list[float] (384-dim) or null |

### `POST /jobs/resume/upload`

Same pipeline as above, but the input is a directly uploaded file
(`multipart/form-data`, field name `file`) — PDF, DOCX, or plain text,
detected by filename extension or content-type. Returns the same
`ResumeFetchResponse` shape.

### `GET /jobs/resume/me`

Returns the caller's own stored resume profile (`ResumeFetchResponse` shape,
same fields as above). 404 if the caller has never submitted a resume.

### `GET /jobs/resume/me/qdrant`

Debug endpoint. Returns the caller's raw Qdrant data directly (a direct
point lookup by ID, not a similarity search):

| Field | Type | Notes |
|---|---|---|
| `username` | string | |
| `profile_point` | object or null | `{id, vector, payload}` from the `resume_profiles` collection. |
| `chunk_points` | list[object] | Each `{id, vector, payload}` from the `resume_chunks` collection. |

---

## 5. Job Posting Endpoints (`routers/job.py`)

### `POST /jobs/exportJobDesc`

Ingest a job posting from structured fields you supply directly.

| Field | Type | Required | Notes |
|---|---|---|---|
| `jd` | string | yes | The full job description text. |
| `company_name` | string | yes | |
| `position` | string | yes | Used as the job title. |
| `location` | list[string] | no | A job can have multiple locations. |
| `source` | string | no | e.g. `"LinkedIn"`. |
| `source_url` | string | no | |
| `external_job_id` | string | no | The source system's own ID for this posting. |
| `employment_type` | string | no | Overrides whatever is inferred from `jd` if provided. |
| `salary_min` | integer | no | |
| `salary_max` | integer | no | |
| `salary_currency` | string | no | |

Skills, industry, experience range, remote type, and employment type (if not
overridden) are all extracted from `jd` via regex/keyword matching — this
endpoint never calls the LLM, since the caller already supplies the fields
the LLM would otherwise infer.

Deduplicates by content hash (`sha256(title + company + location +
description)`): submitting the same posting again returns the existing row
with `is_duplicate: true` instead of creating a duplicate.

**Response:** see `JobResponse` shape below.

### `POST /jobs/scrapeJobPosting`

Ingest a job posting from raw, unstructured text — no separate
title/company/location fields required.

| Field | Type | Required | Notes |
|---|---|---|---|
| `raw_text` | string | yes | The full raw job posting, exactly as copied from a job board. |
| `source` | string | no | |
| `source_url` | string | no | |
| `external_job_id` | string | no | |

Extraction order:
1. **Groq LLM** (if `GROQ_API_KEY` is set) — asks for structured JSON (title,
   company, location, required/preferred skills, experience range, industry,
   remote/employment type) in one call.
2. **Regex/keyword fallback** — used entirely if no LLM key is configured or
   the call fails; used partially (to fill in `title`/`company_name` only) if
   the LLM found everything else but missed one of those two fields.
3. Generic section headers ("About the job", "Job Description", etc.) are
   never accepted as a real title, and text with no genuine company name
   anywhere is rejected — both raise a `400` with a clear message rather than
   silently storing a useless placeholder.

Text is sanitized before extraction: smart quotes/dashes → ASCII, invisible
Unicode characters and non-breaking spaces removed (copy-paste artifacts),
without touching legitimate accented characters or newlines.

Same dedup/persist/embed pipeline as `/exportJobDesc` from that point on.

**Response (`JobResponse`):**

| Field | Type |
|---|---|
| `id` | integer |
| `title` | string |
| `company_name` | string |
| `description` | string (original, unmodified input) |
| `location` | list[string] or null |
| `remote_type` | `"remote"` / `"hybrid"` / `"onsite"` or null |
| `employment_type` | `"full_time"` / `"part_time"` / `"contract"` / `"internship"` or null |
| `experience_min` | integer or null |
| `experience_max` | integer or null |
| `industry_type` | string or null |
| `required_skills` | list[string] |
| `preferred_skills` | list[string] |
| `status` | string (always `"ACTIVE"` on ingestion) |
| `is_duplicate` | boolean |
| `created_at` | datetime |

### `GET /jobs`

List job postings from Postgres, newest first. **Public — no auth required.**

| Query param | Type | Default | Notes |
|---|---|---|---|
| `status` | string | none | e.g. `ACTIVE`. |
| `industry_type` | string | none | e.g. `fintech`. |
| `limit` | integer | 20 | 1–100. |
| `offset` | integer | 0 | |

**Response (`JobListResponse`):** `{total, limit, offset, jobs: [JobResponse, ...]}`

### `GET /jobs/{job_id}`

Single job detail by ID, including joined required/preferred skills.
Returns `JobResponse`. 404 if not found.

### `GET /jobs/{job_id}/qdrant`

Debug endpoint. Direct Qdrant point lookup for that job (not a similarity
search): `{job_id, point: {id, vector, payload} or null}`.

### `GET /jobs/user`

Debug endpoint only — echoes the authenticated JWT payload. Not
job-data-related; kept for auth troubleshooting.

---

## 6. ATS Matching Endpoints (`routers/ats_computing.py`)

Both endpoints score fit using a **hybrid formula**: cosine similarity
between resume and job embeddings (`vector_score`), combined with exact
skill-overlap against the job's required/preferred skills (`skill_score`,
weighted 80% required / 20% preferred), fused as
`0.4 * vector_score + 0.6 * skill_score`. If a job has no extracted skills at
all, `fused_score` falls back to `vector_score` alone rather than being
diluted by a meaningless zero (`has_skill_data: false` signals this case).

### `POST /jobs/atsCheck`

Score the caller's resume against one specific job.

| Field | Type | Required |
|---|---|---|
| `job_id` | integer | yes |

**Response (`AtsCheckResponse`):**

| Field | Type |
|---|---|
| `job_id` | integer |
| `vector_score` | float (0–1) |
| `skill_score` | float (0–1) |
| `fused_score` | float (0–1) |
| `has_skill_data` | boolean |
| `matched_required_skills` | list[string] |
| `missing_required_skills` | list[string] |
| `matched_preferred_skills` | list[string] |
| `missing_preferred_skills` | list[string] |

404 if the caller has no resume yet, or the job doesn't exist.

### `GET /jobs/recommendations`

Recommend active job postings for the caller's resume, ranked by likely fit
(highest `fused_score` first) — not just raw semantic similarity.

| Query param | Type | Default | Notes |
|---|---|---|---|
| `limit` | integer | 10 | 1–50. |

Two-stage retrieval + rerank: (1) vector similarity search across all
`ACTIVE` jobs pulls a wide candidate pool (up to 50) for recall, (2) each
candidate is rescored with the same hybrid formula as `/jobs/atsCheck` and
sorted by `fused_score` descending. Both Qdrant and Postgres lookups for the
candidate pool are batched (not one query per candidate), so cost stays flat
regardless of pool size.

**Response (`JobRecommendationsResponse`):**
`{recommendations: [{job: JobResponse, vector_score, skill_score, fused_score, has_skill_data, matched_required_skills, missing_required_skills, matched_preferred_skills, missing_preferred_skills}, ...]}`

404 if the caller has no resume yet.

---

## 7. Data Model Summary

```
users ──< user_details (resume profile: skills, experience, industry, vector)
  │        └──< resume_chunks (per-section text + vector)
  │
companies ──< jobs ──< job_skills >── skills
                │
                └── location is a JSON list (a job can have multiple locations)
```

- One `Company` row per company name, looked-up-or-created — never
  duplicated across multiple job postings from the same employer.
- `job_skills.importance` is `"required"` or `"preferred"`.
- Postgres is the source of truth; Qdrant is a derived, always-rebuildable
  search index kept in sync on every write.

---

## 8. Known Follow-ups

- **CPU-only torch** — the Docker image currently pulls the full
  CUDA-enabled `torch` build via `sentence-transformers`, downloading unused
  GPU/nvidia-* packages. Switching to the CPU-only wheel would shrink the
  image and speed up cold builds.
- **`AuthMiddleware` `SECRET_KEY`** (`middleware/auth.py`) is a hardcoded
  placeholder string — move to `dev.env` before any production use.
- **Skill/industry keyword taxonomy** (`helpers/constant.py`) is manually
  curated and English-only.
- **`GET /health`** is referenced in `middleware/auth.py`'s public-paths
  allowlist but no longer defined in `main.py` — use `GET /` for liveness
  instead.
- **`GET /jobs` has no auth dependency** — confirm this is intentional
  before relying on it as access-controlled.
- **Hybrid search (BM25 + vector) and reranking beyond the current
  skill-overlap formula** are deferred — the current fused-score approach
  covers the core use case; revisit if match quality needs improvement at
  larger scale.
