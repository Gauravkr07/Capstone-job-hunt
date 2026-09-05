from datetime import datetime, timedelta
import hashlib
import json
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from typing import Any

from db.connection import get_session
from db.models import User, UserDetails
from db.schemas import LoginResponse, UserCreate
from db.vector_store import search_resume_chunks, search_similar_jobs
from helpers.chatbot_rag import answer_user_question, rerank_jobs, rerank_resume_chunks, stream_answer_user_question
from helpers.embeddings import embed_text
from middleware.auth import SECRET_KEY, ALGORITHM, encode
from services.resume_service import get_resume_profile

router = APIRouter()
security = HTTPBearer()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_authenticated_username(request: Request, fallback: str = "demo-user") -> str:
    username = request.state.user.get("sub") if hasattr(request.state, "user") else None
    if username:
        return username

    username = request.query_params.get("username") if hasattr(request, "query_params") else None
    if username and username.strip():
        return username.strip().lower()

    if request.headers.get("x-demo-user"):
        return request.headers.get("x-demo-user").strip().lower()

    return fallback


def serialize_user_details(user: User, user_details: UserDetails | None = None) -> dict:
    return {
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "age": user.age,
        "graduated": user.graduated,
        "linkedin_url": str(user.linkedin_url) if user.linkedin_url else None,
        "leetcode_url": str(user.leetcode_url) if user.leetcode_url else None,
        "hackerrank_url": str(user.hackerrank_url) if user.hackerrank_url else None,
        "resume_skills": user_details.skills if user_details and user_details.skills else [],
        "experience": user_details.experience if user_details else None,
        "industry_type": user_details.industry_type if user_details else None,
        "resume_text_available": bool(user_details and user_details.resume_text and user_details.resume_text.strip()),
        "resume_text": user_details.resume_text if user_details else None,
    }


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(user_data: UserCreate, session: AsyncSession = Depends(get_session)):
    username = user_data.username.strip().lower()

    existing = await session.execute(select(User).where(User.username == username))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="User already exists")

    linkedin_url = str(user_data.linkedin_url) if user_data.linkedin_url else None
    leetcode_url = str(user_data.leetcode_url) if user_data.leetcode_url else None
    hackerrank_url = str(user_data.hackerrank_url) if user_data.hackerrank_url else None

    user = User(
        username=username,
        password_hash=_hash_password(user_data.password),
        full_name=user_data.full_name,
        email=user_data.email,
        age=user_data.age,
        graduated=user_data.graduated,
        linkedin_url=linkedin_url,
        leetcode_url=leetcode_url,
        hackerrank_url=hackerrank_url,
    )
    session.add(user)
    await session.commit()

    return {"message": "User registered successfully"}


@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest, session: AsyncSession = Depends(get_session)):
    username = credentials.username.strip().lower()
    result = await session.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    if not user or user.password_hash != _hash_password(credentials.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    jwt_expiration = int(time.time()) + 3600
    payload = {
        "sub": username,
        "exp": jwt_expiration,
    }
    token = encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    bearer_token = f"Bearer {token}"

    refresh_token = secrets.token_urlsafe(32)
    refresh_expires_dt = datetime.utcnow() + timedelta(days=30)
    refresh_expires = int(refresh_expires_dt.timestamp())

    return {
        "bearer_token": bearer_token,
        "jwt": token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "jwt_expires_at": jwt_expiration,
        "refresh_expires_at": refresh_expires,
    }


@router.post("/auth/login", response_model=LoginResponse)
async def login_alias(credentials: LoginRequest, session: AsyncSession = Depends(get_session)):
    return await login(credentials, session)


@router.post("/auth/register")
async def register_alias(user_data: UserCreate, session: AsyncSession = Depends(get_session)):
    return await register(user_data, session)


@router.get("/me", dependencies=[Depends(security)])
async def get_my_user_details(request: Request, session: AsyncSession = Depends(get_session)):
    username = get_authenticated_username(request)

    user_result = await session.execute(select(User).where(User.username == username))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_details = await get_resume_profile(session, username)
    return serialize_user_details(user, user_details)


@router.get("/chatbot/users")
async def list_registered_users(session: AsyncSession = Depends(get_session)):
    """Lists registered usernames, so the chatbot UI can offer a 'look up user' picker."""
    result = await session.execute(select(User.username).order_by(User.username))
    return [row[0] for row in result.all()]


@router.get("/chatbot/user-details")
async def get_user_details_for_chatbot(request: Request, session: AsyncSession = Depends(get_session)):
    """Dedicated route for a user-details-only chatbot. Returns only profile + resume summary."""
    username = get_authenticated_username(request)

    user_result = await session.execute(select(User).where(User.username == username))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_details = await get_resume_profile(session, username)
    return serialize_user_details(user, user_details)


@router.get("/chatbot/user-details/stream")
async def stream_user_details_for_chatbot(request: Request, session: AsyncSession = Depends(get_session)):
    """SSE stream for a lightweight chatbot that only needs the authenticated user's details."""
    username = get_authenticated_username(request)

    user_result = await session.execute(select(User).where(User.username == username))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_details = await get_resume_profile(session, username)
    payload = serialize_user_details(user, user_details)

    async def event_generator():
        yield "event: user_details\n"
        yield f"data: {json.dumps(payload)}\n\n"
        yield "event: done\n"
        yield "data: complete\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class ChatMessageRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    target_username: str | None = None


CHAT_CONVERSATIONS: dict[str, dict[str, Any]] = {}


def _get_or_create_conversation(username: str, conversation_id: str | None) -> dict[str, Any]:
    bucket = CHAT_CONVERSATIONS.setdefault(username, {})
    if conversation_id and conversation_id in bucket:
        return bucket[conversation_id]
    new_id = conversation_id or str(int(time.time() * 1000))
    conv = {"id": new_id, "title": "New conversation", "messages": []}
    bucket[new_id] = conv
    return conv


@router.get("/chat/conversations")
async def list_chat_conversations(request: Request):
    username = get_authenticated_username(request)
    conversations = list(CHAT_CONVERSATIONS.get(username, {}).values())
    return [{"id": item["id"], "title": item["title"]} for item in conversations]


@router.get("/chat/conversations/{conversation_id}")
async def get_chat_conversation(request: Request, conversation_id: str):
    username = get_authenticated_username(request)
    conversation = CHAT_CONVERSATIONS.get(username, {}).get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conversation["id"], "title": conversation["title"], "messages": conversation["messages"]}


async def _load_user_and_profile(session: AsyncSession, username: str) -> tuple[User, dict[str, Any]]:
    user_result = await session.execute(select(User).where(User.username == username))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail=f"No registered user named '{username}'")

    user_details = await get_resume_profile(session, username)
    return user, serialize_user_details(user, user_details)


def _resolve_target_username(request: Request, payload: "ChatMessageRequest") -> str:
    """
    The user the chatbot should answer *about*: an explicit target_username
    on the request (looking up another registered user), else the caller's
    own username. Kept separate from get_authenticated_username's result,
    which is also used to key the (per-caller) conversation history.
    """
    target = (payload.target_username or "").strip().lower()
    return target or get_authenticated_username(request)


def _retrieve_and_rerank(query: str, username: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Hybrid RAG retrieval: over-fetch candidates from Qdrant (vector search),
    then rerank by blending vector similarity with lexical keyword overlap
    against the raw query, keeping only the top candidates to ground the LLM
    response in. Over-fetching (15) before reranking down to 5 gives the
    lexical signal a wider pool to pick from than plain top-5 vector search.
    """
    query_vector = embed_text(query)
    if not query_vector:
        return [], []

    raw_chunks = search_resume_chunks(query_vector, limit=15, username=username)
    raw_jobs = search_similar_jobs(query_vector, limit=15)

    resume_chunks = rerank_resume_chunks(query, raw_chunks, top_k=5)
    jobs = rerank_jobs(query, raw_jobs, top_k=5)
    return resume_chunks, jobs


def _build_sources(resume_chunks: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = [
        {
            "id": f"chunk-{c.get('section')}-{c.get('chunk_index')}",
            "title": f"Resume: {c.get('section', 'unknown')}",
            "kind": "resume_chunk",
            "score": c.get("rerank_score", c.get("score")),
        }
        for c in resume_chunks
    ]
    sources += [
        {
            "id": f"job-{j.get('job_id')}",
            "title": j.get("title", "Job posting"),
            "kind": "job",
            "score": j.get("rerank_score", j.get("score")),
        }
        for j in jobs
    ]
    return sources


def _fallback_answer(details: dict[str, Any]) -> str:
    name = details.get("full_name") or details["username"]
    return (
        f"**{name}** ({details['username']})\n"
        f"- Email: {details.get('email') or 'not provided'}\n"
        f"- Age: {details['age']}, {'graduate' if details['graduated'] else 'non-graduate'}\n"
        f"- Skills: {', '.join(details['resume_skills']) or 'not added yet'}\n"
        f"- Experience: {details['experience'] or 'not specified'}\n"
        f"- Industry: {details['industry_type'] or 'not specified'}"
    )


@router.post("/chat")
async def chat_with_user(payload: ChatMessageRequest, request: Request, session: AsyncSession = Depends(get_session)):
    caller = get_authenticated_username(request)
    target_username = _resolve_target_username(request, payload)
    conversation = _get_or_create_conversation(caller, payload.conversation_id)

    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    history = list(conversation["messages"])
    conversation["messages"].append({"role": "user", "content": user_message})

    _, details = await _load_user_and_profile(session, target_username)
    resume_chunks, jobs = _retrieve_and_rerank(user_message, target_username)
    sources = _build_sources(resume_chunks, jobs)

    response_text = await answer_user_question(user_message, details, resume_chunks, jobs, history=history)
    if response_text is None:
        response_text = _fallback_answer(details)
        sources = [{"id": "profile", "title": "User profile", "kind": "profile", "score": 1.0}]

    if not conversation["title"] or conversation["title"] == "New conversation":
        conversation["title"] = user_message[:40] if user_message else "New conversation"

    assistant_payload = {
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    }
    conversation["messages"].append(assistant_payload)

    return {
        "conversation_id": conversation["id"],
        "message": response_text,
        "sources": sources,
    }


@router.post("/chat/stream")
async def chat_with_user_stream(payload: ChatMessageRequest, request: Request, session: AsyncSession = Depends(get_session)):
    """SSE variant of /chat: streams the assistant's answer token-by-token as it's generated."""
    caller = get_authenticated_username(request)
    target_username = _resolve_target_username(request, payload)
    conversation = _get_or_create_conversation(caller, payload.conversation_id)

    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    history = list(conversation["messages"])
    conversation["messages"].append({"role": "user", "content": user_message})

    _, details = await _load_user_and_profile(session, target_username)
    resume_chunks, jobs = _retrieve_and_rerank(user_message, target_username)
    sources = _build_sources(resume_chunks, jobs)

    if not conversation["title"] or conversation["title"] == "New conversation":
        conversation["title"] = user_message[:40] if user_message else "New conversation"

    async def event_generator():
        yield f"event: meta\ndata: {json.dumps({'conversation_id': conversation['id'], 'sources': sources})}\n\n"

        collected = ""
        async for delta in stream_answer_user_question(user_message, details, resume_chunks, jobs, history=history):
            collected += delta
            yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"

        final_sources = sources
        if not collected:
            collected = _fallback_answer(details)
            final_sources = [{"id": "profile", "title": "User profile", "kind": "profile", "score": 1.0}]
            yield f"event: token\ndata: {json.dumps({'delta': collected})}\n\n"

        conversation["messages"].append({"role": "assistant", "content": collected, "sources": final_sources})
        yield f"event: done\ndata: {json.dumps({'message': collected, 'sources': final_sources})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
