"""
Chatbot-facing API surface: no-auth-required routes for looking up a
registered candidate's technical profile (RAG over Qdrant + Postgres) and
chatting about it. All routes here are exempt from auth -- see
middleware.auth.PUBLIC_PREFIXES ("/chatbot", "/chat").
"""
import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.models import User
from db.vector_store import search_resume_chunks, search_similar_jobs
from helpers.chatbot_rag import rerank_jobs, rerank_resume_chunks, stream_answer_user_question
from helpers.embeddings import embed_text
from routers.auth import get_authenticated_username, serialize_user_details
from services.resume_service import get_resume_profile

router = APIRouter()

# This chatbot deployment is scoped to a single candidate -- every chat
# answers about this user's profile/resume regardless of who's asking or
# what target_username is sent, so it's safe to share as a personal link.
CHATBOT_LOCKED_USERNAME = "gaurav"


@router.get("/chatbot/user-details")
async def get_user_details_for_chatbot(session: AsyncSession = Depends(get_session)):
    """Dedicated route for the (single-candidate) chatbot. Returns only CHATBOT_LOCKED_USERNAME's profile + resume summary."""
    user_result = await session.execute(select(User).where(User.username == CHATBOT_LOCKED_USERNAME))
    user = user_result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_details = await get_resume_profile(session, CHATBOT_LOCKED_USERNAME)
    return serialize_user_details(user, user_details)


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


@router.post("/chat/stream")
async def chat_with_user_stream(payload: ChatMessageRequest, request: Request, session: AsyncSession = Depends(get_session)):
    """
    SSE endpoint: streams the assistant's RAG-grounded answer token-by-token
    as it's generated. Always answers about CHATBOT_LOCKED_USERNAME --
    payload.target_username is accepted but ignored, since this deployment
    is scoped to a single candidate.
    """
    caller = get_authenticated_username(request)
    conversation = _get_or_create_conversation(caller, payload.conversation_id)

    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    history = list(conversation["messages"])
    conversation["messages"].append({"role": "user", "content": user_message})

    _, details = await _load_user_and_profile(session, CHATBOT_LOCKED_USERNAME)
    resume_chunks, jobs = _retrieve_and_rerank(user_message, CHATBOT_LOCKED_USERNAME)
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
