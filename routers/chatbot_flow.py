"""
Chatbot-facing API surface: no-auth-required routes for looking up a
registered candidate's technical profile and chatting about it. The RAG
pipeline itself is a LangGraph graph -- see helpers/chatbot_rag.py. All
routes here are exempt from auth -- see middleware.auth.PUBLIC_PREFIXES
("/chatbot", "/chat").
"""
import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_session
from db.models import ChatLead, ChatMessage, User, UserDetails
from helpers.auth_context import get_caller_id, require_username
from helpers.chatbot_rag import MAX_MESSAGE_CHARS, cached, get_conversation, list_conversations, stream_chat_events
from services.resume_service import get_resume_profile

router = APIRouter()

# This chatbot deployment is scoped to a single candidate -- every chat
# answers about this user's profile/resume regardless of who's asking, so
# it's safe to share as a personal link.
CHATBOT_LOCKED_USERNAME = "gaurav"


class ChatMessageRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = Field(default=None, max_length=64)


def serialize_user_details(user: User, user_details: UserDetails | None = None) -> dict[str, Any]:
    return {
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "age": user.age,
        "graduated": user.graduated,
        "linkedin_url": user.linkedin_url,
        "leetcode_url": user.leetcode_url,
        "hackerrank_url": user.hackerrank_url,
        "resume_skills": (user_details.skills or []) if user_details else [],
        "experience": user_details.experience if user_details else None,
        "industry_type": user_details.industry_type if user_details else None,
        "resume_text_available": bool(user_details and user_details.resume_text and user_details.resume_text.strip()),
        "resume_text": user_details.resume_text if user_details else None,
    }


async def _load_profile(session: AsyncSession, username: str) -> dict[str, Any]:
    async def load() -> dict[str, Any]:
        user = await session.get(User, username)
        if not user:
            raise HTTPException(status_code=404, detail=f"No registered user named '{username}'")
        return serialize_user_details(user, await get_resume_profile(session, username))

    return await cached(username, "profile", load)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/chatbot/user-details")
async def get_user_details_for_chatbot(session: AsyncSession = Depends(get_session)):
    """Dedicated route for the (single-candidate) chatbot. Returns only CHATBOT_LOCKED_USERNAME's profile + resume summary."""
    return await _load_profile(session, CHATBOT_LOCKED_USERNAME)


@router.get("/chat/conversations")
async def list_chat_conversations(request: Request):
    return list_conversations(get_caller_id(request))


@router.get("/chat/conversations/{conversation_id}")
async def get_chat_conversation(request: Request, conversation_id: str):
    conversation = await get_conversation(get_caller_id(request), conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/chat/stream")
async def chat_with_user_stream(payload: ChatMessageRequest, request: Request, session: AsyncSession = Depends(get_session)):
    """
    SSE endpoint: streams the assistant's RAG-grounded answer token-by-token
    as it's generated. Always answers about CHATBOT_LOCKED_USERNAME.

    Events: `meta` {conversation_id, sources}, `token` {delta}..., `done` {message, sources}.
    """
    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    caller = get_caller_id(request)
    conversation_id = payload.conversation_id or str(int(time.time() * 1000))
    # Loaded before streaming starts so a missing user is still a clean 404.
    profile = await _load_profile(session, CHATBOT_LOCKED_USERNAME)

    async def event_generator():
        async for event in stream_chat_events(caller, conversation_id, user_message, CHATBOT_LOCKED_USERNAME, profile):
            if "meta" in event:
                yield _sse("meta", {"conversation_id": conversation_id, "sources": event["meta"]})
            elif "token" in event:
                yield _sse("token", {"delta": event["token"]})
            elif "done" in event:
                yield _sse("done", event["done"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/owner/chat-inbox")
async def chat_inbox(request: Request, limit: int = 20, session: AsyncSession = Depends(get_session)):
    """
    Private to the candidate (requires CHATBOT_LOCKED_USERNAME's JWT): contact
    details visitors left in the chat, plus recent conversations with every message.
    """
    if require_username(request) != CHATBOT_LOCKED_USERNAME:
        raise HTTPException(status_code=403, detail="Only the candidate can read the chat inbox")
    limit = max(1, min(limit, 100))

    leads = (await session.execute(select(ChatLead).order_by(ChatLead.created_at.desc()).limit(50))).scalars().all()

    recent = (
        await session.execute(
            select(ChatMessage.conversation_id, func.max(ChatMessage.created_at).label("last_at"))
            .group_by(ChatMessage.conversation_id)
            .order_by(func.max(ChatMessage.created_at).desc())
            .limit(limit)
        )
    ).all()
    conversation_ids = [row.conversation_id for row in recent]
    messages = (
        await session.execute(
            select(ChatMessage).where(ChatMessage.conversation_id.in_(conversation_ids)).order_by(ChatMessage.created_at, ChatMessage.id)
        )
    ).scalars().all()

    by_conversation: dict[str, list[dict[str, Any]]] = {cid: [] for cid in conversation_ids}
    for m in messages:
        by_conversation[m.conversation_id].append({"role": m.role, "content": m.content, "at": m.created_at})

    return {
        "leads": [
            {"email": l.email, "phone": l.phone, "message": l.message, "conversation_id": l.conversation_id, "at": l.created_at}
            for l in leads
        ],
        "conversations": [
            {"conversation_id": row.conversation_id, "last_at": row.last_at, "messages": by_conversation[row.conversation_id]}
            for row in recent
        ],
    }
