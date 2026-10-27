"""
Chatbot-facing API surface: public routes for chatting with a candidate's
CV ("talk to my CV"). Every registered user has their own bot at
/chatbot?u=<username>, open to anyone with the link; plain /chatbot uses
CHATBOT_DEFAULT_USERNAME. The RAG pipeline is a LangGraph graph -- see
helpers/chatbot_rag.py. /chatbot* and /chat* are exempt from auth (see
middleware.auth.PUBLIC_PREFIXES); /owner/chat-inbox is not.
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
from config import CHATBOT_DEFAULT_USERNAME
from helpers.auth_context import get_caller_id, require_username
from helpers.chatbot_rag import MAX_MESSAGE_CHARS, cached, get_conversation, list_conversations, stream_chat_events
from services.resume_service import get_resume_profile

router = APIRouter()

class ChatMessageRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = Field(default=None, max_length=64)
    # Whose CV to chat with; defaults to CHATBOT_DEFAULT_USERNAME.
    target_username: str | None = Field(default=None, max_length=150)


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


async def _candidate_profile(session: AsyncSession, requested: str | None) -> dict[str, Any]:
    """The profile of the candidate whose bot is being used (404 if no such user)."""
    username = (requested or CHATBOT_DEFAULT_USERNAME).strip().lower()
    try:
        return await _load_profile(session, username)
    except HTTPException:
        raise HTTPException(status_code=404, detail="This chatbot isn't available.") from None


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/chatbot/user-details")
async def get_user_details_for_chatbot(u: str | None = None, session: AsyncSession = Depends(get_session)):
    """Public header info for /chatbot?u=<username>. Deliberately minimal: the CV itself is only reachable through chat answers."""
    profile = await _candidate_profile(session, u)
    return {
        "username": profile["username"],
        "full_name": profile["full_name"],
        "has_cv": profile["resume_text_available"],
    }


@router.get("/chat/conversations")
async def list_chat_conversations(request: Request, u: str | None = None):
    return list_conversations(get_caller_id(request), (u or CHATBOT_DEFAULT_USERNAME).strip().lower())


@router.get("/chat/conversations/{conversation_id}")
async def get_chat_conversation(request: Request, conversation_id: str, u: str | None = None):
    candidate = (u or CHATBOT_DEFAULT_USERNAME).strip().lower()
    conversation = await get_conversation(get_caller_id(request), candidate, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/chat/stream")
async def chat_with_user_stream(payload: ChatMessageRequest, request: Request, session: AsyncSession = Depends(get_session)):
    """
    SSE endpoint: streams the assistant's RAG-grounded answer token-by-token
    as it's generated, about the CV of `target_username` (default candidate if omitted).

    Events: `meta` {conversation_id, sources}, `token` {delta}..., `done` {message, sources}.
    """
    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    caller = get_caller_id(request)
    conversation_id = payload.conversation_id or str(int(time.time() * 1000))
    # Loaded before streaming starts so an unavailable chatbot is still a clean 404.
    profile = await _candidate_profile(session, payload.target_username)
    candidate = profile["username"]

    async def event_generator():
        async for event in stream_chat_events(caller, conversation_id, user_message, candidate, profile):
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
    The signed-in user's own chatbot inbox: contact details visitors left in
    their chat, plus recent conversations with every message.
    """
    me = require_username(request)
    limit = max(1, min(limit, 100))
    # Rows logged before chats were tagged by candidate belong to the default candidate.
    mine_lead = ChatLead.candidate == me
    mine_msg = ChatMessage.candidate == me
    if me == CHATBOT_DEFAULT_USERNAME:
        mine_lead = mine_lead | ChatLead.candidate.is_(None)
        mine_msg = mine_msg | ChatMessage.candidate.is_(None)

    leads = (await session.execute(select(ChatLead).where(mine_lead).order_by(ChatLead.created_at.desc()).limit(50))).scalars().all()

    recent = (
        await session.execute(
            select(ChatMessage.conversation_id, func.max(ChatMessage.created_at).label("last_at"))
            .where(mine_msg)
            .group_by(ChatMessage.conversation_id)
            .order_by(func.max(ChatMessage.created_at).desc())
            .limit(limit)
        )
    ).all()
    conversation_ids = [row.conversation_id for row in recent]
    messages = (
        await session.execute(
            select(ChatMessage).where(mine_msg, ChatMessage.conversation_id.in_(conversation_ids)).order_by(ChatMessage.created_at, ChatMessage.id)
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
