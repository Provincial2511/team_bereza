from __future__ import annotations

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    session_id: str
    response: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
