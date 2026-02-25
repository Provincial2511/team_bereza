from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    session_id: str
    response: str
    patient_summary: str = ""
    structured: Optional[dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
