from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.models import AnalyzeResponse, ChatRequest, ChatResponse
from api.state import app_state
from main import load_patient_docx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML components once on startup."""
    app_state.load()
    yield


app = FastAPI(
    title="team_bereza API",
    description="RAG-пайплайн для генерации клинических рекомендаций",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="DOCX-файл карты пациента"),
    mode: str = Form("doctor", description="Режим: 'doctor' или 'patient'"),
) -> AnalyzeResponse:
    """
    Принять DOCX пациента, запустить RAG-пайплайн, вернуть ответ и session_id.

    session_id используется для последующих вопросов через /api/chat.
    """
    if mode not in ("doctor", "patient"):
        raise HTTPException(status_code=400, detail="mode must be 'doctor' or 'patient'")

    # Сохраняем загруженный файл во временный файл
    suffix = Path(file.filename or "patient.docx").suffix or ".docx"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        patient_text = load_patient_docx(tmp_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Не удалось прочитать файл: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if not patient_text.strip():
        raise HTTPException(status_code=422, detail="Документ пациента пустой.")

    # Эмбеддинг + поиск в FAISS
    cfg = app_state.cfg
    patient_emb = app_state.embedder.embed_text(patient_text)
    results = app_state.store.search(patient_emb[0], top_k=cfg.top_k)
    retrieved_sections = [r["text"] for r in results]

    # Генерация
    response = app_state.generator.generate(
        patient_text=patient_text,
        retrieved_sections=retrieved_sections,
        mode=mode,
        max_new_tokens=cfg.max_new_tokens,
    )

    session_id = app_state.create_session(patient_text, retrieved_sections, mode)
    logger.info("Analyze complete. session_id=%s mode=%s", session_id, mode)
    return AnalyzeResponse(session_id=session_id, response=response)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """
    Ответить на вопрос пользователя в контексте уже выполненного анализа.

    Требует session_id, полученный из /api/analyze.
    """
    session = app_state.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")

    reply = app_state.generator.answer(
        question=body.message,
        patient_text=session.patient_text,
        retrieved_sections=session.retrieved_sections,
        mode=session.mode,
    )

    session.history.append({"role": "user", "content": body.message})
    session.history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply)


# Раздача статики website/ — должна быть последней, чтобы не перекрыть API
_website_dir = Path(__file__).parent.parent / "website"
if _website_dir.exists():
    app.mount("/", StaticFiles(directory=str(_website_dir), html=True), name="static")
