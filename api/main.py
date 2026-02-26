from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.models import AnalyzeResponse, ChatRequest, ChatResponse
from api.state import app_state
from main import load_patient_docx
from src.ocr import extract_text_from_pdf

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


# Keywords that typically appear in diagnosis / treatment lines of Russian
# clinical documents.  Used by _build_retrieval_query to extract a short,
# semantically focused query string for FAISS retrieval.
_DIAGNOSIS_KEYWORDS: tuple[str, ...] = (
    "диагноз", "diagnos",
    "рак", "лимфома", "саркома", "меланома", "опухоль", "карцинома",
    "аденокарцинома", "глиобластома", "гепатоцеллюлярн",
    "стадия", "pT", "pN", "pM", "T1", "T2", "T3", "T4",
    "N0", "N1", "N2", "N3", "M0", "M1",
    "EGFR", "KRAS", "HER2", "BRCA", "BRAF", "ALK", "PD-L1", "MSI", "ROS1",
    "химиотерапия", "таргетн", "иммунотерапия",
    "операция", "резекция", "мастэктомия", "нефрэктомия",
    "яичник", "молочной железы", "легкого", "простат", "матк", "толстой",
    "прямой кишки", "пищевода", "желудк", "поджелудочной",
)


def _build_retrieval_query(patient_text: str, max_chars: int = 400) -> str:
    """
    Extract a short, diagnosis-focused query string for FAISS retrieval.

    ``sentence-transformers/all-MiniLM-L6-v2`` is optimised for short texts
    (~256 tokens ≈ 800–1200 Russian chars).  Embedding the full patient
    record makes the query vector dominated by the administrative header
    (patient name, admission date, hospital name), not by the actual
    diagnosis — this hurts retrieval recall significantly.

    Strategy: scan lines for diagnosis/treatment keywords; join the first
    matching lines up to *max_chars*.  Fallback: first *max_chars* chars of
    patient_text if no keyword lines are found.
    """
    lines = [ln.strip() for ln in patient_text.splitlines() if ln.strip()]
    selected: list[str] = []
    total = 0
    for line in lines:
        lower = line.lower()
        if any(kw.lower() in lower for kw in _DIAGNOSIS_KEYWORDS):
            if total + len(line) + 1 > max_chars:
                break
            selected.append(line)
            total += len(line) + 1
        if total >= max_chars:
            break
    if selected:
        return " ".join(selected)
    # Fallback: use the beginning of the document.
    return patient_text[:max_chars].strip()


def _format_chunk(result: dict) -> str:
    """
    Prepend a human-readable document label to a retrieved chunk.

    Label priority:
    1. ``metadata["title"]`` — extracted from PDF metadata at index-build time.
    2. Cleaned ``metadata["source"]`` filename (underscores → spaces, no .pdf).
    3. «неизвестно» as a last resort.

    Format: ``[КР: <label>]\\n<chunk text>``
    """
    meta = result.get("metadata", {})
    label = (meta.get("title") or "").strip()
    if not label:
        raw_source = meta.get("source", "")
        label = raw_source.removesuffix(".pdf").replace("_", " ").strip()
    header = f"[КР: {label}]" if label else "[КР: неизвестно]"
    return f"{header}\n{result['text']}"


def _extract_patient_summary(patient_text: str, max_chars: int = 600) -> str:
    """
    Return the first meaningful block of patient text (up to max_chars).

    Takes up to 3 non-empty paragraphs from the beginning of the document —
    this is where patient demographics and diagnosis are typically located.
    No LLM required, always reliable.
    """
    paragraphs = [p.strip() for p in patient_text.split("\n\n") if p.strip()]
    summary_parts: list[str] = []
    total = 0
    for p in paragraphs[:6]:
        if total + len(p) > max_chars:
            break
        summary_parts.append(p)
        total += len(p)
    return "\n\n".join(summary_parts) if summary_parts else patient_text[:max_chars].strip()


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
    suffix = Path(file.filename or "patient.docx").suffix.lower() or ".docx"
    if suffix not in (".docx", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат '{suffix}'. Загрузите DOCX или PDF.",
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            patient_text = extract_text_from_pdf(
                tmp_path, languages=app_state.cfg.ocr_languages
            )
        else:
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

    # Build a short, diagnosis-focused query instead of embedding the full
    # patient record.  all-MiniLM-L6-v2 truncates inputs at ~256 tokens; the
    # full document would be represented mostly by the administrative header.
    retrieval_query = _build_retrieval_query(patient_text)
    logger.info(
        "Retrieval query (%d chars): %.150s", len(retrieval_query), retrieval_query
    )

    patient_emb = app_state.embedder.embed_text(retrieval_query)
    results = app_state.store.search(patient_emb[0], top_k=cfg.top_k)

    # Log retrieved chunks with L2 distances for debugging.
    for i, r in enumerate(results):
        logger.info(
            "Retrieved chunk %d: score=%.4f  text=%.80s",
            i,
            r["score"],
            r["text"].replace("\n", " "),
        )

    # Filter out chunks whose L2 distance exceeds the threshold (cosine sim too low).
    # If fewer than 2 chunks survive, fall back to all top_k results to avoid
    # passing empty context to the LLM.
    threshold = cfg.retrieval_score_threshold
    filtered = [r for r in results if r["score"] <= threshold]
    if len(filtered) < 2:
        logger.warning(
            "Only %d/%d chunks below threshold %.2f; using all retrieved chunks.",
            len(filtered),
            len(results),
            threshold,
        )
        filtered = results
    else:
        logger.info(
            "Threshold filter (%.2f): kept %d/%d chunks.",
            threshold,
            len(filtered),
            len(results),
        )

    retrieved_sections = [_format_chunk(r) for r in filtered]

    # Генерация (синхронные вызовы — блокируют event loop на время инференса,
    # но это безопасно: один запрос за раз, модель не thread-safe)
    logger.info("Starting main generation (device=%s, max_new_tokens=%d)...",
                cfg.device, cfg.max_new_tokens)
    response = app_state.generator.generate(
        patient_text=patient_text,
        retrieved_sections=retrieved_sections,
        mode=mode,
        max_new_tokens=cfg.max_new_tokens,
    )
    logger.info("Main generation done (%d chars). Starting structured extraction...",
                len(response))

    structured = app_state.generator.generate_structured(
        patient_text=patient_text,
        main_analysis=response,
    )
    if structured is None:
        logger.warning("generate_structured returned None; structured sections will be empty")
    else:
        logger.info("Structured extraction done.")

    # Краткая выжимка из карты пациента — без LLM, всегда надёжно.
    patient_summary = _extract_patient_summary(patient_text)

    session_id = app_state.create_session(patient_text, retrieved_sections, mode)
    logger.info("Analyze complete. session_id=%s mode=%s", session_id, mode)
    return AnalyzeResponse(
        session_id=session_id,
        response=response,
        patient_summary=patient_summary,
        structured=structured,
    )


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


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the landing page."""
    return RedirectResponse(url="/index2.html")


# Раздача статики website/ — должна быть последней, чтобы не перекрыть API
_website_dir = Path(__file__).parent.parent / "website"
if _website_dir.exists():
    app.mount("/", StaticFiles(directory=str(_website_dir), html=True), name="static")
