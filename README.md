# OncoCheck Assist

Локальный RAG-пайплайн для проверки соответствия лечения онкологических пациентов клиническим рекомендациям. Принимает карту пациента (DOCX или PDF), сопоставляет с базой клинических руководств и возвращает структурированный анализ в двух режимах: для врача и для пациента.

Работает полностью локально — никаких внешних API.

---

## Возможности

- **Два режима ответа** — для врача (клинический язык) и для пациента (доступный язык)
- **Структурированный результат** — пункты лечения разбиты на соответствующие и не соответствующие КР
- **Поддержка PDF и DOCX** — нативное извлечение текста или OCR для сканированных документов
- **Follow-up чат** — задавайте уточняющие вопросы в контексте уже выполненного анализа
- **GPU / CPU** — переключение через переменную окружения

---

## Стек

| Компонент | Решение |
|-----------|---------|
| OCR / извлечение текста | PyMuPDF + EasyOCR |
| Семантический чанкер | tokenizer `all-MiniLM-L6-v2` |
| Эмбеддер | `sentence-transformers/all-MiniLM-L6-v2` |
| Векторное хранилище | FAISS `IndexFlatL2` |
| Генератор | `Qwen/Qwen2-7B-Instruct` |
| API | FastAPI + uvicorn |

---

## Установка

```bash
pip install -r requirements.txt
```

> По умолчанию устанавливается `torch==2.5.0` (CPU). Для CUDA замените на нужную версию в `requirements.txt`.

---

## Запуск

### Первый запуск — построение FAISS-индекса

Перед запуском веб-приложения нужно один раз построить индекс из PDF клинических рекомендаций:

```bash
python main.py
```

Индекс сохранится в `data/faiss_index/` и при последующих запусках загружается с диска.

### Веб-приложение

**CPU:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**GPU (CUDA):**
```bash
# Linux / Git Bash
DEVICE=cuda uvicorn api.main:app --host 0.0.0.0 --port 8000

# Windows CMD
set DEVICE=cuda && uvicorn api.main:app --host 0.0.0.0 --port 8000

# PowerShell
$env:DEVICE="cuda"; uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Открыть в браузере: `http://localhost:8000`

Модели и индекс загружаются один раз при старте — все последующие запросы используют уже загруженные объекты.

### Пересборка индекса (после добавления нового PDF)

```bash
rm -rf data/faiss_index/
python main.py
```

---

## Конфигурация

Все параметры задаются в `src/config.py`:

```python
@dataclass
class Config:
    device: str = "cpu"          # "cpu" | "cuda"
    mode: str = "doctor"         # "doctor" | "patient"
    top_k: int = 5               # кол-во чанков для retrieval
    max_new_tokens: int = 512    # лимит генерации
    chunk_size: int = 500        # размер чанка в токенах
    overlap: int = 50            # перекрытие чанков
```

---

## Структура проекта

```
team_bereza/
├── main.py                     # CLI: построение индекса и запуск пайплайна
├── tests.ipynb                 # изолированные тесты каждого модуля
├── requirements.txt
│
├── api/
│   ├── main.py                 # FastAPI: эндпоинты + раздача website/
│   ├── state.py                # AppState: синглтон с моделью и индексом
│   └── models.py               # Pydantic-схемы запросов / ответов
│
├── src/
│   ├── config.py               # централизованная конфигурация
│   ├── ocr.py                  # извлечение текста из PDF (PyMuPDF + EasyOCR)
│   ├── parser.py               # семантический чанкер
│   ├── embeddings.py           # TextEmbedder
│   ├── faiss_store.py          # векторное хранилище
│   └── generator.py            # LocalGenerator (Qwen2-7B-Instruct)
│
├── website/
│   ├── index2.html             # лендинг: выбор режима
│   ├── upload.html             # загрузка карты пациента
│   ├── results.html            # результаты + чат-панель
│   └── js/app.js               # API-клиент
│
└── data/
    ├── clinical_guideline/     # PDF клинических рекомендаций
    ├── input_example/          # примеры карт пациентов
    ├── faiss_index/            # кэш индекса (генерируется автоматически)
    └── generator_response/     # версионированные JSON-ответы
```

---

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/analyze` | Загрузить DOCX/PDF пациента, получить анализ и `session_id` |
| `POST` | `/api/chat` | Follow-up вопрос в контексте сессии |

**`POST /api/analyze`** — form-data:
- `file` — карта пациента (`.docx` или `.pdf`)
- `mode` — `doctor` или `patient`

Ответ: `{ session_id, response, structured }`

**`POST /api/chat`** — JSON:
```json
{ "session_id": "...", "message": "..." }
```

---

## Тестирование

```bash
jupyter notebook tests.ipynb
```

Ноутбук содержит изолированные тесты OCR, парсера, эмбеддера, FAISS, генератора и полный end-to-end прогон с оценкой качества ответа.

---

## История версий

| Версия | Дата | Изменения |
|--------|------|-----------|
| v0.1 | 2026-02-20 | Минимальная работоспособная версия RAG |
| v0.2 | 2026-02-21 | Режимы ответов: doctor / patient |
| v0.3 | 2026-02-22 | EasyOCR + PyMuPDF, семантический чанкер, chat template |
| v0.4 | 2026-02-24 | FastAPI-бэкенд, веб-интерфейс, чат-панель |
| v0.5 | 2026-02-25 | Структурированный вывод (compliant / non-compliant), GPU, PDF-загрузка |
