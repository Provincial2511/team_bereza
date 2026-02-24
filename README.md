# team_bereza

Локальный RAG-пайплайн для генерации клинических рекомендаций по онкологии.
Принимает на вход PDF клинических руководств и DOCX-карту пациента, возвращает структурированный ответ в двух режимах: для врача и для пациента.

---

## Архитектура

```
Браузер
    → upload.html  (загрузить DOCX пациента + выбрать режим)
    → POST /api/analyze
              │
              ▼
         FastAPI [api/main.py]
              │  AppState: embedder + FAISS + Qwen загружены ОДИН РАЗ
              ├─ embed patient text → FAISS search → top-5 чанков
              └─ LocalGenerator.generate() → ответ
              │
              ▼
    ← session_id + response → results.html
    → POST /api/chat  (follow-up вопросы через чат-панель)

RAG-пайплайн (первый запуск / пересборка индекса):
PDF (клинические рекомендации)
    → OCR / native extract   [src/ocr.py]
    → Semantic chunker        [src/parser.py]
    → TextEmbedder            [src/embeddings.py]
    → FaissStore (кэш)        [src/faiss_store.py]
```

Пайплайн полностью локальный — никаких внешних API.

---

## Установка

```bash
pip install -r requirements.txt
```

> Torch устанавливается с CUDA 12.1. Для CPU-only замени строку в `requirements.txt` на `torch==2.5.0`.

---

## Запуск

### Веб-приложение (основной способ)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Открыть `http://localhost:8000` — сайт и API поднимаются вместе.
При первом старте модели и FAISS-индекс загружаются один раз; все последующие запросы используют уже загруженные объекты.

### Только пайплайн (без сайта)

```bash
python main.py
```

Все параметры задаются в `src/config.py`:

```python
from src.config import Config

cfg = Config(
    mode="doctor",          # "doctor" | "patient"
    device="cuda",          # "cpu" | "cuda"
    top_k=5,
    max_new_tokens=1024,
)
```

При первом запуске пайплайн выполняет OCR, чанкинг и строит FAISS-индекс (`data/faiss_index/`). При последующих запусках индекс загружается с диска.

Чтобы пересобрать индекс (после добавления нового PDF):

```bash
rm -rf data/faiss_index/
python main.py
```

---

## Тестирование

Jupyter-ноутбук `tests.ipynb` в корне проекта содержит изолированные тесты каждого модуля и полный end-to-end прогон.

```bash
jupyter notebook tests.ipynb
```

---

## Структура проекта

```
team_bereza/
├── main.py                         # CLI-запуск пайплайна без веба
├── tests.ipynb                     # тесты всех модулей
├── requirements.txt
├── api/
│   ├── main.py                     # FastAPI-приложение, эндпоинты + раздача website/
│   ├── state.py                    # AppState: синглтон с моделью и индексом
│   └── models.py                   # Pydantic-схемы запросов / ответов
├── src/
│   ├── config.py                   # централизованная конфигурация
│   ├── ocr.py                      # извлечение текста из PDF
│   ├── parser.py                   # семантический чанкер
│   ├── embeddings.py               # TextEmbedder (sentence-transformers)
│   ├── faiss_store.py              # векторное хранилище
│   └── generator.py                # LocalGenerator (Qwen2-7B-Instruct)
├── website/
│   ├── index2.html                 # лендинг: выбор режима doctor / patient
│   ├── upload.html                 # загрузка DOCX пациента
│   ├── results.html                # результаты анализа + чат-панель
│   └── js/app.js                   # общая логика, API-клиент
└── data/
    ├── clinical_guideline/         # PDF клинических рекомендаций
    ├── input_example/              # DOCX пациента
    ├── faiss_index/                # кэш индекса (генерируется автоматически)
    └── generator_response/         # версионированные JSON-ответы
```

---

## Дизайн решения

### Компоненты

| Компонент | Класс / функция | Модель / библиотека |
|-----------|----------------|---------------------|
| OCR / извлечение текста | `extract_text_from_pdf()` | PyMuPDF + EasyOCR |
| Чанкер | `GuidelineParserStub` | tokenizer all-MiniLM-L6-v2 |
| Эмбеддер | `TextEmbedder` | sentence-transformers/all-MiniLM-L6-v2 |
| Векторное хранилище | `FaissStore` | faiss.IndexFlatL2 |
| Генератор | `LocalGenerator` | Qwen2-7B-Instruct |

### Полный поток данных

```
ВХОДНЫЕ ДАННЫЕ
├── PDF (клинические рекомендации)
└── DOCX (карта пациента)
         │
         ▼
┌─────────────────────────────────────────────┐
│  ВЕТКА А: FAISS-индекс                      │
│                                             │
│  Проверка: index.faiss + texts.pkl          │
│           + metadata.pkl существуют?        │
│                                             │
│  ДА ──────────────────► загрузить индекс    │
│                          с диска            │
│                                             │
│  НЕТ                                        │
│   │                                         │
│   ▼                                         │
│  PDF → extract_text_from_pdf()              │
│   │                                         │
│   │  Детекция типа PDF:                     │
│   ├─ есть текстовый слой → PyMuPDF          │
│   └─ только картинки   → EasyOCR            │
│             │                               │
│             ▼                               │
│        сырой текст (с \n\n абзацами)        │
│             │                               │
│             ▼                               │
│  GuidelineParserStub.parse()                │
│   1. split по \n\n → абзацы                 │
│   2. группировка мелких абзацев             │
│      до chunk_size=500 токенов              │
│   3. крупные абзацы → токен-окна            │
│      с overlap=50                           │
│             │                               │
│             ▼                               │
│        list[str] — чанки                   │
│             │                               │
│             ▼                               │
│  TextEmbedder.embed_batch()                 │
│  → ndarray (N, 384), L2-нормализованные     │
│             │                               │
│             ▼                               │
│  FaissStore.add() → сохранить на диск       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ВЕТКА Б: ПАЦИЕНТ                           │
│                                             │
│  DOCX → load_patient_docx()                 │
│       → str (текст карты пациента)          │
│             │                               │
│             ▼                               │
│  TextEmbedder.embed_text()                  │
│  → ndarray (1, 384), L2-нормализованный     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  RETRIEVAL                                  │
│                                             │
│  FaissStore.search(query, top_k=5)          │
│  → 5 ближайших чанков (L2 = cosine dist)   │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ГЕНЕРАЦИЯ                                  │
│                                             │
│  LocalGenerator.generate()                 │
│                                             │
│  apply_chat_template:                       │
│  ┌─ system: промпт режима                  │
│  └─ user:  [чанки] + [карта пациента]      │
│             │                               │
│             ▼                               │
│  Qwen2-7B-Instruct                          │
│  do_sample=False, max_new_tokens=1024       │
│             │                               │
│             ▼                               │
│  decode(new_tokens_only) → str             │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ВЫХОД                                      │
│                                             │
│  version_N.json:                            │
│  ├── final_response                         │
│  ├── mode (doctor / patient)                │
│  ├── model_name                             │
│  ├── stage_durations                        │
│  └── timestamp_start / end                 │
└─────────────────────────────────────────────┘
```

### Ключевые развилки

**1. При старте пайплайна:**
- `index.faiss` существует? → **да** → загрузить → пропустить OCR / чанкинг / эмбеддинг гайдлайнов
- → **нет** → OCR → чанкинг → эмбеддинг → сохранить индекс

**2. Внутри OCR:**
- PDF имеет текстовый слой? → **да** → PyMuPDF (мгновенно)
- → **нет** → EasyOCR (рендер страниц в 300 DPI → numpy → OCR)

**3. Внутри чанкера (для каждого абзаца):**
- абзац > 500 токенов? → **да** → токен-окна с overlap
- накопленный буфер + абзац > 500 токенов? → **да** → flush буфера, начать новый
- → **нет** → добавить абзац в буфер

**4. Флаг `add_new_guidelines`:**
- `True` и индекс существует и новый PDF существует → допиндексировать новый PDF и сохранить обновлённый индекс

### Что не входит в основную схему

- Валидации пустого текста и размерностей — guard-клозы внутри модулей, не ветки бизнес-логики
- Версионирование JSON — линейная запись после генерации, отдельной ветки нет
- `GuidelineParser` (LLM-парсер) — реализован в `parser.py`, но в пайплайн не включён

---

## История версий

| Версия | Дата       | Изменения |
|--------|------------|-----------|
| v0.1   | 2026-02-20 | Минимальная работоспособная версия RAG |
| v0.2   | 2026-02-21 | Добавлена логика режимов ответов: doctor / patient |
| v0.3   | 2026-02-22 | Рефакторинг и улучшение всех модулей (см. ниже) |
| v0.4   | 2026-02-24 | FastAPI-бэкенд + интеграция сайта с инференсом (см. ниже) |

---

## Изменения v0.3

### `src/ocr.py` — замена Tesseract на EasyOCR + PyMuPDF

**Было:** `pytesseract` + `pdf2image` с захардкоженным путём к `tesseract.exe`.
**Стало:** два бэкенда в одной функции `extract_text_from_pdf()`:

- **Native PDF** (есть текстовый слой) → `PyMuPDF` извлекает текст напрямую, мгновенно.
- **Scanned PDF** (нет текстового слоя) → страницы рендерятся через `PyMuPDF` в 300-DPI numpy-массивы и передаются в `EasyOCR`.

EasyOCR Reader кэшируется в памяти (`_READER_CACHE`) — при повторных вызовах веса не перезагружаются.
Нормализация теперь **сохраняет структуру абзацев** (`\n\n`), а не схлопывает весь текст в одну строку.
Удалены зависимости: `pytesseract`, `pdf2image`, системный Tesseract.

---

### `src/parser.py` — семантический чанкер

**Было:** `GuidelineParserStub` — простой токен-слайсинг с фиксированным шагом, без учёта структуры текста.
**Стало:** трёхуровневый алгоритм в `GuidelineParserStub.parse()`:

1. Разбивка по `\n\n` (абзацы, сохранённые новым OCR).
2. Накопление мелких абзацев в один чанк, пока не превышен `chunk_size`.
3. Для одиночных абзацев крупнее `chunk_size` — токен-окна с `overlap` (прежнее поведение).

Результат: чанки семантически связны, не режут предложения посередине там, где это не нужно.

---

### `src/generator.py` — chat template

**Было:** системный промпт и данные конкатенировались в сырую f-строку, затем промпт вручную срезался из ответа (`startswith` + слайс).
**Стало:**

- Системные промпты вынесены в константу класса `_SYSTEM_PROMPTS` — один dict на оба режима.
- `_get_system_prompt(mode)` выбрасывает явный `ValueError` при неизвестном режиме.
- Форматирование через `tokenizer.apply_chat_template()` — правильный формат для Qwen2-7B-Instruct (и любой другой chat-модели с шаблоном).
- Декодируются **только новые токены** (`output_ids[0][input_length:]`) — хак со срезанием промпта из ответа удалён.

---

### `src/config.py` — новый файл

Датакласс `Config` с дефолтами для всех параметров пайплайна: пути к файлам, имена моделей, языки OCR, `chunk_size`, `overlap`, `top_k`, `max_new_tokens`, `device`, `mode`.
`main.py` теперь содержит только `cfg = Config()` — никаких разбросанных констант.

---

### `tests.ipynb` — новый файл

Jupyter-ноутбук с изолированными тестами каждого модуля:
- автоопределение CUDA и `DEVICE`
- проверка всех зависимостей
- OCR с замером времени и статистикой по абзацам
- parser: юнит-тест на синтетическом тексте + реальный PDF
- embedder: проверка L2-нормализации, матрица косинусного сходства
- FAISS: add / search / save / load
- generator: проверка chat template, генерация в обоих режимах
- полный пайплайн через `main()` + просмотр `version_N.json`

---

### `requirements.txt`

| Было | Стало |
|------|-------|
| `pytesseract` | удалено |
| `pdf2image` | удалено |
| `Pillow==11.0.0` | `Pillow` (без пина) |
| — | `easyocr` |
| — | `PyMuPDF` |

---

## Изменения v0.4

### `api/` — новая папка: FastAPI-приложение

**Было:** сайт был статическим макетом; инференс запускался только через `python main.py`.
**Стало:** полноценный REST API, объединяющий модель и сайт под одним сервером.

**`api/state.py`** — синглтон `AppState`:
- Загружает `TextEmbedder`, `FaissStore` и `LocalGenerator` один раз при старте сервера (FastAPI lifespan).
- `create_session()` сохраняет `patient_text`, `retrieved_sections` и `mode` для follow-up чата.
- `get_session()` возвращает данные сессии по UUID.

**`api/models.py`** — Pydantic-схемы:
- `AnalyzeResponse` — `session_id` + `response`.
- `ChatRequest` — `session_id` + `message`.
- `ChatResponse` — `reply`.

**`api/main.py`** — FastAPI-приложение:
- `POST /api/analyze` — принимает DOCX-файл и `mode` (form-data), запускает полный RAG-пайплайн, возвращает `AnalyzeResponse`.
- `POST /api/chat` — принимает `ChatRequest`, вызывает `generator.answer()` с сохранённым контекстом сессии, возвращает `ChatResponse`.
- Папка `website/` раздаётся как статические файлы: `GET /` открывает `index2.html`.

---

### `src/generator.py` — метод `answer()`

Добавлен метод `answer(question, patient_text, retrieved_sections, mode)` для follow-up вопросов в чате:
- Использует те же системный промпт и retrieved-контекст, что и основной `generate()`.
- К user-сообщению добавляется блок `=== Вопрос ===` с вопросом пользователя.
- Гарантирует наличие дисклеймера через `_ensure_disclaimer()`.

---

### `website/js/app.js` — реальные API-вызовы

**Было:** `startVerification()` делала `setTimeout` с захардкоженным ответом, чат открывался через `prompt()`.
**Стало:**
- `startVerification()` отправляет `FormData` на `POST /api/analyze`, сохраняет `session_id` и `response` в `localStorage`, перенаправляет на `results.html`.
- Обработка ошибок: блок `uploadError` показывает сообщение об ошибке без `alert()`.

---

### `website/results.html` — карточка ответа и чат-панель

**Было:** статичная страница-заглушка с кнопкой FAB, открывавшей `prompt()`.
**Стало:**
- **Карточка ответа** — загружает `oncoai_response` из `localStorage` и отображает текст рекомендации.
- **Чат-панель** — выезжает справа по нажатию FAB; поддерживает историю сообщений, поле ввода и кнопку отправки.
- `sendChatMessage()` отправляет `POST /api/chat` с `session_id` из `localStorage`, добавляет ответ модели в историю чата.

---

### `requirements.txt` — новые зависимости v0.4

| — | Добавлено |
|---|-----------|
| — | `fastapi` |
| — | `uvicorn[standard]` |
| — | `python-multipart` |
