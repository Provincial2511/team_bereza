## Minimal Local RAG Pipeline for Clinical Guidelines

This project implements a **fully local**, minimal, and explicit RAG (Retrieval-Augmented Generation) pipeline
for processing clinical guidelines and patient records. It runs entirely on local resources:

The goal is to extract text from scanned clinical guidelines, chunk them, build a vector index, and generate
patient-specific clinical recommendations using a local LLM. The current production flow uses a **stub parser**
for chunking rather than a full JSON-structuring parser.

---

## Modules and Responsibilities

- **`src/ocr.py`**  
  - **Function**: `extract_text_from_scanned_pdf(path: str) -> str`  
  - **Purpose**:  
    - Convert a scanned clinical guideline PDF into plain text using OCR.
  - **Key steps**:
    - Uses `pdf2image.convert_from_path` to turn each PDF page into a PIL image.
    - Uses `pytesseract.image_to_string(..., lang="rus")` to extract Russian text per page.
    - Concatenates page texts with double newlines.
    - Normalizes excessive whitespace into single spaces.
  - **Errors**:
    - Raises `FileNotFoundError` if the PDF path does not exist.
    - Raises `RuntimeError` if PDF-to-image conversion or OCR fails.

- **`src/parser.py`**  
  - **Class**: `GuidelineParser`  
    - **Purpose**:  
      - Use a local HuggingFace causal model (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) to convert raw OCR
        guideline text into a structured set of sections as JSON.
    - **Initialization**:
      - Loads tokenizer and model via `AutoTokenizer` and `AutoModelForCausalLM`.
      - Runs on a configurable device (default `"cpu"`).
    - **Method**: `parse(text: str) -> dict[str, str]`  
      - Builds an explicit instruction prompt (in Russian) describing which guideline sections to extract.
      - Calls the model with deterministic generation:
        - `do_sample=False`
        - `temperature=0.0`
        - `num_beams=1`
      - Decodes model output and parses it as JSON.
      - Returns a `dict[str, str]` mapping section names to their text.
      - Raises `RuntimeError` if the model output is not valid JSON.
    - **Note**: This class is implemented but not used in the current main pipeline (MVP uses the stub below).

  - **Class**: `GuidelineParserStub`  
    - **Purpose**:  
      - Lightweight, deterministic **chunker** used in the current MVP instead of the full JSON parser.
      - Splits raw guideline text into overlapping chunks by token count (no JSON, no section semantics).
    - **Initialization**:
      - Uses `AutoTokenizer` (default model: `sentence-transformers/all-MiniLM-L6-v2`) to tokenize text.
      - Configurable `chunk_size` and `overlap`.
    - **Method**: `parse(text: str) -> list[str]`  
      - Encodes text to tokens, walks over them with stride `chunk_size - overlap`.
      - Decodes each token slice back to text.
      - Returns a list of chunk strings used as guideline sections for embedding and FAISS.

- **`src/embeddings.py`**  
  - **Class**: `TextEmbedder`  
  - **Purpose**:  
    - Generate dense vector embeddings for guideline sections and patient text using a local
      sentence-transformers model (default: `sentence-transformers/all-MiniLM-L6-v2`).
  - **Initialization**:
    - Loads `SentenceTransformer` locally (no external API calls), fixed to `"cpu"` device for determinism.
  - **Methods**:
    - `embed_text(text: str) -> np.ndarray`  
      - Embeds a single string.
      - Output shape: `(1, dimension)` with `dtype=float32`.
      - Performs manual L2 normalization using `numpy` so that vectors have unit length.
      - Raises `ValueError` on empty text.
    - `embed_batch(texts: list[str]) -> np.ndarray`  
      - Embeds a list of texts in order.
      - Output shape: `(n_items, dimension)` with `dtype=float32`.
      - Applies L2 normalization row-wise:
        - This enables cosine similarity to be computed via L2 distance in FAISS.
      - Raises `ValueError` if the list is empty.

- **`src/faiss_store.py`**  
  - **Class**: `FaissStore`  
  - **Purpose**:  
    - Provide a minimal FAISS-backed vector store for retrieval.
    - Uses `faiss.IndexFlatL2` (no advanced indexing, no filters).
  - **Initialization**:
    - `FaissStore(dimension: int)`  
      - Creates an in-memory `IndexFlatL2` for the given embedding dimension.
      - Initializes:
        - `self.texts: list[str]`
        - `self.metadata: list[dict]`
  - **Methods**:
    - `add(texts: list[str], embeddings: np.ndarray, metadata: list[dict]) -> None`  
      - Validates:
        - `embeddings` is a 2D `numpy.ndarray` of shape `(n_items, dimension)`.
        - `len(texts) == n_items` and `len(metadata) == n_items`.
        - Embedding dimension matches the store’s `dimension`.
      - Casts embeddings to `float32` and adds them to the FAISS index.
      - Extends `texts` and `metadata` lists.
    - `search(query_embedding: np.ndarray, top_k: int) -> list[dict]`  
      - Accepts a 1D or 2D query embedding (converts to 2D if needed).
      - Validates dimension and converts to `float32`.
      - Runs FAISS search to get nearest neighbors.
      - Returns a list of:
        - `{"text": str, "metadata": dict, "score": float}`  
        - `score` is the L2 distance (with normalized vectors this corresponds to cosine distance).
      - Raises `RuntimeError` if the index is empty.
    - `save(path: str) -> None` / `load(path: str) -> None`  
      - Persist and restore:
        - FAISS index (`index.faiss`)
        - Texts (`texts.pkl`)
        - Metadata (`metadata.pkl`)
      - Used by the current pipeline to **avoid recomputing embeddings and OCR** on subsequent runs.

- **`src/generator.py`**  
  - **Class**: `LocalGenerator`  
  - **Purpose**:  
    - Use a local causal LLM to generate final clinical recommendations from retrieved guideline sections
      and patient data.
  - **Initialization**:
    - `LocalGenerator(model_name: str, device: str = "cpu")`  
      - Loads tokenizer and model via `AutoTokenizer` and `AutoModelForCausalLM`.
      - Moves the model to the given device (currently forced to `"cpu"` in `main.py`).
      - Ensures a padding token is set (falls back to `eos_token` if needed).
  - **Method**:  
    - `generate(patient_text: str, retrieved_sections: list[str], max_new_tokens: int = 512) -> str`  
      - Builds an explicit prompt (in Russian) combining:
        - Retrieved guideline sections.
        - Patient information.
        - Clear instructions for how to produce recommendations.
      - Tokenizes the prompt and runs deterministic generation (`do_sample=False`).
      - Decodes the output and strips the prompt from the front if present.
      - Falls back to decoding only the new tokens if needed.
      - Returns the final recommendation text.

- **`main.py`**  
  - **Function**: `main() -> None`  
  - **Purpose**:  
    - Compose all modules from `src/` into a **linear, minimal** RAG pipeline that goes from raw files
      to a printed (and saved) clinical recommendation.
  - **Helper**:
    - `load_patient_docx(path: str) -> str`  
      - Uses `python-docx` to load a DOCX file and concatenate non-empty paragraphs.
      - Raises `FileNotFoundError` if the file is missing.
  - **Timing and metadata**:
    - Tracks pipeline timing for:
      - Total runtime
      - OCR (when performed)
      - Chunking
      - Guideline embedding
      - Patient embedding
      - Retrieval
      - Generation
    - Saves all metadata with each generated response as JSON for reproducibility.

---

## End-to-End Data Flow

The pipeline in `main.py` performs the following steps:

1. **Check for existing FAISS index**  
   - Index directory: `data/faiss_index/`  
   - Expected files:
     - `index.faiss`
     - `texts.pkl`
     - `metadata.pkl`
   - If **all exist**:
     - Load index with `FaissStore.load(faiss_index_path)`.
     - Skip OCR, chunking, and guideline embedding.
   - If **any are missing**:
     - Build a new index from the guideline PDF (steps 2–4 below), then save it.

2. **OCR clinical guideline PDF** (only when rebuilding index)  
   - Input path: `data/clinical_guideline/cg_30_5_lung_cancer.pdf`  
   - Call:  
     - `guideline_text = extract_text_from_scanned_pdf(guideline_pdf_path)`  
   - Validation:
     - If the PDF is missing → error printed, pipeline stops.
     - If OCR returns empty text → error printed, pipeline stops.
   - Timing:
     - OCR duration is recorded in `stage_durations["ocr"]`.

3. **Chunk guideline into sections (stub)**  
   - Create parser stub:  
     - `guideline_parser = GuidelineParserStub(chunk_size=500, overlap=50)`  
   - Call:
     - `guideline_sections = guideline_parser.parse(guideline_text)`  
   - Output:
     - List of overlapping text chunks used as “sections” for retrieval.
   - Validation:
     - If `guideline_sections` is empty → error printed, pipeline stops.
   - Timing:
     - Chunking duration is recorded in `stage_durations["chunking"]`.

4. **Embed guideline sections**  
   - Initialize embedder:
     - `embedder = TextEmbedder()`  
   - Call:
     - `section_embeddings = embedder.embed_batch(guideline_sections)`  
   - Each section becomes a normalized vector `(dimension,)`.  
   - Validation:
     - If `section_embeddings.size == 0` → error printed, pipeline stops.
   - Shape:
     - `section_embeddings.shape == (n_items, dimension)`
   - Timing:
     - Embedding duration is recorded in `stage_durations["embedding"]`.

5. **Build and (if needed) persist FAISS index**  
   - Derive dimension from embeddings:
     - `n_items, dimension = section_embeddings.shape`  
   - Initialize:
     - `store = FaissStore(dimension=dimension)`  
   - Prepare metadata:
     - `empty_metadata = [{} for _ in range(n_items)]`  
   - Add to store:
     - `store.add(guideline_sections, section_embeddings, empty_metadata)`  
   - Result:
     - `store.index` holds all guideline vectors.
     - `store.texts` holds the corresponding section texts.
   - Save to disk (first run / rebuild only):
     - `store.save("data/faiss_index")`

6. **Load and prepare patient data**  
   - Input path: `data/input_example/case_example_2.docx`  
   - Call:
     - `patient_text = load_patient_docx(patient_docx_path)`  
   - Validation:
     - If DOCX is missing → error printed, pipeline stops.
     - If resulting text is empty → error printed, pipeline stops.

7. **Embed patient text**  
   - Call:
     - `patient_embedding = embedder.embed_text(patient_text)`  
   - Shape:
     - `(1, dimension)` (normalized vector).
   - Dimension check:
     - If `patient_embedding.shape[1] != store.dimension` → raises `RuntimeError`.
   - Timing:
     - Patient embedding duration is recorded in `stage_durations["patient_embedding"]`.

8. **Retrieve relevant guideline sections**  
   - Set:
     - `top_k = 5`  
   - Use the 1D query vector:
     - `query_vector = patient_embedding[0]`  
   - Call FAISS:
     - `search_results = store.search(query_vector, top_k=top_k)`  
   - Extract texts:
     - `retrieved_sections = [r["text"] for r in search_results]`  
   - Validation:
     - If `retrieved_sections` is empty → error printed, pipeline stops.
   - Timing:
     - Retrieval duration is recorded in `stage_durations["retrieval"]`.

9. **Generate final clinical recommendation**  
   - Device selection:
     - Currently forced to `"cpu"` in `main.py` for stability and reproducibility.  
       (Original design allowed switching to `"cuda"` if available.)
   - Initialize local generator:
     - `generator = LocalGenerator(model_name="Qwen/Qwen2-7B-Instruct", device=device)`  
   - Call:
     - `final_response = generator.generate(patient_text, retrieved_sections, max_new_tokens=2048)`  
   - The prompt includes:
     - Retrieved guideline sections as context.
     - Patient data.
     - An explicit instruction for generating a recommendation.
   - Timing:
     - Generation duration is recorded in `stage_durations["generation"]`.

10. **Save response with versioned JSON and timing metadata**  
    - Output directory:
      - `data/generator_response/`
    - Versioning:
      - Files named `version_X.json` where `X = 0, 1, 2, ...`.
      - On each run, the code:
        - Scans existing `version_*.json` files.
        - Finds the highest `X`.
        - Saves the new run as `version_{X+1}.json` (no overwrites).
    - Each JSON file contains:
      - `version`: Integer version number.
      - `timestamp_start`: Pipeline start time (ISO format).
      - `timestamp_end`: Pipeline end time (ISO format).
      - `total_duration_seconds`: Total runtime in seconds (rounded).
      - `model_name`: Name/path of the generator model (e.g. `"Qwen/Qwen2-7B-Instruct"`).
      - `stage_durations`: Dictionary of stage name → duration seconds.
      - `final_response`: Full generated recommendation text.

11. **Console output**  
    - The final recommendation is printed to the console:
      - Header: `"=== Clinical Recommendation ==="`
      - Followed by `final_response`.

---

## Testing and Extensibility Notes

- Each module is designed to be **individually testable**:
  - `src.ocr.extract_text_from_scanned_pdf` can be tested with a sample scanned PDF.
  - `GuidelineParser.parse` / `GuidelineParserStub.parse` can be tested with synthetic guideline text.
  - `TextEmbedder` and `FaissStore` can be tested with dummy strings and vectors.
  - `LocalGenerator.generate` can be tested with simple dummy inputs.


To run the full pipeline:

```bash
python main.py
```

Ensure required models and data files are available locally before running.
