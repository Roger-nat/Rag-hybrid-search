# RAG Pipeline with Hybrid Search

Upload internal documents (PDF / TXT / DOCX), ask questions, and get answers that are
**grounded in your documents, with citations**. Retrieval combines semantic vector
search and BM25 keyword search, fuses them, reranks, and only then calls the LLM.
If the documents don't contain the answer, the system says so instead of guessing.

## Quick start (Docker)

```bash
cp .env.example .env          # then set ANTHROPIC_API_KEY
docker compose up --build
```

- UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

First start downloads two small Hugging Face models (~110 MB) into a Docker volume; later starts reuse them.
Documents and the vector DB persist in the `rag_data` volume.

## Run locally (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                              # set ANTHROPIC_API_KEY
uvicorn app.main:app --reload                     # terminal 1
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py   # terminal 2
pytest -q                                         # offline tests (no models / API key needed)
```

## Architecture

```
            ┌──────────────── INGESTION (POST /documents/upload) ────────────────┐
 file ──►  loaders ──► cleaner ──► chunker ──► embedder ──► ChromaDB (vectors + metadata)
 pdf/txt/docx  pages     normalise   sentence-aware   MiniLM          │
                                      size+overlap                    └──► BM25 index (rebuilt from Chroma)

            ┌──────────────────────── QUERY (POST /ask) ─────────────────────────┐
 question ─┬─► dense search (Chroma, cosine, top 20) ──┐
           └─► BM25 search (top 20) ───────────────────┴─► RRF fusion ─► cross-encoder rerank ─► top 5
                                                                                   │
                         answer + [n] citations  ◄── LLM (sources-only prompt) ◄──┘
                         "not found" if unsupported
```

```
app/
  config.py            env-driven settings
  models.py            Chunk, RetrievedChunk
  ingestion/           loaders.py · cleaner.py · chunker.py
  retrieval/           embedder.py · vector_store.py (dense) · bm25_index.py · hybrid.py (RRF)
  reranking/           reranker.py (cross-encoder)
  generation/          generator.py (prompt, LLM call, citation parsing)
  services/pipeline.py orchestration
  schemas.py · main.py Pydantic models · FastAPI routes
ui/streamlit_app.py    front-end (HTTP only)
tests/                 offline pytest suite
```

## How the RAG pipeline works

1. **Load** – `pypdf` (per page), `python-docx` (paragraphs + tables), UTF-8 text. DOCX/TXT count as page 1.
2. **Clean** – NFKC normalise, re-join hyphenated line breaks, collapse whitespace, keep paragraph breaks.
3. **Chunk** – sentence-aware packing up to `CHUNK_SIZE` characters; the last sentences (≤ `CHUNK_OVERLAP` chars) are repeated at the start of the next chunk. Chunks never cross pages, so page citations are exact.
4. **Embed + store** – `all-MiniLM-L6-v2` (normalised) into a persistent ChromaDB collection (cosine). Metadata: `doc_id`, `document`, `page`, `chunk_index`; chunk id = `<doc_id>_p<page>_c<n>`.
5. **BM25** – in-memory Okapi BM25 (k1=1.5, b=0.75), rebuilt from Chroma on startup and after each upload/delete.
6. **Hybrid** – Reciprocal Rank Fusion: `score = Σ 1/(60 + rank)` over both result lists. Rank-based, so cosine and BM25 scales never need normalising; chunks found by both retrievers rise to the top.
7. **Rerank** – cross-encoder `ms-marco-MiniLM-L-6-v2` rescores the ~20-40 fused candidates; top `TOP_K_FINAL` (5) go to the LLM. Set `RERANKER_ENABLED=false` to skip it.
8. **Generate** – the prompt allows only the supplied `<source>` blocks, tells the model to treat them as untrusted data, requires `[n]` after every claim, and requires the exact token `NOT_FOUND` otherwise.
9. **Guardrails in code (not just the prompt)** – empty index / no matches → LLM is never called; `NOT_FOUND` → "not found" message, no citations; an answer with no valid `[n]` marker is treated as unsupported; only the sources actually cited are returned as citations.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents/upload` | multipart `files=` (one or many). Extract → chunk → embed → index. Per-file `indexed`/`error` status. Same file name or same content replaces the old version. |
| GET | `/documents` | list indexed documents (pages, chunks) |
| DELETE | `/documents/{doc_id}` | remove a document from both indexes |
| POST | `/documents/reindex` | rebuild the BM25 index from the vector store |
| POST | `/search` | retrieval only: `{query, mode: hybrid\|dense\|bm25, top_k, rerank}` with ranks/scores per chunk |
| POST | `/ask` | `{question, top_k}` → `{answer, found, citations[], retrieval{...}, model}` |
| GET | `/health` | liveness + counts |

Errors: 422 validation, 404 unknown doc, 502 LLM failure, 503 missing `ANTHROPIC_API_KEY`.

```bash
curl -F "files=@handbook.pdf" localhost:8000/documents/upload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"What is the refund window?"}'
```

## Configuration

All in `.env` (see `.env.example`): `ANTHROPIC_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`,
`RERANKER_ENABLED`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_DENSE`, `TOP_K_BM25`, `TOP_K_FINAL`, `RRF_K`, `MAX_UPLOAD_MB`, `DATA_DIR`.
Changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL` only affects newly uploaded files; re-upload to re-chunk.

## Known limitations (be upfront about these in an interview)

- No OCR: scanned PDFs yield no text and are rejected with a clear error.
- BM25 is rebuilt in memory on every upload/delete: fine for thousands of chunks, not millions.
- Single-process design (write lock + in-memory BM25); scale-out would need a shared keyword index (e.g. OpenSearch, or Qdrant sparse vectors).
- No authentication or per-user document isolation.
- Fixed RRF fusion; no tuned weights or evaluation set yet. Next step would be a small labelled Q&A set to measure recall@k / answer faithfulness.
- Citation checking verifies `[n]` markers exist and are in range, not that each sentence is truly entailed by its source.

## Tests

`pytest -q` runs 29 offline tests: chunking/cleaning, loaders (DOCX/PDF), BM25, RRF, answer parsing, and the full API flow with a
hashing embedder and a stubbed LLM client (upload → search → ask → delete, validation, not-found, 503, persistence across restart).
