# Document Analyzer

Reads a folder of documents (`.txt`, `.md`, `.pdf`, `.docx`, plus images via
OCR), indexes them into a local vector store, and answers questions about them
through a chat UI - grounded only in what's actually in the folder (RAG), with
source excerpts shown for every answer.

Built on **LangChain**: retrieval (embeddings + vector search) stays local via a
small on-device sentence-transformers model, and answer generation runs through a
failover chain of hosted LLMs - **Groq → OpenRouter → Claude** by default. Only
your question plus the retrieved excerpts leave the machine, never the whole
corpus.

## Incremental indexing

A rescan only re-reads what changed. On each `/api/ingest` the app diffs the
folder against what it already indexed and:

- **new** files are extracted and embedded,
- **changed** files are re-extracted, with their previous chunks purged first,
- **unchanged** files are skipped entirely - not opened, not OCR'd,
- **deleted** files have their chunks removed from the index.

Each run also reconciles the index against the folder, so chunks belonging to any
path the folder no longer contains are purged. That covers files deleted before
tracking existed, and files left over from a previous `DOCUMENTS_FOLDER`. After a
rescan the index contains exactly what's on disk - nothing stale can keep
surfacing in answers.

This matters most for images: without it, every rescan paid a full OCR pass for
every image in the folder. Uploads record their own state too, so a file you just
uploaded isn't re-extracted by the next rescan.

Change detection is size + mtime (the rsync heuristic) - fast, and no file reads
for unchanged documents. A change in `CHUNK_SIZE` or `CHUNK_OVERLAP` also counts
as changed, so stale chunk boundaries don't silently survive a config change. An
edit that preserves both size and mtime would be missed, so
`POST /api/ingest?force=true` rebuilds everything unconditionally.

Bookkeeping lives in the `indexed_documents` table in `data/app.db`, keyed on the
resolved absolute path. Deleting that table (or the file) just makes the next
rescan a full one. The response reports `added` / `updated` / `unchanged` /
`removed` plus a `failed` list, and the left panel summarises it: *"Filed 1 new
(3 passages). 11 unchanged, skipped."*

## File types and image OCR

| Type                                          | How text is extracted          |
|-----------------------------------------------|--------------------------------|
| `.txt`, `.md`                                 | read directly                  |
| `.pdf`                                        | `pypdf` text layer             |
| `.docx`                                       | `python-docx` paragraphs       |
| `.png` `.jpg` `.jpeg` `.bmp` `.tif` `.tiff` `.webp` | **local OCR** ([ocr.py](backend/app/ocr.py)) |

Images go through **RapidOCR** (PaddleOCR models on onnxruntime). It's
pip-installable with no system binary to install - unlike Tesseract - and runs
entirely offline, so OCR stays on the same footing as embeddings: local, free,
and unaffected by chat-provider rate limits. Once text is extracted an image is
chunked, embedded, and cited in answers exactly like any other document.

Upload an image or drop it in `DOCUMENTS_FOLDER` and rescan; both paths work.
Lines OCR reports below `OCR_MIN_CONFIDENCE` (default 0.5) are discarded as
noise. Set `OCR_ENABLED=false` to reject images outright - they then become
uniformly unsupported, rejected at upload and ignored by folder scans, rather
than stored but unsearchable.

An image with no legible text still uploads and is stored, but indexes zero
chunks, so it can never come back from a search. The API reports this in the
`note` field and the UI shows it in amber rather than claiming plain success.
Accuracy is good on screenshots and clean scans, weaker on handwriting, heavy
skew, and low resolution.

**OCR is slow** - seconds per image, versus milliseconds for a text file. It
runs in a threadpool so it won't stall other requests, but a rescan over a
folder of many images takes a while.

## Answer providers and failover

Generation walks `LLM_PROVIDERS` left to right and uses the first provider that
answers. If one is rate limited or out of credit, the next takes over on the
same request - no retry needed from the user.

| Order | Provider   | Default model                        | Env var              |
|-------|------------|--------------------------------------|----------------------|
| 1     | Groq       | `llama-3.3-70b-versatile`            | `GROQ_API_KEY`       |
| 2     | OpenRouter | `meta-llama/llama-3.3-70b-instruct`  | `OPENROUTER_API_KEY` |
| 3     | Claude     | `claude-sonnet-4-6`                  | `ANTHROPIC_API_KEY`  |

Reorder or shorten the chain with `LLM_PROVIDERS` (e.g. `anthropic,groq` to
prefer Claude). **Providers with a blank API key are skipped automatically**, so
you can run on any subset - one key is enough to start.

Groq and OpenRouter both expose OpenAI-compatible APIs, so both are driven
through `ChatOpenAI` with a per-provider `base_url`; only `langchain-openai` was
added, not a client library per vendor. Client retries are disabled on purpose
(`max_retries=0` in [llm.py](backend/app/llm.py)) - sitting in SDK backoff on a
429 defeats the point of having a fallback ready.

Each answer records which provider produced it: the chat reply carries
`provider` and `model`, the UI shows a small "answered by Groq" caption under the
bubble, and it's persisted so reopened threads show it too. `GET /api/providers`
reports the live chain and which entries are actually usable.

When every provider fails, `/api/chat` returns **429** if they were all merely
rate limited (retry later and it will work) or **502** otherwise, with a detail
string naming each provider and its reason. **503** means no provider has a key
configured at all. Note that indexing and upload never touch these providers -
embeddings are local, so the document side keeps working during a total chat
outage.

## Architecture

```
document-analyzer/
├── backend/        FastAPI service: folder scan → chunk → embed → vector store → chat
│   └── app/
│       ├── ingest.py        text extraction + chunking + upload handling
│       ├── ocr.py           image -> text via local RapidOCR
│       ├── docstate.py      tracks what's indexed, for delta-only rescans
│       ├── llm.py           provider failover chain + local embeddings
│       ├── vectorstore.py   Chroma persistence
│       ├── db.py            SQLite schema: users, sessions, chat history
│       ├── auth.py          password hashing + session tokens
│       ├── conversations.py per-user saved chats
│       └── main.py          REST endpoints
└── frontend/        Angular 17 standalone app
    └── src/app/
        ├── components/login/       sign-in / sign-up gate on first load
        ├── components/documents/   left panel: upload, file list, rescan
        ├── components/chat/        center panel: questions, answers, source cards
        └── components/history/     right panel: previous chats + New chat button
```

## Accounts and chat history

The app opens on a sign-in form; there's no anonymous access. Create an account
from the same screen (**Create one**), and the session is remembered in
`localStorage` until it expires (`SESSION_TTL_HOURS`, default two weeks) or you
sign out.

Every exchange is saved automatically and listed newest-first in the right-hand
panel. Click a row to reopen that thread, **New chat** to start a fresh one, or
the `×` on a row to delete it. Reopened threads are true continuations - the
last few turns are replayed to Claude, so follow-ups like "what about the second
one?" resolve correctly.

Chat history is private per user. The **document corpus is shared** - anything
one user uploads is searchable by everyone, since there is a single documents
folder and one vector index behind it.

Accounts live in `data/app.db` (SQLite, created on first run). Passwords are
stored as PBKDF2-HMAC-SHA256 hashes with a per-user salt; session tokens are
random 256-bit values held server-side. Note that there's no rate limiting,
account lockout, or password reset flow, and tokens travel in an
`Authorization` header - fine for local or trusted-network use, but put it
behind HTTPS and add throttling before exposing it publicly.

## Prerequisites

- Python 3.11+
- Node.js 18+ and Angular CLI (`npm i -g @angular/cli`)
- At least one answer-provider key - Groq, OpenRouter, or Anthropic (see
  the table above). Groq has a free tier, so that's the cheapest start.
- ~90MB free for the local embedding model (downloads automatically on
  first run - no account or key needed for that part)

## Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set at least one of GROQ_API_KEY / OPENROUTER_API_KEY /
# ANTHROPIC_API_KEY, and point DOCUMENTS_FOLDER at the folder you want analyzed
mkdir -p data/documents data/vector_db
uvicorn app.main:app --reload --port 8000
```

Endpoints:

| Method | Path                          | Purpose                                     |
|--------|-------------------------------|---------------------------------------------|
| POST   | `/api/auth/register`          | Create an account, returns a session token  |
| POST   | `/api/auth/login`             | Sign in, returns a session token            |
| POST   | `/api/auth/logout`            | Revoke the presented token                  |
| GET    | `/api/auth/me`                | Who the current token belongs to            |
| GET    | `/api/documents`              | List files on disk and whether indexed      |
| POST   | `/api/documents/upload`       | Upload files into the folder and index them |
| POST   | `/api/ingest`                 | Rescan the folder, indexing only what changed |
| POST   | `/api/chat`                   | Ask a question, get an answer + sources     |
| GET    | `/api/providers`              | The failover chain and which keys are set   |
| GET    | `/api/conversations`          | Your saved chats, newest first              |
| GET    | `/api/conversations/{id}`     | Full message list for one chat              |
| PATCH  | `/api/conversations/{id}`     | Rename a chat                               |
| DELETE | `/api/conversations/{id}`     | Delete a chat and its messages              |
| GET    | `/api/health`                 | Liveness + indexed chunk count              |

Everything except `/api/health` and the two auth entry points requires an
`Authorization: Bearer <token>` header. `/api/chat` takes an optional
`conversation_id` - omit it to start a new thread, and the reply carries the new
id and title.

`/api/documents/upload` takes a multipart `files` field and accepts multiple
files per request. Each file is written into `documents_folder` and indexed
immediately, so no separate rescan is needed. Unsupported types and files over
`max_upload_mb` (default 50) are reported per file in the `skipped` list rather
than failing the whole batch; re-uploading a name replaces the stored copy and
its chunks.

## Run the frontend

```bash
cd frontend
npm install
ng serve
```

Open `http://localhost:4200`, create an account, then either drop files onto
**Add documents** in the left panel or click **Rescan folder** to index whatever
is already in `DOCUMENTS_FOLDER`. Ask away - each exchange is filed in the
**Previous Chats** panel on the right.

## Notes / next steps

- Chunking is a simple sliding character window (`ingest.py`) - swap for a
  token-aware or semantic splitter if answer quality needs tightening.
- **Scanned PDFs are not OCR'd.** A PDF with no text layer extracts as empty and
  indexes nothing (the upload `note` says so). Routing those pages through
  `ocr.py` would close the gap - the OCR engine is already in place.
- OCR quality is fixed at RapidOCR's defaults. For heavy scanned-document work,
  Tesseract via `pytesseract` is usually more accurate on clean printed text,
  at the cost of a system binary install.
- Chroma runs embedded (no separate server) - fine for a single machine;
  move to a hosted Qdrant/Chroma server for multi-user or larger corpora.
- Orphan reconciliation reads all chunk metadata on every rescan. Negligible at
  a few thousand chunks; if the corpus grows to six figures, gate it behind a
  flag or run it periodically instead of every time.
- Add a file-watcher (`watchdog` is already in requirements.txt) to trigger the
  incremental rescan automatically on file changes instead of manually. Now that
  rescans are cheap when nothing changed, this is a small step.
- Model IDs go stale. `CHAT_MODEL` defaults to `claude-sonnet-4-6`; check
  https://docs.claude.com/en/docs/about-claude/models/overview for the current
  recommendation, and Groq/OpenRouter's own model lists for theirs - a retired
  ID shows up as a 404 and the chain silently moves to the next provider, so
  check `GET /api/providers` and the server log if answers stop coming from
  where you expect.
- Failover is per request, with no memory. A provider that just 429'd is still
  tried first on the next question. Adding a short-lived cooldown per provider
  would cut wasted round-trips under sustained rate limiting.
- Retrieved excerpts are sent to whichever provider answers, so with the
  default chain your document text can reach Groq, OpenRouter, or Anthropic
  depending on availability. If that data-flow doesn't fit your compliance
  needs, narrow `LLM_PROVIDERS` to the one vendor you've cleared, or point
  `llm.py` at a fully local model (e.g. Ollama) - nothing else needs to change.
