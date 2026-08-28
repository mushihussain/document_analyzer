"""
Central configuration for the Document Analyzer backend.
All values can be overridden with environment variables (see .env.example).
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Folder that will be scanned for source documents (pdf, docx, txt, md)
    documents_folder: str = "./data/documents"

    # Where the Chroma vector store persists its index on disk
    vector_db_path: str = "./data/vector_db"
    collection_name: str = "documents"

    # SQLite file holding user accounts, sessions, and saved chat history
    db_path: str = "./data/app.db"

    # How long a login stays valid (default: two weeks)
    session_ttl_hours: int = 24 * 14

    # Comma-separated usernames with access to the admin page (/api/admin/*).
    # Matched case-insensitively; blank means nobody has admin access. Editing
    # this and restarting the backend is currently the only way to grant or
    # revoke admin - there's no in-app UI for it, to keep that capability out
    # of reach of anyone who isn't already trusted with server config.
    admin_usernames: str = ""

    # --- Answer generation ---------------------------------------------------
    # Providers are tried left to right; when one is rate limited or out of
    # quota the next one answers instead. Providers without an API key are
    # skipped, so you can run on any subset of these.
    llm_providers: str = "groq,openrouter,anthropic"

    # Groq (OpenAI-compatible endpoint) - https://console.groq.com/keys
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenRouter (OpenAI-compatible endpoint) - https://openrouter.ai/keys
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"

    # Claude - last resort in the chain (see llm_providers above), so it's
    # rarely on the hot path. Haiku is the right default for that: cheap and
    # fast, with quality to spare for a fallback role. Bump to a Sonnet/Opus
    # model here if the fallback answers need to be smarter, not just cheaper.
    anthropic_api_key: str = ""
    chat_model: str = "claude-haiku-4-5-20251001"

    # Per-provider request timeout, seconds. Kept short so a stalled provider
    # hands over to the next one quickly instead of hanging the request.
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2

    # Local embedding model (Claude has no embeddings endpoint, so retrieval
    # uses a small local sentence-transformers model - free, on-prem, no
    # API calls). Swap for a hosted embedding provider later if you prefer.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 1400
    chunk_overlap: int = 200

    # Largest single file accepted by the upload endpoint
    max_upload_mb: int = 50

    # --- OCR (image -> text) -------------------------------------------------
    # Runs locally via RapidOCR/onnxruntime - no API calls, no cost. Turn off to
    # reject image uploads outright.
    ocr_enabled: bool = True
    # Detected lines below this confidence are discarded as noise (0.0 - 1.0).
    ocr_min_confidence: float = 0.5

    # Retrieval
    # Chunks pulled per question in normal (similarity-search) mode. Higher
    # values cover more of a document per answer at the cost of more tokens
    # sent to the LLM - 12 chunks * 1400 chars is still well inside every
    # configured provider's context window.
    top_k: int = 12

    # "Whole document" chat mode (see ChatRequest.doc_name) skips similarity
    # search and feeds every chunk of the chosen document instead. If that
    # would exceed this character budget, chunks are evenly sampled across
    # the document so the model still sees the beginning, middle, and end
    # rather than just being cut off partway through.
    full_document_max_chars: int = 120_000

    # CORS - Angular dev server
    allowed_origins: list[str] = ["http://localhost:4200"]

    class Config:
        env_file = ".env"


settings = Settings()
