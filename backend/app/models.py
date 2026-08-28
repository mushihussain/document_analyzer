from pydantic import BaseModel


class SkippedUpload(BaseModel):
    name: str
    reason: str


class IngestResponse(BaseModel):
    documents_found: int
    chunks_indexed: int
    # Incremental breakdown: only added + updated were actually re-read.
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: list[SkippedUpload] = []


class DocumentInfo(BaseModel):
    name: str
    indexed: bool


class ClearResponse(BaseModel):
    removed: int


class UploadedDocument(BaseModel):
    name: str
    chunks_indexed: int
    replaced: bool
    # Set when the file stored fine but produced nothing searchable, e.g. an
    # image with no legible text or a scanned PDF with no text layer.
    note: str | None = None


class UploadResponse(BaseModel):
    uploaded: list[UploadedDocument]
    skipped: list[SkippedUpload]


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    expires_at: str
    is_admin: bool = False


class UserInfo(BaseModel):
    username: str
    is_admin: bool = False


class ChatRequest(BaseModel):
    question: str
    # Omit to start a fresh conversation; the server returns the new id.
    conversation_id: int | None = None
    # Omit (or leave null) for normal similarity-search retrieval across all
    # documents. Set to an indexed document's name to switch to "whole
    # document" mode: every chunk of that document is used as context instead
    # of just the top-k most similar ones - see full_document_max_chars.
    doc_name: str | None = None


class SourceSnippet(BaseModel):
    doc_name: str
    excerpt: str
    # 1-indexed source page, when known (PDFs only) - lets the UI open the
    # citation straight to where the answer came from instead of page one.
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
    conversation_id: int
    title: str
    # Which provider in the failover chain actually answered.
    provider: str
    model: str
    # Echoes whether this answer used whole-document mode, so the UI can
    # label it even after the conversation is reloaded from history.
    full_document: bool = False
    # Set only when full_document is true and the document was too long to
    # include in full - chunks were evenly sampled across it instead.
    truncated: bool = False


class ProviderStatus(BaseModel):
    name: str
    model: str
    configured: bool
    order: int


class ChatMessage(BaseModel):
    role: str
    text: str
    sources: list[SourceSnippet] = []
    # Null for user turns, and for assistant turns saved before multi-provider
    # support existed.
    provider: str | None = None


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetail(BaseModel):
    id: int
    title: str
    messages: list[ChatMessage]


class RenameRequest(BaseModel):
    title: str


class AdminUserSummary(BaseModel):
    id: int
    username: str
    created_at: str
    is_admin: bool
    is_disabled: bool
    document_count: int
    conversation_count: int


class AdminSetDisabledRequest(BaseModel):
    disabled: bool


class AdminResetPasswordRequest(BaseModel):
    new_password: str
