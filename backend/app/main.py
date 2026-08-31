import logging
import mimetypes
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from . import admin, auth, conversations, db, docstate, ingest, llm, ocr, vectorstore
from .auth import User
from .config import settings
from .models import (
    AdminResetPasswordRequest,
    AdminSetDisabledRequest,
    AdminUserSummary,
    AuthResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ClearResponse,
    ConversationDetail,
    ConversationSummary,
    DocumentInfo,
    IngestResponse,
    LoginRequest,
    ProviderStatus,
    RegisterRequest,
    RenameRequest,
    SkippedUpload,
    SourceSnippet,
    UploadedDocument,
    UploadResponse,
    UserInfo,
)

log = logging.getLogger(__name__)

READ_CHUNK = 1024 * 1024

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()  # create the accounts/history schema before the first request
    yield


app = FastAPI(title="Document Analyzer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    BaseHTTPMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],    
    max_body_size=50 * 1024 * 1024  # 50MB
)

_bearer = HTTPBearer(auto_error=False)


@app.get("/api/health")
async def health():
    """Unauthenticated liveness probe.

    Doesn't report an indexed-chunk count here - the index is now per-user
    (see vectorstore.py), and there's no "current user" on an unauthenticated
    probe to count for.
    """
    return {"status": "ok"}


@app.post("/api/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    try:
        user = auth.register(req.username, req.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    token, expires_at = auth.create_session(user)
    return AuthResponse(
        token=token, username=user.username, expires_at=expires_at, is_admin=user.is_admin
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    try:
        user = auth.authenticate(req.username, req.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    token, expires_at = auth.create_session(user)
    return AuthResponse(
        token=token, username=user.username, expires_at=expires_at, is_admin=user.is_admin
    )


@app.post("/api/auth/logout", status_code=204)
async def logout(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    """Idempotent: revokes the presented token if it exists, 204 either way."""
    if creds and creds.credentials:
        auth.end_session(creds.credentials)


@app.get("/api/auth/me", response_model=UserInfo)
async def me(user: User = Depends(auth.current_user)):
    return UserInfo(username=user.username, is_admin=user.is_admin)


@app.get("/api/admin/users", response_model=list[AdminUserSummary])
async def admin_list_users(_admin: User = Depends(auth.require_admin)):
    return admin.list_users()


@app.patch("/api/admin/users/{user_id}", response_model=AdminUserSummary)
async def admin_set_disabled(
    user_id: int, req: AdminSetDisabledRequest, current_admin: User = Depends(auth.require_admin)
):
    if user_id == current_admin.id and req.disabled:
        raise HTTPException(status_code=400, detail="You can't disable your own account")
    try:
        admin.set_disabled(user_id, req.disabled)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")
    return admin.get_user(user_id)


@app.post("/api/admin/users/{user_id}/reset-password", status_code=204)
async def admin_reset_password(
    user_id: int, req: AdminResetPasswordRequest, _admin: User = Depends(auth.require_admin)
):
    if admin.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        auth.admin_set_password(user_id, req.new_password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/admin/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: int, current_admin: User = Depends(auth.require_admin)):
    """Deletes the account, their conversations/sessions, their document
    folder, and their vector collection. Irreversible - the frontend
    requires an explicit confirm before calling this.
    """
    if user_id == current_admin.id:
        raise HTTPException(status_code=400, detail="You can't delete your own account")
    try:
        admin.delete_user(user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/api/providers", response_model=list[ProviderStatus])
async def list_providers(user: User = Depends(auth.current_user)):
    """The answer-provider failover chain, in the order it will be tried.

    `order` is the position in the live chain, or 0 for a provider that is named
    in LLM_PROVIDERS but has no API key and will therefore be skipped.
    """
    active = llm.provider_chain()
    out = []
    for name in [p.strip().lower() for p in settings.llm_providers.split(",") if p.strip()]:
        if name not in llm.KNOWN_PROVIDERS:
            continue
        out.append(
            ProviderStatus(
                name=name,
                model=llm.model_name(name),
                configured=name in active,
                order=active.index(name) + 1 if name in active else 0,
            )
        )
    return out


@app.get("/api/conversations", response_model=list[ConversationSummary])
async def list_conversations(user: User = Depends(auth.current_user)):
    return [ConversationSummary(**row) for row in conversations.list_for_user(user.id)]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: int, user: User = Depends(auth.current_user)):
    summary = next(
        (c for c in conversations.list_for_user(user.id) if c["id"] == conversation_id), None
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=conversation_id,
        title=summary["title"],
        messages=[
            ChatMessage(
                role=m["role"],
                text=m["text"],
                sources=m["sources"],
                provider=m["provider"],
            )
            for m in conversations.messages(conversation_id, user.id)
        ],
    )


@app.patch("/api/conversations/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: int, req: RenameRequest, user: User = Depends(auth.current_user)
):
    title = conversations.make_title(req.title)
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title must not be empty")
    if not conversations.rename(conversation_id, user.id, title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    row = next(c for c in conversations.list_for_user(user.id) if c["id"] == conversation_id)
    return ConversationSummary(**row)


@app.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: int, user: User = Depends(auth.current_user)):
    if not conversations.delete(conversation_id, user.id):
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/api/documents", response_model=list[DocumentInfo])
async def list_documents(user: User = Depends(auth.current_user)):
    folder = ingest.user_folder(settings.documents_folder, user.id)
    on_disk = {p.name for p in ingest.list_documents(str(folder))}
    indexed = set(vectorstore.list_indexed_documents(user.id))
    return [DocumentInfo(name=name, indexed=name in indexed) for name in sorted(on_disk)]


@app.get("/api/documents/{name}/file")
async def get_document_file(name: str, user: User = Depends(auth.current_user)):
    """Serve one source file so chat citations can be opened, not just quoted.

    Matched by sanitized basename against the caller's own workspace folder
    rather than a raw path, for the same traversal reasons uploads are
    sanitized - `name` here comes straight from the client. Scoping to the
    caller's folder also means one user can't fetch another's file by name.
    """
    folder = ingest.user_folder(settings.documents_folder, user.id)
    safe_name = ingest.sanitize_filename(name)
    match = next(
        (p for p in ingest.list_documents(str(folder)) if p.name == safe_name),
        None,
    )
    if match is None:
        raise HTTPException(status_code=404, detail="Document not found")

    media_type = mimetypes.guess_type(match.name)[0] or "application/octet-stream"
    return FileResponse(match, media_type=media_type, filename=match.name)


@app.post("/api/documents/clear", response_model=ClearResponse)
async def clear_documents(user: User = Depends(auth.current_user)):
    """Delete every file in the caller's workspace and wipe their index clean.

    Destructive and irreversible - the frontend requires an explicit confirm
    before calling this. For starting over, rather than pruning one file at
    a time. Only ever touches the caller's own workspace.
    """
    folder = ingest.user_folder(settings.documents_folder, user.id)
    removed = 0
    for path in ingest.list_documents(str(folder)):
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            log.warning("[clear] could not delete %s: %s", path, exc)

    vectorstore.reset_collection(user.id)
    docstate.clear(user.id)

    return ClearResponse(removed=removed)


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_folder(force: bool = False, user: User = Depends(auth.current_user)):
    """Bring the caller's index in line with their workspace folder, incrementally.

    Only files that are new or changed since the last run are re-read; unchanged
    ones are left alone, which matters a lot once images are involved since each
    costs a full OCR pass. Files deleted from the folder have their chunks
    purged. Pass force=true to rebuild everything from scratch.
    """
    # Extraction (OCR, PDF parsing) and embedding are blocking CPU work; keep
    # them off the event loop so one big rescan doesn't stall other requests.
    return await run_in_threadpool(_run_ingest, force, user.id)


def _run_ingest(force: bool, user_id: int) -> IngestResponse:
    folder = ingest.user_folder(settings.documents_folder, user_id)
    known = docstate.load(user_id)
    plan = ingest.plan_incremental(
        str(folder),
        known,
        settings.chunk_size,
        settings.chunk_overlap,
        force=force,
    )

    # Files gone from disk shouldn't keep answering questions.
    for doc_path in plan.removed:
        vectorstore.delete_document_path(user_id, doc_path)
        docstate.forget(user_id, doc_path)
    removed = len(plan.removed)

    chunks_written = 0
    failed: list[SkippedUpload] = []
    indexed_added = indexed_updated = 0

    for path in plan.to_index:
        key = str(path)
        was_known = key in known
        try:
            size, mtime = docstate.signature(path)
            chunks = ingest.build_chunks_for_document(
                path, settings.chunk_size, settings.chunk_overlap
            )
        except Exception as exc:  # noqa: BLE001 - report per file, keep scanning
            log.warning("[ingest] skipped %s: %s", path, exc)
            failed.append(
                SkippedUpload(name=path.name, reason=str(exc) or exc.__class__.__name__)
            )
            continue

        # Clear this file's old chunks before writing new ones, so a document
        # that shrank doesn't leave the tail of its previous version behind.
        # Deleting by name as well as by path sweeps up strays written before
        # paths were normalised to absolute.
        vectorstore.delete_document_path(user_id, key)
        vectorstore.delete_document(user_id, path.name)

        _index_chunks(user_id, chunks)
        chunks_written += len(chunks)
        docstate.record(
            user_id=user_id,
            doc_path=key,
            doc_name=path.name,
            size=size,
            mtime=mtime,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_count=len(chunks),
        )
        if was_known:
            indexed_updated += 1
        else:
            indexed_added += 1

    # Reconcile whatever is left against the folder. plan.removed only knows
    # about files whose state was tracked, so this is what clears chunks orphaned
    # by files deleted before tracking existed - or by a changed folder setting.
    removed += _prune_orphans(user_id, {str(p) for p in plan.to_index + plan.unchanged})

    log.info(
        "[ingest] %d added, %d updated, %d unchanged, %d removed, %d failed (%d chunks written)",
        indexed_added,
        indexed_updated,
        len(plan.unchanged),
        removed,
        len(failed),
        chunks_written,
    )

    return IngestResponse(
        documents_found=plan.total_on_disk,
        chunks_indexed=chunks_written,
        added=indexed_added,
        updated=indexed_updated,
        unchanged=len(plan.unchanged),
        removed=removed,
        failed=failed,
    )


def _prune_orphans(user_id: int, live_paths: set[str]) -> int:
    """Drop chunks for any path the folder no longer contains."""
    pruned = 0
    for indexed_path in vectorstore.list_indexed_paths(user_id):
        if indexed_path in live_paths:
            continue
        log.info("[ingest] purging orphaned chunks for %s", indexed_path)
        vectorstore.delete_document_path(user_id, indexed_path)
        docstate.forget(user_id, indexed_path)
        pruned += 1
    return pruned


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...), user: User = Depends(auth.current_user)
):
    """Accept one or more uploads, write them into the caller's workspace
    folder, and index each straight away so it is queryable without a
    separate rescan. Unsupported or oversized files are reported back rather
    than aborting the whole batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    folder = ingest.user_folder(settings.documents_folder, user.id)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    uploaded: list[UploadedDocument] = []
    skipped: list[SkippedUpload] = []

    for upload in files:
        original = upload.filename or ""
        name = ingest.sanitize_filename(original)

        if not name:
            skipped.append(SkippedUpload(name=original or "(unnamed)", reason="Invalid file name"))
            continue
        if not ingest.is_supported(name):
            if ocr.is_image(name) and not settings.ocr_enabled:
                reason = "Image uploads need OCR - set OCR_ENABLED=true to accept them"
            else:
                supported = ", ".join(sorted(ingest.supported_extensions()))
                reason = f"Unsupported type - allowed: {supported}"
            skipped.append(SkippedUpload(name=name, reason=reason))
            continue

        try:
            dest = ingest.resolve_upload_path(str(folder), name)
        except ValueError as exc:
            skipped.append(SkippedUpload(name=name, reason=str(exc)))
            continue

        replaced = dest.exists()
        written = 0
        try:
            with dest.open("wb") as sink:
                while chunk := await upload.read(READ_CHUNK):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"Larger than the {settings.max_upload_mb} MB limit")
                    sink.write(chunk)
            if written == 0:
                raise ValueError("File is empty")
            # OCR on a large image can take several seconds - run it in the
            # threadpool rather than blocking the event loop.
            chunks = await run_in_threadpool(
                ingest.build_chunks_for_document,
                dest,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        except Exception as exc:  # noqa: BLE001 - report per file, keep the batch going
            # Don't leave a truncated or unreadable file behind, but never
            # clobber a good copy that this upload only partially overwrote.
            if not replaced:
                dest.unlink(missing_ok=True)
            skipped.append(SkippedUpload(name=name, reason=str(exc) or exc.__class__.__name__))
            continue
        finally:
            await upload.close()

        if replaced:
            vectorstore.delete_document(user.id, name)
            vectorstore.delete_document_path(user.id, str(dest))
        await run_in_threadpool(_index_chunks, user.id, chunks)

        # Record it as indexed so the next rescan treats it as unchanged rather
        # than re-extracting (and re-OCR'ing) what we just processed.
        try:
            size, mtime = docstate.signature(dest)
            docstate.record(
                user_id=user.id,
                doc_path=str(dest),
                doc_name=name,
                size=size,
                mtime=mtime,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                chunk_count=len(chunks),
            )
        except OSError as exc:  # noqa: PERF203 - non-fatal; worst case is a re-scan
            log.warning("[upload] could not record state for %s: %s", name, exc)

        # The file is stored either way, but with no chunks it can never come
        # back from a search - say so instead of reporting a bare success.
        note = None
        if not chunks:
            note = (
                "no readable text found - is the image blank, very low "
                "resolution, or handwritten?"
                if ocr.is_image(name)
                else "no extractable text found - if this is a scanned PDF it "
                "has no text layer"
            )

        uploaded.append(
            UploadedDocument(
                name=name, chunks_indexed=len(chunks), replaced=replaced, note=note
            )
        )

    if not uploaded and skipped:
        raise HTTPException(
            status_code=400,
            detail="; ".join(f"{s.name}: {s.reason}" for s in skipped),
        )

    return UploadResponse(uploaded=uploaded, skipped=skipped)


def _evenly_sample(items: list, n: int) -> list:
    """Pick up to `n` items spread across the list, keeping original order.

    Used to shrink an over-budget document down to a fixed size while still
    touching its beginning, middle, and end - plain head-truncation would
    silently drop everything past the budget instead.
    """
    if n >= len(items):
        return items
    if n <= 0:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _chunk_metadata(c: ingest.Chunk) -> dict:
    meta = {"doc_name": c.doc_name, "doc_path": c.doc_path, "chunk_index": c.chunk_index}
    if c.page is not None:  # Chroma metadata can't hold None - omit rather than store it
        meta["page"] = c.page
    return meta


def _index_chunks(user_id: int, chunks: list[ingest.Chunk]) -> None:
    if not chunks:
        return
    vectorstore.add_chunks(
        user_id=user_id,
        ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        metadatas=[_chunk_metadata(c) for c in chunks],
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: User = Depends(auth.current_user)):
    """Answer a question and append both turns to the user's chat history.

    Pass `conversation_id` to continue an existing thread, or omit it to start a
    new one - the reply carries the id and title to keep the UI in sync.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    if vectorstore.count(user.id) == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents indexed yet. Upload a document or rescan the folder first.",
        )

    title = conversations.make_title(question)
    if req.conversation_id is None:
        conversation_id = conversations.create(user.id, title)
        history: list[dict] = []
    else:
        conversation_id = req.conversation_id
        if not conversations.owned_by(conversation_id, user.id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = conversations.messages(
            conversation_id, user.id, limit=conversations.HISTORY_TURNS
        )
        title = next(
            c["title"] for c in conversations.list_for_user(user.id) if c["id"] == conversation_id
        )

    # Persist the question before calling out to Claude, so a failed or slow
    # generation still leaves the user's turn in their history.
    conversations.add_message(conversation_id, "user", question)

    full_document = req.doc_name is not None
    truncated = False

    if full_document:
        doc_chunks = vectorstore.get_document_chunks(user.id, req.doc_name)
        if not doc_chunks:
            raise HTTPException(
                status_code=404, detail=f"'{req.doc_name}' is not an indexed document."
            )

        total_chars = sum(len(c["text"]) for c in doc_chunks)
        if total_chars > settings.full_document_max_chars:
            avg_len = total_chars / len(doc_chunks)
            budget = max(1, int(settings.full_document_max_chars / avg_len))
            doc_chunks = _evenly_sample(doc_chunks, budget)
            truncated = True

        context_chunks = [c["text"] for c in doc_chunks]
        if truncated:
            context_chunks.insert(
                0,
                "[NOTE: this document is long - you are seeing excerpts sampled "
                "evenly across its full length, not the complete text.]",
            )
        sources = [
            SourceSnippet(doc_name=req.doc_name, excerpt=c["text"][:280], page=c["page"])
            for c in doc_chunks[:5]
        ]
    else:
        results = vectorstore.similarity_search(user.id, question, settings.top_k)
        context_chunks = [r.page_content for r in results]
        sources = [
            SourceSnippet(
                doc_name=r.metadata["doc_name"],
                excerpt=r.page_content[:280],
                page=r.metadata.get("page"),
            )
            for r in results
        ]

    try:
        answer = await llm.chat(question, context_chunks, history, cacheable=full_document)
    except llm.NoProviderConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except llm.AllProvidersFailed as exc:
        # 429 when everyone is simply rate limited (retrying later will work),
        # 502 when the providers failed for other reasons.
        raise HTTPException(
            status_code=429 if exc.all_rate_limited else 502,
            detail="Every answer provider is unavailable - "
            + "; ".join(f"{name} ({reason})" for name, reason in exc.failures),
        )

    conversations.add_message(
        conversation_id,
        "assistant",
        answer.text,
        [s.model_dump() for s in sources],
        provider=answer.provider,
    )

    return ChatResponse(
        answer=answer.text,
        sources=sources,
        conversation_id=conversation_id,
        title=title,
        provider=answer.provider,
        model=answer.model,
        full_document=full_document,
        truncated=truncated,
    )
