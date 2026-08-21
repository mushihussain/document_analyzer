"""
Chroma-backed vector store, wired through LangChain's Chroma integration so
it shares the same embeddings object as the rest of the pipeline.

Each user gets their own collection (see _collection_name) - a single Chroma
client/persist_directory can hold any number of named collections, so this is
just an isolation boundary, not a separate store per user. It means a bug in
one query path can't leak another user's chunks into their answers the way a
shared collection filtered by user_id could.
"""
from __future__ import annotations

from langchain_chroma import Chroma

from .config import settings
from .llm import get_embeddings

_stores: dict[int, Chroma] = {}


def _collection_name(user_id: int) -> str:
    return f"{settings.collection_name}_u{user_id}"


def get_store(user_id: int) -> Chroma:
    store = _stores.get(user_id)
    if store is None:
        store = Chroma(
            collection_name=_collection_name(user_id),
            embedding_function=get_embeddings(),
            persist_directory=settings.vector_db_path,
        )
        _stores[user_id] = store
    return store


def reset_collection(user_id: int) -> None:
    get_store(user_id).delete_collection()
    _stores.pop(user_id, None)


def add_chunks(user_id: int, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    if not ids:
        return
    store = get_store(user_id)
    # upsert semantics: drop any existing rows with these ids first, since
    # Chroma's add_texts doesn't overwrite in place
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def delete_document(user_id: int, doc_name: str) -> None:
    """Drop every chunk belonging to one document.

    Needed when a file is replaced by a shorter version - chunk ids are derived
    from (path, index), so the tail of the previous version would otherwise
    linger in the index and keep surfacing in retrieval.
    """
    try:
        get_store(user_id)._collection.delete(where={"doc_name": doc_name})
    except Exception as exc:  # noqa: BLE001 - a failed cleanup must not fail the upload
        print(f"[vectorstore] could not purge chunks for {doc_name}: {exc}")


def delete_document_path(user_id: int, doc_path: str) -> None:
    """Drop chunks for one exact file path.

    More precise than delete_document(): two files with the same basename in
    different subfolders share a doc_name, so name-based deletion would take out
    both. Used by incremental reindexing, where only one file actually changed.
    """
    try:
        get_store(user_id)._collection.delete(where={"doc_path": doc_path})
    except Exception as exc:  # noqa: BLE001 - a failed cleanup must not fail the run
        print(f"[vectorstore] could not purge chunks for {doc_path}: {exc}")


def similarity_search(user_id: int, query: str, top_k: int):
    return get_store(user_id).similarity_search(query, k=top_k)


def count(user_id: int) -> int:
    return get_store(user_id)._collection.count()


def list_indexed_paths(user_id: int) -> list[str]:
    """Every distinct doc_path currently in this user's index.

    Used to reconcile the index against the folder, which catches chunks left
    behind by files deleted before their state was being tracked.
    """
    data = get_store(user_id)._collection.get(include=["metadatas"])
    paths = {m["doc_path"] for m in data.get("metadatas", []) if m and m.get("doc_path")}
    return sorted(paths)


def list_indexed_documents(user_id: int) -> list[str]:
    data = get_store(user_id)._collection.get(include=["metadatas"])
    names = {m["doc_name"] for m in data.get("metadatas", []) if m}
    return sorted(names)


def get_document_chunks(user_id: int, doc_name: str) -> list[dict]:
    """Every indexed chunk of one document, in original reading order.

    Used by "whole document" chat mode to bypass similarity search - it
    trades "most relevant" for "complete", so callers get the full text back
    (modulo any downstream truncation) instead of just the best-matching
    fragments. Each entry is {"text": ..., "page": int | None}.
    """
    data = get_store(user_id)._collection.get(
        where={"doc_name": doc_name}, include=["metadatas", "documents"]
    )
    pairs = sorted(
        zip(data.get("metadatas", []), data.get("documents", [])),
        key=lambda pair: pair[0].get("chunk_index", 0),
    )
    return [{"text": text, "page": meta.get("page")} for meta, text in pairs]
