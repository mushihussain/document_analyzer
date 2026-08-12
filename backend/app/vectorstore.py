"""
Chroma-backed vector store, wired through LangChain's Chroma integration so
it shares the same embeddings object as the rest of the pipeline.
"""
from __future__ import annotations

from langchain_chroma import Chroma

from .config import settings
from .llm import get_embeddings

_store: Chroma | None = None


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=settings.vector_db_path,
        )
    return _store


def reset_collection() -> None:
    get_store().delete_collection()
    global _store
    _store = None


def add_chunks(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    if not ids:
        return
    store = get_store()
    # upsert semantics: drop any existing rows with these ids first, since
    # Chroma's add_texts doesn't overwrite in place
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def delete_document(doc_name: str) -> None:
    """Drop every chunk belonging to one document.

    Needed when a file is replaced by a shorter version - chunk ids are derived
    from (path, index), so the tail of the previous version would otherwise
    linger in the index and keep surfacing in retrieval.
    """
    try:
        get_store()._collection.delete(where={"doc_name": doc_name})
    except Exception as exc:  # noqa: BLE001 - a failed cleanup must not fail the upload
        print(f"[vectorstore] could not purge chunks for {doc_name}: {exc}")


def delete_document_path(doc_path: str) -> None:
    """Drop chunks for one exact file path.

    More precise than delete_document(): two files with the same basename in
    different subfolders share a doc_name, so name-based deletion would take out
    both. Used by incremental reindexing, where only one file actually changed.
    """
    try:
        get_store()._collection.delete(where={"doc_path": doc_path})
    except Exception as exc:  # noqa: BLE001 - a failed cleanup must not fail the run
        print(f"[vectorstore] could not purge chunks for {doc_path}: {exc}")


def similarity_search(query: str, top_k: int):
    return get_store().similarity_search(query, k=top_k)


def count() -> int:
    return get_store()._collection.count()


def list_indexed_paths() -> list[str]:
    """Every distinct doc_path currently in the index.

    Used to reconcile the index against the folder, which catches chunks left
    behind by files deleted before their state was being tracked.
    """
    data = get_store()._collection.get(include=["metadatas"])
    paths = {m["doc_path"] for m in data.get("metadatas", []) if m and m.get("doc_path")}
    return sorted(paths)


def list_indexed_documents() -> list[str]:
    data = get_store()._collection.get(include=["metadatas"])
    names = {m["doc_name"] for m in data.get("metadatas", []) if m}
    return sorted(names)


def get_document_chunks(doc_name: str) -> list[dict]:
    """Every indexed chunk of one document, in original reading order.

    Used by "whole document" chat mode to bypass similarity search - it
    trades "most relevant" for "complete", so callers get the full text back
    (modulo any downstream truncation) instead of just the best-matching
    fragments. Each entry is {"text": ..., "page": int | None}.
    """
    data = get_store()._collection.get(
        where={"doc_name": doc_name}, include=["metadatas", "documents"]
    )
    pairs = sorted(
        zip(data.get("metadatas", []), data.get("documents", [])),
        key=lambda pair: pair[0].get("chunk_index", 0),
    )
    return [{"text": text, "page": meta.get("page")} for meta, text in pairs]
