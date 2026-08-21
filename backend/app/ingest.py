"""
Folder scanning, text extraction, and chunking.

Supports: .txt, .md, .pdf, .docx, and images (.png/.jpg/.jpeg/.bmp/.tif/
.tiff/.webp) which are run through local OCR - see ocr.py.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument

from . import docstate, ocr
from .config import settings

TEXT_EXTENSIONS = {".txt", ".md"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx"}


def supported_extensions() -> set[str]:
    """What the app will accept right now.

    Images are only included when OCR is switched on, so turning OCR off makes
    them uniformly unsupported - rejected at upload and ignored by folder scans,
    rather than accepted and then silently unsearchable.
    """
    extensions = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS
    if settings.ocr_enabled:
        extensions = extensions | ocr.IMAGE_EXTENSIONS
    return extensions

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    doc_name: str
    text: str
    chunk_index: int
    # 1-indexed source page, when known (PDFs only) - lets a chat citation
    # jump the reader straight to where the answer came from.
    page: int | None = None


def user_folder(base: str, user_id: int) -> Path:
    """Where one user's documents live: a private subfolder under `base`.

    Keyed on the numeric user id rather than username - it's stable (doesn't
    change if the account is ever renamed) and needs no sanitizing, unlike an
    arbitrary user-supplied name.
    """
    return Path(base).resolve() / f"user_{user_id}"


def list_documents(folder: str) -> list[Path]:
    """Supported files under `folder`, as resolved absolute paths.

    Resolving matters: chunk ids are derived from the path, and uploads write to
    an absolute destination. If scans returned relative paths the same file
    would get two different id sets and end up double-indexed.
    """
    root = Path(folder)
    if not root.exists():
        return []
    allowed = supported_extensions()
    return sorted(
        p.resolve() for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed
    )


def sanitize_filename(name: str) -> str:
    """Reduce a client-supplied filename to a bare, safe basename.

    Strips any directory components (defeats ../ traversal), drops characters
    Windows/POSIX disallow, and collapses leading dots so uploads can never
    escape the documents folder or land as a hidden file.
    """
    base = Path(name.replace("\\", "/")).name
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")
    return base[:180]


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in supported_extensions()


def resolve_upload_path(folder: str, filename: str) -> Path:
    """Absolute destination inside `folder` for a sanitized filename."""
    root = Path(folder).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dest = (root / filename).resolve()
    if root not in dest.parents:
        raise ValueError(f"Refusing to write outside the documents folder: {filename}")
    return dest


def extract_text(path: Path) -> str:
    return "\n".join(extract_pages(path))


def extract_pages(path: Path) -> list[str]:
    """Text broken out per page for PDFs; a single "page" for everything else.

    Keeping pages separate (rather than joining them, as extract_text used
    to) is what lets chunk_text_with_pages() report which page a chunk came
    from - joined text has no boundary left to recover it from.
    """
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return [path.read_text(encoding="utf-8", errors="ignore")]
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return ["\n".join(p.text for p in doc.paragraphs)]
    if suffix in ocr.IMAGE_EXTENSIONS:
        return [ocr.extract_text(path)]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window chunker over characters (fast, dependency-light).
    Good enough for a first-pass RAG pipeline; swap for a token-aware
    splitter later if you need tighter control over context budgets.
    """
    return [piece for piece, _page in chunk_text_with_pages([text], chunk_size, overlap)]


def chunk_text_with_pages(
    pages: list[str], chunk_size: int, overlap: int
) -> list[tuple[str, int]]:
    """Sliding-window chunker over a document's pages, tagging each chunk
    with the (1-indexed) page it starts on.

    Each page is whitespace-normalized on its own before joining, so
    collapsing whitespace can't smear two pages' text together and erase the
    boundary between them. Page starts are recorded as offsets into the
    joined text, then looked up per chunk with a binary search.
    """
    normalized = [" ".join(p.split()) for p in pages]

    page_starts: list[int] = []
    parts: list[str] = []
    cursor = 0
    for page_text in normalized:
        page_starts.append(cursor)
        if page_text:
            parts.append(page_text)
            cursor += len(page_text) + 1  # +1 for the space join() will add
    text = " ".join(parts)
    if not text:
        return []

    def page_for(offset: int) -> int:
        return max(bisect.bisect_right(page_starts, offset) - 1, 0) + 1

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append((text[start:end], page_for(start)))
        start = end - overlap
        if start <= 0:
            break
    return chunks


def build_chunks_for_document(path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    is_pdf = path.suffix.lower() == ".pdf"
    pieces = chunk_text_with_pages(extract_pages(path), chunk_size, overlap)
    chunks = []
    for i, (piece, page) in enumerate(pieces):
        digest = hashlib.sha1(f"{path}:{i}".encode()).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=f"{digest}",
                doc_path=str(path),
                doc_name=path.name,
                text=piece,
                chunk_index=i,
                page=page if is_pdf else None,
            )
        )
    return chunks


def scan_and_chunk_folder(folder: str, chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc_path in list_documents(folder):
        try:
            all_chunks.extend(build_chunks_for_document(doc_path, chunk_size, overlap))
        except Exception as exc:  # noqa: BLE001 - surface but keep scanning
            print(f"[ingest] skipped {doc_path}: {exc}")
    return all_chunks


# Filesystem mtimes round-trip through SQLite as doubles; allow for the last bit.
_MTIME_EPSILON = 1e-6


@dataclass
class IngestPlan:
    """What a rescan needs to do, decided before any expensive work happens."""

    added: list[Path]
    modified: list[Path]
    unchanged: list[Path]
    # Paths still recorded as indexed but no longer present on disk.
    removed: list[str]

    @property
    def to_index(self) -> list[Path]:
        return self.added + self.modified

    @property
    def total_on_disk(self) -> int:
        return len(self.added) + len(self.modified) + len(self.unchanged)


def plan_incremental(
    folder: str,
    known: dict[str, dict],
    chunk_size: int,
    overlap: int,
    force: bool = False,
) -> IngestPlan:
    """Diff the folder against what's already indexed.

    A file counts as modified when its size or mtime moved, or when the chunk
    settings changed since it was indexed - stale chunk boundaries would
    otherwise survive a config change unnoticed.
    """
    added: list[Path] = []
    modified: list[Path] = []
    unchanged: list[Path] = []

    on_disk = list_documents(folder)
    seen: set[str] = set()

    for path in on_disk:
        key = str(path)
        seen.add(key)
        previous = known.get(key)

        if previous is None:
            added.append(path)
            continue
        if force:
            modified.append(path)
            continue

        try:
            size, mtime = docstate.signature(path)
        except OSError:
            # Unreadable right now - treat as changed so it gets retried.
            modified.append(path)
            continue

        changed = (
            size != previous["size"]
            or abs(mtime - previous["mtime"]) > _MTIME_EPSILON
            or previous["chunk_size"] != chunk_size
            or previous["chunk_overlap"] != overlap
        )
        (modified if changed else unchanged).append(path)

    removed = [key for key in known if key not in seen]

    return IngestPlan(added=added, modified=modified, unchanged=unchanged, removed=removed)
