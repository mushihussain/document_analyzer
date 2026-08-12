"""
Image-to-text via RapidOCR (PaddleOCR models running on onnxruntime).

Chosen because it is pip-installable with no system binary to chase down, runs
entirely offline once the bundled models are on disk, and reuses the onnxruntime
that Chroma already pulls in. That keeps OCR on the same footing as embeddings:
local, free, and unaffected by chat-provider rate limits.

The engine is loaded lazily - importing this module must stay cheap, because the
API imports it at startup whether or not anyone uploads an image.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# RapidOCR returns (text, confidence) per detected line; drop the ones it is
# barely guessing at, which are usually noise from compression artefacts.
_engine = None
_engine_failed = False


class OcrUnavailable(RuntimeError):
    """RapidOCR could not be loaded, so image text extraction is impossible."""


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _get_engine():
    """Build the OCR engine once and reuse it; model load takes a second or two."""
    global _engine, _engine_failed
    if _engine is not None:
        return _engine
    if _engine_failed:
        raise OcrUnavailable(
            "OCR engine unavailable - reinstall with: pip install rapidocr-onnxruntime"
        )
    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        log.info("[ocr] RapidOCR engine loaded")
        return _engine
    except Exception as exc:  # noqa: BLE001 - surface as a clean, actionable error
        _engine_failed = True
        log.error("[ocr] could not load RapidOCR: %s", exc)
        raise OcrUnavailable(f"OCR engine unavailable: {exc}") from exc


def extract_text(path: Path) -> str:
    """Read the visible text out of an image file.

    Returns "" when the image genuinely holds no legible text - callers treat
    that as "stored but not searchable" rather than an error, since a blank or
    purely graphical image is a valid thing to upload.
    """
    if not settings.ocr_enabled:
        raise OcrUnavailable("OCR is disabled (set OCR_ENABLED=true to index images)")

    engine = _get_engine()
    result, _elapsed = engine(str(path))
    if not result:
        return ""

    lines: list[str] = []
    dropped = 0
    for entry in result:
        # Each entry is [box, text, confidence]
        text = (entry[1] or "").strip()
        confidence = float(entry[2]) if len(entry) > 2 and entry[2] is not None else 1.0
        if not text:
            continue
        if confidence < settings.ocr_min_confidence:
            dropped += 1
            continue
        lines.append(text)

    if dropped:
        log.info(
            "[ocr] %s: dropped %d low-confidence line(s) below %.2f",
            path.name,
            dropped,
            settings.ocr_min_confidence,
        )

    return "\n".join(lines)
