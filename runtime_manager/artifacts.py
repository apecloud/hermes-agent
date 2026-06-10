from __future__ import annotations

import hashlib
import mimetypes
import re
from pathlib import Path
from typing import Any

_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_BROWSABLE_MIME_PREFIXES = ("text/",)
_BROWSABLE_MIME_TYPES = {
    "application/pdf",
}
DEFAULT_EXTRACT_BYTES = 64 * 1024
MAX_EXTRACT_BYTES = 256 * 1024
_EXTRACTABLE_MIME_PREFIXES = ("text/",)
_EXTRACTABLE_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/yaml",
    "application/x-yaml",
}


def artifact_dir_for(user_home: Path, session_id: str, run_id: str) -> Path:
    session_segment = safe_artifact_segment(session_id, fallback="session")
    run_segment = safe_artifact_segment(run_id, fallback="run")
    return user_home / "sessions" / f"{session_segment}.artifacts" / run_segment


def safe_artifact_segment(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    text = _SEGMENT_PATTERN.sub("_", text)
    text = text.strip("._-")[:80]
    return text or fallback


def metadata_for_artifact_file(
    path: Path,
    artifact_root: Path,
    *,
    source: str = "tool",
    kind: str = "report",
    artifact_id: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    root = artifact_root.resolve()
    file_path = path.resolve()
    file_path.relative_to(root)
    if not file_path.is_file():
        raise FileNotFoundError(str(path))

    size_bytes, digest = file_digest(file_path)
    file_name = file_path.name
    mime_type = infer_mime_type(file_path)
    resolved_artifact_id = artifact_id or artifact_id_for_file(file_path, artifact_root, digest)
    return {
        "artifactId": resolved_artifact_id,
        "fileName": file_name,
        "sizeBytes": size_bytes,
        "mimeType": mime_type,
        "sha256": digest,
        "kind": kind,
        "source": source,
        "summary": summary or f"Generated report file: {file_name}",
        "canDownload": True,
        "canBrowse": can_browse_mime_type(mime_type),
        "boundedExtraction": bounded_extraction_capability(mime_type),
    }


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def artifact_id_for_file(path: Path, artifact_root: Path, sha256: str) -> str:
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError:
        relative = path.name
    stem = Path(str(relative)).stem or Path(str(relative)).name
    name_segment = safe_artifact_segment(stem, fallback="artifact")[:48]
    return f"{name_segment}-{sha256[:16]}"


def infer_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed == "text/html":
        return "text/html; charset=utf-8"
    if guessed and guessed.startswith("text/"):
        return f"{guessed}; charset=utf-8"
    if guessed:
        return guessed
    return "text/plain; charset=utf-8"


def can_browse_mime_type(mime_type: str) -> bool:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    if not normalized:
        return False
    return normalized in _BROWSABLE_MIME_TYPES or normalized.startswith(_BROWSABLE_MIME_PREFIXES)


def can_extract_mime_type(mime_type: str) -> bool:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    if not normalized:
        return False
    return normalized in _EXTRACTABLE_MIME_TYPES or normalized.startswith(_EXTRACTABLE_MIME_PREFIXES)


def bounded_extraction_capability(mime_type: str) -> dict[str, Any]:
    """Describe the runtime contract for model-facing bounded reads.

    Cloud may expose this through its own auth layer, but the runtime never
    returns full artifact content through metadata or turn-action events.
    """
    return {
        "available": can_extract_mime_type(mime_type),
        "defaultBytes": DEFAULT_EXTRACT_BYTES,
        "maxBytes": MAX_EXTRACT_BYTES,
        "supportsOffset": True,
        "encoding": "utf-8",
    }


def extract_artifact_text(
    path: Path,
    artifact_root: Path,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return a bounded UTF-8 text slice from a controlled artifact file."""
    root = artifact_root.resolve()
    file_path = path.resolve()
    file_path.relative_to(root)
    metadata = metadata_for_artifact_file(file_path, root)
    if not can_extract_mime_type(str(metadata.get("mimeType") or "")):
        raise ValueError("artifact is not text-extractable")

    safe_offset = max(0, int(offset or 0))
    requested_limit = int(limit or DEFAULT_EXTRACT_BYTES)
    safe_limit = min(max(1, requested_limit), MAX_EXTRACT_BYTES)
    size_bytes = int(metadata.get("sizeBytes") or 0)
    if safe_offset >= size_bytes:
        chunk = b""
    else:
        with file_path.open("rb") as handle:
            handle.seek(safe_offset)
            chunk = handle.read(safe_limit)

    next_offset = safe_offset + len(chunk)
    truncated = next_offset < size_bytes
    text = chunk.decode("utf-8", errors="replace")
    return {
        "object": "runtime_manager.artifact_extraction",
        "artifactId": metadata["artifactId"],
        "fileName": metadata["fileName"],
        "sizeBytes": metadata["sizeBytes"],
        "mimeType": metadata["mimeType"],
        "sha256": metadata["sha256"],
        "kind": metadata["kind"],
        "source": metadata["source"],
        "summary": metadata["summary"],
        "text": text,
        "offset": safe_offset,
        "limit": safe_limit,
        "extractedBytes": len(chunk),
        "nextOffset": next_offset if truncated else None,
        "truncated": truncated,
        "encoding": "utf-8",
    }


def discover_artifacts(
    artifact_root: Path,
    *,
    seen_artifact_ids: set[str] | None = None,
    max_files: int = 50,
) -> list[dict[str, Any]]:
    try:
        root = artifact_root.resolve()
    except Exception:
        return []
    if not root.is_dir():
        return []

    seen = seen_artifact_ids if seen_artifact_ids is not None else set()
    artifacts: list[dict[str, Any]] = []
    try:
        candidates = sorted(root.rglob("*"))
    except OSError:
        return []
    for path in candidates:
        if len(artifacts) >= max_files:
            break
        try:
            metadata = metadata_for_artifact_file(path, root)
        except (FileNotFoundError, OSError, ValueError):
            continue
        artifact_id = metadata["artifactId"]
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        artifacts.append(metadata)
    return artifacts


def find_artifact_file(
    artifact_root: Path,
    artifact_id: str,
) -> tuple[Path, dict[str, Any]] | None:
    requested_id = str(artifact_id or "").strip()
    if not requested_id or "/" in requested_id or "\\" in requested_id:
        return None
    try:
        root = artifact_root.resolve()
    except Exception:
        return None
    if not root.is_dir():
        return None
    try:
        candidates = sorted(root.rglob("*"))
    except OSError:
        return None
    for path in candidates:
        try:
            metadata = metadata_for_artifact_file(path, root)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if path.stem == requested_id or metadata["artifactId"] == requested_id:
            return path.resolve(), metadata
    return None
