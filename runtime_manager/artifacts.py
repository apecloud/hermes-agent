from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
from pathlib import Path
from typing import Any

_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_PUBLISHED_DIR_NAME = ".published"
_BROWSABLE_MIME_PREFIXES = ("text/",)
_BROWSABLE_MIME_TYPES = {
    "application/pdf",
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
    _ = sha256
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError:
        relative = path.name
    relative_text = str(relative)
    suffix = Path(relative_text).suffix
    if suffix:
        relative_text = relative_text[: -len(suffix)]
    return safe_artifact_segment(relative_text, fallback="artifact")[:64]


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
        if _is_published_path(path, root):
            continue
        try:
            metadata = metadata_for_artifact_file(path, root)
            metadata = publish_artifact_snapshot(path, root, metadata=metadata)
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
    if found := find_published_artifact_file(root, requested_id):
        return found
    try:
        candidates = sorted(root.rglob("*"))
    except OSError:
        return None
    for path in candidates:
        if _is_published_path(path, root):
            continue
        try:
            metadata = metadata_for_artifact_file(path, root)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if path.stem == requested_id or metadata["artifactId"] == requested_id:
            return path.resolve(), metadata
    return None


def publish_artifact_snapshot(path: Path, artifact_root: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    root = artifact_root.resolve()
    file_path = path.resolve()
    file_path.relative_to(root)
    if _is_published_path(file_path, root):
        raise ValueError("published artifact snapshots are not source artifacts")
    metadata = dict(metadata or metadata_for_artifact_file(file_path, root))
    artifact_id = str(metadata.get("artifactId") or "").strip()
    if not artifact_id:
        raise ValueError("artifactId is required")
    snapshot_dir = _published_root(root) / safe_artifact_segment(artifact_id, fallback="artifact")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / file_path.name
    tmp_path = snapshot_dir / f".{file_path.name}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(file_path, tmp_path)
        os.replace(tmp_path, snapshot_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return metadata


def find_published_artifact_file(
    artifact_root: Path,
    artifact_id: str,
) -> tuple[Path, dict[str, Any]] | None:
    requested_id = str(artifact_id or "").strip()
    if not requested_id or "/" in requested_id or "\\" in requested_id:
        return None
    try:
        root = artifact_root.resolve()
        snapshot_dir = _published_root(root) / safe_artifact_segment(requested_id, fallback="artifact")
    except Exception:
        return None
    if not snapshot_dir.is_dir():
        return None
    try:
        candidates = sorted(path for path in snapshot_dir.iterdir() if path.is_file())
    except OSError:
        return None
    for path in candidates:
        try:
            metadata = metadata_for_artifact_file(path, snapshot_dir, artifact_id=requested_id)
        except (FileNotFoundError, OSError, ValueError):
            continue
        return path.resolve(), metadata
    return None


def _published_root(artifact_root: Path) -> Path:
    return artifact_root / _PUBLISHED_DIR_NAME


def _is_published_path(path: Path, artifact_root: Path) -> bool:
    try:
        path.resolve().relative_to(_published_root(artifact_root.resolve()).resolve())
        return True
    except ValueError:
        return False
