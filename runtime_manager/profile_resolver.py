from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cloud_kubeconfig import CloudKubeconfigResolver
from .home_resolver import UserHomeResolver

_DEFAULT_ENABLED_TOOLSETS = ("terminal", "file")
_DEFAULT_SYSTEM_PROMPT_FILENAMES = ("system-prompt.md", "system_prompt.md")


@dataclass(frozen=True)
class ResolvedRunContext:
    user_id: str
    conversation_id: str
    session_id: str
    user_home: Path
    home_dir: Path
    workspace_dir: Path
    worker_env: dict[str, str]
    cluster_contexts: list[dict[str, str]]
    system_prompt: str | None
    skills: list[str]
    enabled_toolsets: list[str] | None
    disabled_toolsets: list[str] | None
    max_iterations: int


class RuntimeProfileResolver:
    """Resolve per-run Hermes profile inputs before a worker process starts."""

    def __init__(
        self,
        *,
        users_root: str | Path,
        cloud_kubeconfig_resolver: CloudKubeconfigResolver | None = None,
    ) -> None:
        self.home_resolver = UserHomeResolver(users_root)
        self.cloud_kubeconfig_resolver = cloud_kubeconfig_resolver or CloudKubeconfigResolver()
        default_profile_dir = os.getenv("RUNTIME_MANAGER_DEFAULT_PROFILE_DIR")
        self.default_profile_dir = Path(default_profile_dir).expanduser() if default_profile_dir else None
        default_profile_manifest = _load_default_profile_manifest(self.default_profile_dir)
        self.default_enabled_toolsets = _parse_toolset_env(
            os.getenv("RUNTIME_MANAGER_DEFAULT_ENABLED_TOOLSETS"),
            default=list(_DEFAULT_ENABLED_TOOLSETS),
        )
        self.default_max_iterations = int(os.getenv("RUNTIME_MANAGER_DEFAULT_MAX_ITERATIONS", "20"))
        self.default_system_prompt = _load_default_system_prompt(
            os.getenv("RUNTIME_MANAGER_DEFAULT_SYSTEM_PROMPT_FILE")
            or default_profile_manifest.get("systemPrompt"),
            default_profile_dir=self.default_profile_dir,
        )
        self.default_skills = _parse_string_list_env(
            os.getenv("RUNTIME_MANAGER_DEFAULT_SKILLS"),
            default=_extract_manifest_skill_names(default_profile_manifest),
        )

    def validate_user_id(self, user_id: str) -> str:
        return self.home_resolver.validate_user_id(user_id)

    def resolve(self, payload: dict[str, Any]) -> ResolvedRunContext:
        user_id = self.validate_user_id(str(payload.get("user_id") or ""))
        conversation_id = str(payload.get("conversation_id") or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id is required")
        session_id = str(payload.get("session_id") or conversation_id)
        user_home = self.home_resolver.resolve(user_id)
        home_dir = user_home / "home"
        workspace_dir = user_home / "workspace"
        home_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        workspace_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._ensure_default_profile_assets(user_home)
        cluster_contexts = self.cloud_kubeconfig_resolver.prepare_contexts(
            user_home,
            _extract_contexts_payload(payload),
        )
        cluster_context_prompt = self.cloud_kubeconfig_resolver.build_prompt(cluster_contexts)
        enabled_toolsets = _normalize_toolsets(payload.get("enabled_toolsets"))
        if enabled_toolsets is None:
            enabled_toolsets = (
                list(self.default_enabled_toolsets)
                if self.default_enabled_toolsets is not None
                else None
            )
        system_prompt = _join_prompt_parts(
            self.default_system_prompt,
            cluster_context_prompt,
            payload.get("system_prompt"),
        )
        worker_env = os.environ.copy()
        worker_env["PYTHONUNBUFFERED"] = "1"
        worker_env["HERMES_HOME"] = str(user_home)
        worker_env["HOME"] = str(home_dir)
        worker_env["TERMINAL_CWD"] = str(workspace_dir)
        return ResolvedRunContext(
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            user_home=user_home,
            home_dir=home_dir,
            workspace_dir=workspace_dir,
            worker_env=worker_env,
            cluster_contexts=cluster_contexts,
            system_prompt=system_prompt,
            skills=_merge_string_lists(self.default_skills, payload.get("skills")),
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=_normalize_toolsets(payload.get("disabled_toolsets")),
            max_iterations=_first_present(payload.get("max_iterations"), self.default_max_iterations),
        )

    def _ensure_default_profile_assets(self, user_home: Path) -> None:
        if self.default_profile_dir is None:
            return
        skills_root = user_home / "skills"
        skills_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_skills_root = self.default_profile_dir / "skills"
        if not source_skills_root.exists():
            return
        for skill_md in source_skills_root.rglob("SKILL.md"):
            source_skill_dir = skill_md.parent
            if _has_excluded_path_part(source_skill_dir.relative_to(source_skills_root)):
                continue
            destination = skills_root / source_skill_dir.relative_to(source_skills_root)
            _copy_skill_directory(source_skill_dir, destination)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _extract_contexts_payload(payload: dict[str, Any]) -> Any:
    for key in ("contexts", "cluster_contexts", "clusterContexts"):
        if key in payload:
            return payload.get(key)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("contexts", "cluster_contexts", "clusterContexts"):
            if key in metadata:
                return metadata.get(key)
    return None


def _normalize_toolsets(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "all":
            return None
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, list):
        if len(value) == 1 and str(value[0]).strip().lower() == "all":
            return None
        return [str(item).strip() for item in value if str(item).strip()]
    return None


def _parse_toolset_env(value: str | None, *, default: list[str]) -> list[str] | None:
    if value is None:
        return default
    return _normalize_toolsets(value)


def _parse_string_list_env(value: str | None, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    return _normalize_string_list(value)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _merge_string_lists(defaults: list[str], value: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*defaults, *_normalize_string_list(value)]:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _load_default_system_prompt(path_value: Any, *, default_profile_dir: Path | None) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        prompt_path = _find_default_system_prompt(default_profile_dir) if default_profile_dir is not None else None
        if prompt_path is None:
            return ""
    else:
        prompt_path = Path(path_value).expanduser()
        if not prompt_path.is_absolute() and default_profile_dir is not None:
            prompt_path = default_profile_dir / prompt_path
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _find_default_system_prompt(default_profile_dir: Path) -> Path | None:
    for filename in _DEFAULT_SYSTEM_PROMPT_FILENAMES:
        candidate = default_profile_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _load_default_profile_manifest(default_profile_dir: Path | None) -> dict[str, Any]:
    if default_profile_dir is None:
        return {}
    manifest_path = default_profile_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return {}
    try:
        import yaml

        with manifest_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_manifest_skill_names(manifest: dict[str, Any]) -> list[str]:
    skills = manifest.get("skills")
    if not isinstance(skills, list):
        return []
    names: list[str] = []
    for item in skills:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            if item.get("enabled") is False:
                continue
            raw_name = item.get("name")
            raw_path = item.get("path")
            name = str(raw_name or Path(str(raw_path or "")).name).strip()
        else:
            continue
        if name and name not in names:
            names.append(name)
    return names


def _join_prompt_parts(*parts: Any) -> str | None:
    rendered = [str(part).strip() for part in parts if isinstance(part, str) and str(part).strip()]
    if not rendered:
        return None
    return "\n\n".join(rendered)


_SKILL_COPY_IGNORE_NAMES = {
    ".DS_Store",
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}


def _has_excluded_path_part(path: Path) -> bool:
    return any(part in _SKILL_COPY_IGNORE_NAMES for part in path.parts)


def _copy_skill_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*_SKILL_COPY_IGNORE_NAMES),
    )
    _chmod_tree(destination)


def _chmod_tree(root: Path) -> None:
    root.chmod(0o700)
    for item in root.rglob("*"):
        if item.is_dir():
            item.chmod(0o700)
        else:
            item.chmod(0o600)
