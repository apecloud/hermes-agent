"""Business-level agent profile resolver for the API server.

This is separate from Hermes CLI profiles.  A single user's HERMES_HOME stays
fixed, while an API request can bind a whitelisted runtime profile that chooses
the system prompt, visible skill roots, and subprocess environment filter for a
conversation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home


DEFAULT_AGENT_PROFILE = "cluster-diagnosis"
DEFAULT_ALLOWED_AGENT_PROFILES = frozenset({"global-entry", "cluster-diagnosis"})
AGENT_PROFILE_MODEL_CONFIG_KEY = "agent_profile"
AGENT_PROFILE_METADATA_KEY = "agent_profile_metadata"

_GLOBAL_ENTRY_ENV_BLOCKLIST = frozenset(
    {
        "KUBECONFIG",
        "KUBE_CONFIG",
        "HERMES_KUBECONFIG",
        "KB_CLUSTER_KUBECONFIG",
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT",
    }
)


class AgentProfileError(ValueError):
    """Client-safe profile resolution failure."""

    def __init__(self, message: str, *, code: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class AgentProfileSettings:
    root_dir: Path | None
    allowed: frozenset[str]
    default: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class AgentProfileSkill:
    name: str
    rel_path: str
    directory: Path
    skill_md: Path


@dataclass(frozen=True)
class ResolvedAgentProfile:
    name: str
    version: str
    profile_dir: Path
    manifest_path: Path
    system_prompt_path: Path
    system_prompt: str
    skills: tuple[AgentProfileSkill, ...]
    profile_hash: str
    prompt: str
    env_blocklist: frozenset[str]

    @property
    def skill_dirs(self) -> tuple[Path, ...]:
        return tuple(skill.directory for skill in self.skills)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "profile_hash": self.profile_hash,
            "manifest": _relative_posix(self.manifest_path, self.profile_dir),
            "system_prompt": _relative_posix(self.system_prompt_path, self.profile_dir),
            "selected_skills": [
                {"name": skill.name, "path": skill.rel_path}
                for skill in self.skills
            ],
        }

    def model_config_patch(self) -> dict[str, Any]:
        return {
            AGENT_PROFILE_MODEL_CONFIG_KEY: self.name,
            AGENT_PROFILE_METADATA_KEY: self.metadata(),
        }


def extract_requested_agent_profile(body: Mapping[str, Any]) -> str | None:
    """Return requested ``agent_profile``/``agentProfile`` or ``None``."""

    missing = object()
    snake = body.get("agent_profile", missing)
    camel = body.get("agentProfile", missing)
    if snake is not missing and camel is not missing and snake != camel:
        raise AgentProfileError(
            "agent_profile and agentProfile disagree",
            code="invalid_agent_profile",
        )
    raw = snake if snake is not missing else camel
    if raw is missing or raw is None:
        return None
    if not isinstance(raw, str):
        raise AgentProfileError(
            "agent_profile must be a string",
            code="invalid_agent_profile",
        )
    value = raw.strip()
    if not value:
        raise AgentProfileError(
            "agent_profile must not be empty",
            code="invalid_agent_profile",
        )
    return value


def settings_from_mapping(mapping: Mapping[str, Any] | None) -> AgentProfileSettings:
    raw = mapping if isinstance(mapping, Mapping) else {}

    allowed_raw = (
        raw.get("allowed")
        or raw.get("allowlist")
        or raw.get("whitelist")
        or DEFAULT_ALLOWED_AGENT_PROFILES
    )
    allowed = _normalize_allowed(allowed_raw)
    default = str(raw.get("default") or raw.get("default_profile") or DEFAULT_AGENT_PROFILE).strip()
    if default not in allowed:
        allowed = frozenset((*allowed, default))

    root_raw = raw.get("root_dir") or raw.get("root") or raw.get("profiles_dir") or raw.get("profilesRoot")
    root_dir = _normalize_root(root_raw) if root_raw else None
    return AgentProfileSettings(root_dir=root_dir, allowed=allowed, default=default, raw=raw)


def parse_model_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def stored_agent_profile_name(session: Mapping[str, Any] | None) -> str | None:
    if not session:
        return None
    cfg = parse_model_config(session.get("model_config"))
    value = cfg.get(AGENT_PROFILE_MODEL_CONFIG_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    metadata = cfg.get(AGENT_PROFILE_METADATA_KEY)
    if isinstance(metadata, Mapping):
        value = metadata.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def choose_agent_profile_name(
    requested: str | None,
    stored: str | None,
    settings: AgentProfileSettings,
) -> str:
    """Choose a profile and enforce per-session profile immutability."""

    if requested and requested not in settings.allowed:
        raise AgentProfileError(
            f"Unsupported agent_profile: {requested}",
            code="invalid_agent_profile",
        )
    if stored and stored not in settings.allowed:
        raise AgentProfileError(
            f"Stored agent_profile is no longer allowed: {stored}",
            code="invalid_agent_profile",
            status=409,
        )
    if stored and requested and requested != stored:
        raise AgentProfileError(
            f"Session is already bound to agent_profile '{stored}'",
            code="agent_profile_locked",
            status=409,
        )
    return stored or requested or settings.default


def resolve_agent_profile(
    profile_name: str,
    settings: AgentProfileSettings,
) -> ResolvedAgentProfile:
    if profile_name not in settings.allowed:
        raise AgentProfileError(
            f"Unsupported agent_profile: {profile_name}",
            code="invalid_agent_profile",
        )
    if settings.root_dir is None:
        raise AgentProfileError(
            "agent_profiles.root_dir is not configured",
            code="agent_profile_not_configured",
            status=400,
        )

    root_dir = settings.root_dir.resolve()
    profile_dir = (root_dir / profile_name).resolve()
    if not _is_relative_to(profile_dir, root_dir) or not profile_dir.is_dir():
        raise AgentProfileError(
            f"agent_profile is not configured: {profile_name}",
            code="agent_profile_not_configured",
            status=400,
        )

    manifest_path = profile_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise AgentProfileError(
            f"agent_profile manifest is missing: {profile_name}",
            code="agent_profile_manifest_missing",
            status=500,
        )

    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_manifest(manifest_bytes.decode("utf-8"), profile_name)
    version = str(manifest.get("version") or "v1")

    system_prompt_rel = str(
        manifest.get("systemPrompt")
        or manifest.get("system_prompt")
        or "system-prompt.md"
    )
    system_prompt_path = _resolve_manifest_path(
        profile_dir,
        system_prompt_rel,
        field="systemPrompt",
    )
    if not system_prompt_path.is_file():
        raise AgentProfileError(
            f"agent_profile system prompt is missing: {system_prompt_rel}",
            code="agent_profile_system_prompt_missing",
            status=500,
        )
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    skills = _load_manifest_skills(profile_dir, manifest)
    digest = hashlib.sha256()
    digest.update(b"profile-name\0")
    digest.update(profile_name.encode("utf-8"))
    digest.update(b"\nmanifest\0")
    digest.update(manifest_bytes)
    digest.update(b"\nsystem-prompt\0")
    digest.update(system_prompt_path.read_bytes())
    for skill in skills:
        digest.update(b"\nskill\0")
        digest.update(skill.rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(skill.skill_md.read_bytes())
    profile_hash = digest.hexdigest()

    prompt = _build_profile_prompt(
        profile_name=profile_name,
        version=version,
        profile_hash=profile_hash,
        system_prompt=system_prompt,
        skills=skills,
    )
    return ResolvedAgentProfile(
        name=profile_name,
        version=version,
        profile_dir=profile_dir,
        manifest_path=manifest_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        skills=tuple(skills),
        profile_hash=profile_hash,
        prompt=prompt,
        env_blocklist=_env_blocklist_for_profile(profile_name, settings),
    )


def merge_model_config_patch(existing: Any, patch: Mapping[str, Any]) -> str:
    cfg = parse_model_config(existing)
    cfg.update(dict(patch))
    return json.dumps(cfg, ensure_ascii=False, sort_keys=True)


def _normalize_allowed(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = value
    else:
        raw_items = DEFAULT_ALLOWED_AGENT_PROFILES
    items = {str(item).strip() for item in raw_items if str(item).strip()}
    if not items:
        items = set(DEFAULT_ALLOWED_AGENT_PROFILES)
    return frozenset(items)


def _normalize_root(value: Any) -> Path:
    raw = str(value).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = get_hermes_home() / path
    return path


def _load_manifest(content: str, profile_name: str) -> dict[str, Any]:
    try:
        from agent.skill_utils import yaml_load

        parsed = yaml_load(content) or {}
    except Exception as exc:
        raise AgentProfileError(
            f"agent_profile manifest is invalid: {profile_name}",
            code="agent_profile_manifest_invalid",
            status=500,
        ) from exc
    if not isinstance(parsed, dict):
        raise AgentProfileError(
            f"agent_profile manifest must be a mapping: {profile_name}",
            code="agent_profile_manifest_invalid",
            status=500,
        )
    return parsed


def _load_manifest_skills(profile_dir: Path, manifest: Mapping[str, Any]) -> list[AgentProfileSkill]:
    raw_skills = manifest.get("skills") or []
    if not isinstance(raw_skills, list):
        raise AgentProfileError(
            "agent_profile manifest skills must be a list",
            code="agent_profile_manifest_invalid",
            status=500,
        )

    skills: list[AgentProfileSkill] = []
    seen_dirs: set[Path] = set()
    for idx, raw in enumerate(raw_skills):
        if isinstance(raw, str):
            item = {"path": raw}
        elif isinstance(raw, Mapping):
            item = raw
        else:
            raise AgentProfileError(
                f"agent_profile manifest skills[{idx}] must be a mapping or string",
                code="agent_profile_manifest_invalid",
                status=500,
            )
        if item.get("enabled", True) is False:
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AgentProfileError(
                f"agent_profile manifest skills[{idx}].path is required",
                code="agent_profile_manifest_invalid",
                status=500,
            )
        skill_dir = _resolve_manifest_path(profile_dir, raw_path, field=f"skills[{idx}].path")
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise AgentProfileError(
                f"agent_profile skill is missing SKILL.md: {raw_path}",
                code="agent_profile_skill_missing",
                status=500,
            )
        if skill_dir in seen_dirs:
            continue
        seen_dirs.add(skill_dir)
        name = str(item.get("name") or skill_dir.name).strip() or skill_dir.name
        skills.append(
            AgentProfileSkill(
                name=name,
                rel_path=_relative_posix(skill_dir, profile_dir),
                directory=skill_dir,
                skill_md=skill_md,
            )
        )
    return skills


def _resolve_manifest_path(profile_dir: Path, raw_path: str, *, field: str) -> Path:
    rel = Path(raw_path)
    if rel.is_absolute():
        raise AgentProfileError(
            f"agent_profile manifest {field} must be relative",
            code="agent_profile_manifest_invalid",
            status=500,
        )
    resolved = (profile_dir / rel).resolve()
    if not _is_relative_to(resolved, profile_dir):
        raise AgentProfileError(
            f"agent_profile manifest {field} escapes the profile directory",
            code="agent_profile_manifest_invalid",
            status=500,
        )
    return resolved


def _build_profile_prompt(
    *,
    profile_name: str,
    version: str,
    profile_hash: str,
    system_prompt: str,
    skills: list[AgentProfileSkill],
) -> str:
    skill_lines = [
        f"- {skill.name} ({skill.rel_path})"
        for skill in skills
    ]
    parts = [
        "<agent_profile>",
        f"name: {profile_name}",
        f"version: {version}",
        f"profile_hash: {profile_hash}",
        "selected_skills:",
        *(skill_lines or ["- none"]),
        "</agent_profile>",
        "<profile_system_prompt>",
        system_prompt,
        "</profile_system_prompt>",
    ]
    if skills:
        parts.append("<profile_skill_instructions>")
        for skill in skills:
            skill_text = skill.skill_md.read_text(encoding="utf-8").strip()
            parts.extend(
                [
                    f"## {skill.name}",
                    f"Source: {skill.rel_path}/SKILL.md",
                    skill_text,
                ]
            )
        parts.append("</profile_skill_instructions>")
    return "\n".join(part for part in parts if part is not None)


def _env_blocklist_for_profile(
    profile_name: str,
    settings: AgentProfileSettings,
) -> frozenset[str]:
    env_names: set[str] = set()
    if profile_name == "global-entry":
        env_names.update(_GLOBAL_ENTRY_ENV_BLOCKLIST)

    profiles_cfg = settings.raw.get("profiles")
    if isinstance(profiles_cfg, Mapping):
        profile_cfg = profiles_cfg.get(profile_name)
        if isinstance(profile_cfg, Mapping):
            env_names.update(_normalize_env_names(profile_cfg.get("env_blocklist")))
    env_names.update(_normalize_env_names(settings.raw.get("env_blocklist")))
    return frozenset(env_names)


def _normalize_env_names(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = value
    else:
        return set()
    return {str(item).strip() for item in raw_items if str(item).strip()}


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
