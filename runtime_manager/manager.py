from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .home_resolver import UserHomeResolver
from .registry import RunHandle, RunRegistry

logger = logging.getLogger(__name__)

_DEFAULT_ENABLED_TOOLSETS = ("terminal", "file")
_DEFAULT_SYSTEM_PROMPT_FILENAMES = ("system-prompt.md", "system_prompt.md")


class RuntimeManager:
    def __init__(
        self,
        *,
        users_root: str | Path,
        api_key: str = "",
        python_executable: str = sys.executable,
        worker_script: str | Path | None = None,
    ) -> None:
        self.api_key = api_key or ""
        max_events_per_run = int(os.getenv("RUNTIME_MANAGER_MAX_EVENTS_PER_RUN", "1000"))
        completed_run_ttl_seconds = float(os.getenv("RUNTIME_MANAGER_COMPLETED_RUN_TTL_SECONDS", "3600"))
        self.resolver = UserHomeResolver(users_root)
        self.registry = RunRegistry(
            max_events_per_run=max_events_per_run,
            completed_run_ttl_seconds=completed_run_ttl_seconds,
        )
        self.python_executable = python_executable
        self.worker_script = Path(worker_script or (Path(__file__).resolve().parent / "worker_main.py")).resolve()
        self._tasks: set[asyncio.Task] = set()
        self._limit_lock = asyncio.Lock()
        self._active_conversations: set[tuple[str, str]] = set()
        self._active_run_ids: set[str] = set()
        self._stopping_run_ids: set[str] = set()
        self._active_runs_by_user: dict[str, int] = {}
        self.max_active_runs = int(os.getenv("RUNTIME_MANAGER_MAX_ACTIVE_RUNS", "50"))
        self.max_active_runs_per_user = int(os.getenv("RUNTIME_MANAGER_MAX_ACTIVE_RUNS_PER_USER", "2"))
        self.stop_grace_seconds = float(os.getenv("RUNTIME_MANAGER_STOP_GRACE_SECONDS", "10"))
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
        logger.info(
            "runtime-manager default profile dir=%s skills=%s",
            str(self.default_profile_dir or ""),
            ",".join(self.default_skills),
        )
        insecure_allow = os.getenv(
            "RUNTIME_MANAGER_INSECURE_ALLOW_UNAUTHENTICATED",
            os.getenv("RUNTIME_MANAGER_ALLOW_UNAUTHENTICATED", ""),
        )
        self.allow_unauthenticated = insecure_allow.lower() in {
            "1",
            "true",
            "yes",
        }

    def authorize(self, authorization: str | None) -> bool:
        if not self.api_key:
            return self.allow_unauthenticated
        header = (authorization or "").strip()
        return header == f"Bearer {self.api_key}"

    async def start_run(self, payload: dict[str, Any]) -> RunHandle:
        user_id = self.resolver.validate_user_id(str(payload.get("user_id") or ""))
        conversation_id = str(payload.get("conversation_id") or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id is required")
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message is required")
        llm_config = payload.get("llm_config")
        if not isinstance(llm_config, dict):
            llm_config = {}
        model = str(
            payload.get("model")
            or llm_config.get("model")
            or llm_config.get("name")
            or llm_config.get("default")
            or ""
        ).strip()
        provider = _first_present(payload.get("provider"), llm_config.get("provider"))
        api_key = _first_present(payload.get("api_key"), llm_config.get("api_key"), llm_config.get("apiKey"))
        base_url = _first_present(
            payload.get("base_url"),
            payload.get("baseURL"),
            llm_config.get("base_url"),
            llm_config.get("baseURL"),
        )
        run_id = f"run_{uuid.uuid4().hex}"
        session_id = str(payload.get("session_id") or conversation_id)
        user_home = self.resolver.resolve(user_id)
        self._ensure_default_profile_assets(user_home)
        enabled_toolsets = _normalize_toolsets(payload.get("enabled_toolsets"))
        if enabled_toolsets is None:
            enabled_toolsets = (
                list(self.default_enabled_toolsets)
                if self.default_enabled_toolsets is not None
                else None
            )
        disabled_toolsets = _normalize_toolsets(payload.get("disabled_toolsets"))
        system_prompt = _join_prompt_parts(
            self.default_system_prompt,
            payload.get("system_prompt"),
        )
        skills = _merge_string_lists(self.default_skills, payload.get("skills"))
        await self._reserve_run(run_id=run_id, user_id=user_id, conversation_id=conversation_id)
        handle = self.registry.create(
            run_id=run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            model=model,
        )
        handle.publish({
            "event": "run.queued",
            "run_id": run_id,
            "timestamp": time.time(),
        })

        try:
            proc_env = os.environ.copy()
            proc_env["PYTHONUNBUFFERED"] = "1"
            proc_env["HERMES_HOME"] = str(user_home)
            proc_env["HOME"] = str(user_home / "home")
            proc = await asyncio.create_subprocess_exec(
                self.python_executable,
                str(self.worker_script),
                cwd=str(Path(__file__).resolve().parent.parent),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
            )
        except Exception:
            await self._release_run(handle)
            raise
        handle.process = proc
        handle.stdin = proc.stdin

        worker_request = {
            "run_id": run_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "hermes_home": str(user_home),
            "message": message,
            "history": payload.get("history") or [],
            "model": model,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "llm_config": llm_config,
            "system_prompt": system_prompt,
            "skills": skills,
            "enabled_toolsets": enabled_toolsets,
            "disabled_toolsets": disabled_toolsets,
            "skip_memory": bool(payload.get("skip_memory", False)),
            "skip_context_files": bool(payload.get("skip_context_files", True)),
            "max_iterations": _first_present(payload.get("max_iterations"), self.default_max_iterations),
            "metadata": payload.get("metadata") or {},
        }
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(worker_request, ensure_ascii=False) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        self._track(asyncio.create_task(self._pump_stdout(handle)))
        self._track(asyncio.create_task(self._pump_stderr(handle)))
        self._track(asyncio.create_task(self._watch_process(handle)))
        return handle

    def _ensure_default_profile_assets(self, user_home: Path) -> None:
        if self.default_profile_dir is None:
            return
        skills_root = user_home / "skills"
        skills_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_skills_root = self.default_profile_dir / "skills"
        if not source_skills_root.exists():
            return
        for skill_dir in source_skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            source = skill_dir / "SKILL.md"
            if not source.is_file():
                continue
            destination = skills_root / skill_dir.name / "SKILL.md"
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)

    async def _reserve_run(self, *, run_id: str, user_id: str, conversation_id: str) -> None:
        key = (user_id, conversation_id)
        async with self._limit_lock:
            if key in self._active_conversations:
                raise RuntimeError("conversation already has an active run")
            if len(self._active_run_ids) >= self.max_active_runs:
                raise RuntimeError("runtime manager is busy")
            active_for_user = self._active_runs_by_user.get(user_id, 0)
            if active_for_user >= self.max_active_runs_per_user:
                raise RuntimeError("user has too many active runs")
            self._active_conversations.add(key)
            self._active_run_ids.add(run_id)
            self._active_runs_by_user[user_id] = active_for_user + 1

    async def _release_run(self, handle: RunHandle) -> None:
        key = (handle.user_id, handle.conversation_id)
        async with self._limit_lock:
            if handle.run_id not in self._active_run_ids:
                return
            self._active_run_ids.discard(handle.run_id)
            self._active_conversations.discard(key)
            active_for_user = self._active_runs_by_user.get(handle.user_id, 0)
            if active_for_user <= 1:
                self._active_runs_by_user.pop(handle.user_id, None)
            else:
                self._active_runs_by_user[handle.user_id] = active_for_user - 1

    def _track(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send_command(self, run_id: str, command: dict[str, Any]) -> RunHandle:
        handle = self.registry.require(run_id)
        stdin = handle.stdin
        if stdin is None or getattr(stdin, "is_closing", lambda: True)():
            raise RuntimeError(f"run {run_id} is not accepting commands")
        stdin.write((json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
        await stdin.drain()
        return handle

    async def approve_run(self, run_id: str, *, choice: str, resolve_all: bool = False) -> RunHandle:
        normalized = str(choice or "").strip().lower()
        if normalized not in {"once", "session", "always", "deny"}:
            raise ValueError("invalid approval choice; expected one of: once, session, always, deny")
        handle = self.registry.require(run_id)
        if handle.status != "waiting_for_approval":
            raise RuntimeError(f"run {run_id} is not waiting for approval")
        await self.send_command(
            run_id,
            {"type": "approval", "choice": normalized, "resolve_all": resolve_all},
        )
        handle.publish(
            {
                "event": "approval.responded",
                "run_id": run_id,
                "timestamp": time.time(),
                "choice": normalized,
            }
        )
        return handle

    async def stop_run(self, run_id: str) -> RunHandle:
        handle = self.registry.require(run_id)
        if handle.status in {"completed", "failed", "cancelled"}:
            return handle
        self._stopping_run_ids.add(run_id)
        handle.publish(
            {
                "event": "run.cancelling",
                "run_id": run_id,
                "timestamp": time.time(),
            }
        )
        try:
            await self.send_command(run_id, {"type": "stop"})
        except RuntimeError:
            proc = handle.process
            if proc is not None and getattr(proc, "returncode", None) is None:
                proc.terminate()
            raise
        self._track(asyncio.create_task(self._force_kill_if_needed(handle)))
        return handle

    async def _pump_stdout(self, handle: RunHandle) -> None:
        proc = handle.process
        assert proc is not None and proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("runtime-manager: invalid worker json for %s: %s", handle.run_id, raw)
                continue
            handle.publish(event)

    async def _pump_stderr(self, handle: RunHandle) -> None:
        proc = handle.process
        assert proc is not None and proc.stderr is not None
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            logger.warning("runtime-manager worker[%s]: %s", handle.run_id, line.decode("utf-8", errors="replace").rstrip())

    async def _watch_process(self, handle: RunHandle) -> None:
        proc = handle.process
        assert proc is not None
        rc = await proc.wait()
        await self._release_run(handle)
        if handle.stdin is not None:
            try:
                handle.stdin.close()
            except Exception:
                pass
        if handle.status in {"completed", "failed", "cancelled"}:
            return
        if handle.status == "cancelling" or handle.run_id in self._stopping_run_ids:
            self._stopping_run_ids.discard(handle.run_id)
            handle.publish({
                "event": "run.cancelled",
                "run_id": handle.run_id,
                "timestamp": time.time(),
            })
            return
        if rc == 0:
            handle.publish({
                "event": "run.completed",
                "run_id": handle.run_id,
                "timestamp": time.time(),
                "output": handle.output or "",
                "usage": handle.usage or {},
            })
            return
        handle.publish({
            "event": "run.failed",
            "run_id": handle.run_id,
            "timestamp": time.time(),
            "error": f"worker exited with code {rc}",
        })

    async def _force_kill_if_needed(self, handle: RunHandle) -> None:
        await asyncio.sleep(self.stop_grace_seconds)
        if handle.status in {"completed", "failed", "cancelled"}:
            return
        proc = handle.process
        if proc is None or proc.returncode is not None:
            return
        logger.warning("runtime-manager: force killing worker for %s after stop grace period", handle.run_id)
        try:
            proc.kill()
        except ProcessLookupError:
            return


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
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
        logger.warning("runtime-manager: failed to load default profile manifest %s", manifest_path, exc_info=True)
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
    rendered = [str(part).strip() for part in parts if isinstance(part, str) and part.strip()]
    if not rendered:
        return None
    return "\n\n".join(rendered)
