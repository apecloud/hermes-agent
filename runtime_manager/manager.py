from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .cloud_kubeconfig import CloudKubeconfigResolver
from .profile_resolver import RuntimeProfileResolver
from .registry import RunHandle, RunRegistry

logger = logging.getLogger(__name__)

class RuntimeManager:
    def __init__(
        self,
        *,
        users_root: str | Path,
        api_key: str = "",
        python_executable: str = sys.executable,
        worker_script: str | Path | None = None,
        cloud_kubeconfig_resolver: CloudKubeconfigResolver | None = None,
    ) -> None:
        self.api_key = api_key or ""
        max_events_per_run = int(os.getenv("RUNTIME_MANAGER_MAX_EVENTS_PER_RUN", "1000"))
        completed_run_ttl_seconds = float(os.getenv("RUNTIME_MANAGER_COMPLETED_RUN_TTL_SECONDS", "3600"))
        self.registry = RunRegistry(
            max_events_per_run=max_events_per_run,
            completed_run_ttl_seconds=completed_run_ttl_seconds,
        )
        self.profile_resolver = RuntimeProfileResolver(
            users_root=users_root,
            cloud_kubeconfig_resolver=cloud_kubeconfig_resolver,
        )
        self.resolver = self.profile_resolver.home_resolver
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
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message is required")
        resolved = self.profile_resolver.resolve(payload)
        user_id = resolved.user_id
        conversation_id = resolved.conversation_id
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
        await self._reserve_run(run_id=run_id, user_id=user_id, conversation_id=conversation_id)
        handle = self.registry.create(
            run_id=run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=resolved.session_id,
            model=model,
        )
        handle.publish({
            "event": "run.queued",
            "run_id": run_id,
            "timestamp": time.time(),
        })

        try:
            proc_env = resolved.worker_env.copy()
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
            "session_id": resolved.session_id,
            "hermes_home": str(resolved.user_home),
            "message": message,
            "history": payload.get("history") or [],
            "model": model,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "llm_config": llm_config,
            "system_prompt": resolved.system_prompt,
            "skills": resolved.skills,
            "cluster_contexts": resolved.cluster_contexts,
            "enabled_toolsets": resolved.enabled_toolsets,
            "disabled_toolsets": resolved.disabled_toolsets,
            "skip_memory": bool(payload.get("skip_memory", False)),
            "skip_context_files": bool(payload.get("skip_context_files", True)),
            "max_iterations": resolved.max_iterations,
            "metadata": payload.get("metadata") or {},
        }
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(worker_request, ensure_ascii=False) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        self._track(asyncio.create_task(self._pump_stdout(handle)))
        self._track(asyncio.create_task(self._pump_stderr(handle)))
        self._track(asyncio.create_task(self._watch_process(handle)))
        return handle

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
        try:
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
        except ValueError as exc:
            await self._fail_worker_stream(handle, f"worker stdout stream failed: {exc}")
        except Exception as exc:
            logger.exception("runtime-manager: stdout pump failed for %s", handle.run_id)
            await self._fail_worker_stream(handle, f"worker stdout stream failed: {type(exc).__name__}: {exc}")

    async def _fail_worker_stream(self, handle: RunHandle, message: str) -> None:
        if handle.status in {"completed", "failed", "cancelled"}:
            return
        logger.error("runtime-manager: %s for %s", message, handle.run_id)
        handle.publish(
            {
                "event": "run.failed",
                "run_id": handle.run_id,
                "timestamp": time.time(),
                "error": message,
            }
        )
        proc = handle.process
        if proc is not None and getattr(proc, "returncode", None) is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

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
