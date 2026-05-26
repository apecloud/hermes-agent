from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

_OUTPUT_LOCK = threading.Lock()
_AGENT_HOLDER: dict[str, Any] = {"agent": None}


def emit(event: dict[str, Any]) -> None:
    with _OUTPUT_LOCK:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> int:
    first_line = sys.stdin.readline()
    if not first_line:
        emit(
            {
                "event": "run.failed",
                "run_id": "unknown",
                "timestamp": time.time(),
                "error": "missing worker request",
            }
        )
        return 1

    request = json.loads(first_line)
    run_id = str(request.get("run_id") or "run_unknown")
    user_id = str(request.get("user_id") or "")
    session_id = str(request.get("session_id") or request.get("conversation_id") or run_id)
    approval_session_key = session_id

    hermes_home = str(request["hermes_home"])
    os.environ["HERMES_HOME"] = hermes_home
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    env_path = Path(hermes_home) / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path, override=True)
        os.environ["HERMES_HOME"] = hermes_home

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.approval import (
        register_gateway_notify,
        reset_current_session_key,
        resolve_gateway_approval,
        set_current_session_key,
        unregister_gateway_notify,
    )

    stop_requested = threading.Event()

    def on_delta(delta: str | None) -> None:
        if delta is None:
            return
        emit(
            {
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": time.time(),
                "delta": delta,
            }
        )

    def on_tool_progress(
        event_type: str,
        tool_name: str | None = None,
        preview: str | None = None,
        *unused_args,
        **kwargs,
    ) -> None:
        _ = unused_args
        ts = time.time()
        if event_type == "tool.started":
            emit(
                {
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "preview": preview,
                }
            )
        elif event_type == "tool.completed":
            emit(
                {
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                }
            )
        elif event_type == "reasoning.available":
            emit(
                {
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                }
            )

    def on_status(kind: str, message: str) -> None:
        emit(
            {
                "event": "status.message",
                "run_id": run_id,
                "timestamp": time.time(),
                "kind": kind,
                "message": message,
            }
        )

    def approval_notify(data: dict[str, Any]) -> None:
        event = dict(data or {})
        event.update(
            {
                "event": "approval.request",
                "run_id": run_id,
                "timestamp": time.time(),
                "choices": ["once", "session", "always", "deny"],
            }
        )
        emit(event)

    def command_reader() -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            raw = line.strip()
            if not raw:
                continue
            try:
                command = json.loads(raw)
            except json.JSONDecodeError:
                emit(
                    {
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": f"invalid command json: {raw!r}",
                    }
                )
                continue

            kind = command.get("type")
            if kind == "approval":
                try:
                    resolve_gateway_approval(
                        approval_session_key,
                        str(command.get("choice") or "").strip().lower(),
                        resolve_all=bool(command.get("resolve_all", False)),
                    )
                except Exception as exc:
                    emit(
                        {
                            "event": "status.message",
                            "run_id": run_id,
                            "timestamp": time.time(),
                            "kind": "warn",
                            "message": f"approval resolution failed: {exc}",
                        }
                    )
            elif kind == "stop":
                stop_requested.set()
                agent = _AGENT_HOLDER.get("agent")
                if agent is not None:
                    try:
                        agent.interrupt("Stopped by runtime manager")
                    except Exception:
                        pass
                try:
                    resolve_gateway_approval(approval_session_key, "deny", resolve_all=True)
                except Exception:
                    pass
            else:
                emit(
                    {
                        "event": "status.message",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "kind": "warn",
                        "message": f"unknown command type: {kind!r}",
                    }
                )

    listener = threading.Thread(
        target=command_reader,
        name=f"runtime-manager-listener-{run_id}",
        daemon=True,
    )
    listener.start()

    approval_token = None
    session_tokens = []
    try:
        approval_token = set_current_session_key(approval_session_key)
        session_tokens = set_session_vars(
            platform="runtime_manager",
            user_id=user_id,
            session_key=approval_session_key,
        )
        register_gateway_notify(approval_session_key, approval_notify)

        agent = AIAgent(
            model=str(request.get("model") or ""),
            provider=request.get("provider"),
            api_key=request.get("api_key"),
            base_url=request.get("base_url"),
            session_id=session_id,
            session_db=SessionDB(),
            quiet_mode=True,
            verbose_logging=False,
            platform="runtime_manager",
            stream_delta_callback=on_delta,
            tool_progress_callback=on_tool_progress,
            status_callback=on_status,
            enabled_toolsets=request.get("enabled_toolsets"),
            disabled_toolsets=request.get("disabled_toolsets"),
            skip_memory=bool(request.get("skip_memory", False)),
            skip_context_files=bool(request.get("skip_context_files", True)),
            ephemeral_system_prompt=request.get("system_prompt") or None,
            max_iterations=int(request.get("max_iterations") or 90),
        )
        _AGENT_HOLDER["agent"] = agent

        emit({"event": "run.running", "run_id": run_id, "timestamp": time.time()})
        result = agent.run_conversation(
            user_message=request["message"],
            conversation_history=request.get("history") or [],
            task_id=session_id,
        )
        usage = {
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
        }
        if stop_requested.is_set() and not (result.get("final_response") or ""):
            emit({"event": "run.cancelled", "run_id": run_id, "timestamp": time.time()})
            return 0
        if isinstance(result, dict) and result.get("failed"):
            emit(
                {
                    "event": "run.failed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "error": result.get("error") or "agent run failed",
                }
            )
            return 1

        emit(
            {
                "event": "run.completed",
                "run_id": run_id,
                "timestamp": time.time(),
                "output": (result.get("final_response") if isinstance(result, dict) else "") or "",
                "usage": usage,
                "session_id": result.get("session_id", session_id)
                if isinstance(result, dict)
                else session_id,
                "partial": bool(result.get("partial", False)) if isinstance(result, dict) else False,
                "completed": bool(result.get("completed", True)) if isinstance(result, dict) else True,
            }
        )
        return 0
    except Exception as exc:
        emit(
            {
                "event": "run.failed",
                "run_id": run_id,
                "timestamp": time.time(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        try:
            unregister_gateway_notify(approval_session_key)
        except Exception:
            pass
        if approval_token is not None:
            try:
                reset_current_session_key(approval_token)
            except Exception:
                pass
        if session_tokens:
            try:
                clear_session_vars(session_tokens)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
