from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_OUTPUT_LOCK = threading.Lock()
_AGENT_HOLDER: dict[str, Any] = {"agent": None}
HERMES_RUNTIME_PLATFORM = "api_server"
_LOOSE_COMMAND_ARG_PATTERN = re.compile(
    r"""(?is)(?:^|[{,\s])['"]?(?:command|cmd|shell_command)['"]?\s*:\s*"""
    r"""(?P<value>"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|[^,}]+)"""
)
_ENV_ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_KUBECONFIG_ASSIGNMENT_PATTERN = re.compile(r"(?i)(^|\s)(KUBECONFIG=)(?:\"[^\"]*\"|'[^']*'|\S+)")
_KUBECONFIG_FLAG_EQUAL_PATTERN = re.compile(r"(?i)\s*--kubeconfig=(?:\"[^\"]*\"|'[^']*'|\S+)")
_KUBECONFIG_FLAG_VALUE_PATTERN = re.compile(r"(?i)\s*--kubeconfig\s+(?:\"[^\"]*\"|'[^']*'|\S+)")
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(^|\s)([A-Z_]*(?:TOKEN|PASSWORD|SECRET|KEY)[A-Z_]*=)(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_SENSITIVE_FLAG_EQUAL_PATTERN = re.compile(
    r"(?i)(--(?:token|password|secret|key)=)(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_SENSITIVE_FLAG_VALUE_PATTERN = re.compile(
    r"(?i)(--(?:token|password|secret|key))\s+(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_SENSITIVE_STATUS_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|password|secret)(\s*[:=]\s*)([^\s,;.]+)"
)
_BEARER_STATUS_VALUE_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_INTERNAL_USER_PATH_PATTERN = re.compile(r"/opt/data/users/[^\s'\"]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_COMPRESSION_SUMMARY_ERROR_PATTERN = re.compile(
    r"(?is)compression summary failed:\s*(?P<error>.*?)(?:\.\s*inserted\b|$)"
)
_COMPRESSION_ABORT_ERROR_PATTERN = re.compile(
    r"(?is)compression aborted:\s*(?P<error>.*?)(?:\.\s*no messages\b|$)"
)
_TOOL_OUTPUT_PREVIEW_LIMIT = 1200
_OUTPUT_TRUNCATION_MARKERS = (
    "output too long",
    "output truncated",
    "tool response was",
    "too large",
    "capture limit",
    "原始输出隐藏",
    "输出过长",
    "输出已截断",
)


def _project_runtime_model_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project a native transcript into the messages passed to the model.

    This mirrors Hermes gateway replay rules: ``session_meta`` and ``system``
    rows are transcript/session bookkeeping, while tool-call chains need their
    structured fields preserved so replay remains provider-valid.
    """

    projected: list[dict[str, Any]] = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not role or role in {"session_meta", "system"}:
            continue

        has_tool_calls = "tool_calls" in message
        has_tool_call_id = "tool_call_id" in message
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            projected.append(
                {
                    key: value
                    for key, value in message.items()
                    if key not in {"timestamp", "observed"}
                }
            )
            continue

        content = message.get("content")
        if content:
            projected.append({"role": role, "content": content})
    return projected


def _load_runtime_conversation_history(
    session_db: Any,
    session_id: str,
    request_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load model replay history from Hermes' native session store.

    Runtime Manager workers are short-lived, so they must explicitly restore
    the durable Hermes transcript before calling ``run_conversation``.  The raw
    transcript is projected to model replay messages before use; metadata-only
    rows such as ``session_meta`` remain in state.db for transcript/session
    bookkeeping, not in provider-bound history.  The request ``history`` field
    is retained only as a compatibility fallback for older callers that do not
    yet rely on ``state.db``.
    """

    try:
        native_history = session_db.get_messages_as_conversation(
            session_id, repair_alternation=True
        )
    except Exception:
        log_event(
            {
                "event": "runtime.history_restore_failed",
                "session_id": session_id,
                "timestamp": time.time(),
            }
        )
        native_history = []

    restored = _project_runtime_model_history(native_history or [])
    if restored:
        return restored
    return _project_runtime_model_history(request_history or [])


def emit(event: dict[str, Any]) -> None:
    with _OUTPUT_LOCK:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def log_event(event: dict[str, Any]) -> None:
    with _OUTPUT_LOCK:
        sys.stderr.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stderr.flush()


def _configure_runtime_agent_protocol(agent: Any) -> None:
    # Worker stdout is a JSONL protocol stream. Status rendering is owned by
    # status_callback; raw agent status prints would corrupt the stream.
    agent.suppress_status_output = True


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

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    hermes_home = str(request["hermes_home"])
    from runtime_manager.bootstrap import load_profile_environment
    from runtime_manager.ops_env import apply_apiserver_ops_env

    load_profile_environment(hermes_home)
    apply_apiserver_ops_env(request.get("ops_env") or {})
    os.environ["HERMES_RUNTIME_RUN_ID"] = run_id
    os.environ["HERMES_SESSION_KEY"] = approval_session_key
    if request.get("artifact_dir") and not os.environ.get("HERMES_ARTIFACT_DIR"):
        os.environ["HERMES_ARTIFACT_DIR"] = str(request["artifact_dir"])

    from agent.skill_commands import build_preloaded_skills_prompt
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
        if event_type in {"tool.started", "tool.completed"}:
            # Hermes fires tool_progress_callback side-by-side with the richer
            # tool_start/tool_complete callbacks.  Keep start/complete events
            # owned by those structured callbacks so every tool event has a
            # stable tool_call_id and clients do not receive duplicates.
            return
        if event_type == "reasoning.available":
            emit(
                {
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                }
            )

    def on_status(kind: str, message: str) -> None:
        timestamp = time.time()
        emit(
            {
                "event": "status.message",
                "run_id": run_id,
                "timestamp": timestamp,
                "kind": kind,
                "message": message,
            }
        )
        compression_event = _compression_event_from_status(
            kind,
            message,
            provider=provider,
            model=model,
            base_url=base_url,
        )
        if compression_event:
            compression_event["run_id"] = run_id
            compression_event["timestamp"] = timestamp
            emit(compression_event)

    def approval_notify(data: dict[str, Any]) -> None:
        event = dict(data or {})
        event.update(_approval_display_fields(event))
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

    llm_debug_hook_registrations: list[tuple[Any, str, Any]] = []
    tool_started_at: dict[str, float] = {}
    seen_artifact_ids: set[str] = set()
    tool_lock = threading.Lock()

    def on_tool_start(tool_call_id: str, tool_name: str, args: Any) -> None:
        started_at = time.time()
        with tool_lock:
            tool_started_at[str(tool_call_id)] = started_at
        event = {
            "event": "tool.started",
            "run_id": run_id,
            "timestamp": started_at,
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "preview": _safe_tool_action_summary(tool_name, args),
        }
        if command_preview := _safe_tool_command_preview(tool_name, args):
            event["command_preview"] = command_preview
        emit(event)

    def on_tool_complete(tool_call_id: str, tool_name: str, args: Any, result: Any) -> None:
        completed_at = time.time()
        with tool_lock:
            started_at = tool_started_at.pop(str(tool_call_id), None)
        event = {
            "event": "tool.completed",
            "run_id": run_id,
            "timestamp": completed_at,
            "tool_call_id": tool_call_id,
            "tool": tool_name,
        }
        if command_preview := _safe_tool_command_preview(tool_name, args):
            event["command_preview"] = command_preview
        event.update(
            _safe_tool_result_fields(
                result,
                run_id=run_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                seen_artifact_ids=seen_artifact_ids,
            )
        )
        if started_at is not None:
            event["duration"] = round(completed_at - started_at, 3)
        emit(event)

    def on_reasoning_delta(text: str | None) -> None:
        if not text:
            return
        emit(
            {
                "event": "reasoning.available",
                "run_id": run_id,
                "timestamp": time.time(),
                "text": text,
            }
        )

    def on_interim_assistant(text: str | None, *unused_args, **kwargs) -> None:
        _ = unused_args
        if not text:
            return
        emit(
            {
                "event": "message.interim",
                "run_id": run_id,
                "timestamp": time.time(),
                "text": text,
                "already_streamed": bool(kwargs.get("already_streamed", False)),
            }
        )

    def on_step(iteration: int, prev_tools: list[Any]) -> None:
        previous_tools = _summarize_previous_tools(prev_tools)
        emit(
            {
                "event": "agent.step",
                "run_id": run_id,
                "timestamp": time.time(),
                "iteration": iteration,
                "previous_tool_count": len(prev_tools or []),
                "previous_tools": previous_tools,
            }
        )

    def on_tool_generating(tool_name: str) -> None:
        emit(
            {
                "event": "tool.generating",
                "run_id": run_id,
                "timestamp": time.time(),
                "tool": tool_name,
            }
        )

    def on_thinking(message: str | None) -> None:
        if not message:
            return
        emit(
            {
                "event": "status.message",
                "run_id": run_id,
                "timestamp": time.time(),
                "kind": "thinking",
                "message": message,
            }
        )

    approval_token = None
    session_tokens = []
    try:
        approval_token = set_current_session_key(approval_session_key)
        session_tokens = set_session_vars(
            platform=HERMES_RUNTIME_PLATFORM,
            user_id=user_id,
            session_key=approval_session_key,
        )
        register_gateway_notify(approval_session_key, approval_notify)
        runtime_llm_config = _resolve_runtime_llm_config(request)
        model = runtime_llm_config["model"]
        provider = runtime_llm_config["provider"]
        base_url = runtime_llm_config["base_url"]

        system_prompt = _compose_effective_system_prompt(
            request,
            session_id=session_id,
            skill_prompt_builder=build_preloaded_skills_prompt,
        )

        session_db = SessionDB()
        conversation_history = _load_runtime_conversation_history(
            session_db,
            session_id,
            request.get("history") or [],
        )

        agent = AIAgent(
            model=runtime_llm_config["model"],
            provider=runtime_llm_config["provider"],
            api_key=runtime_llm_config["api_key"],
            base_url=runtime_llm_config["base_url"],
            max_tokens=runtime_llm_config.get("max_tokens"),
            session_id=session_id,
            session_db=session_db,
            quiet_mode=True,
            verbose_logging=False,
            platform=HERMES_RUNTIME_PLATFORM,
            gateway_session_key=approval_session_key,
            stream_delta_callback=on_delta,
            tool_progress_callback=on_tool_progress,
            tool_start_callback=on_tool_start,
            tool_complete_callback=on_tool_complete,
            thinking_callback=on_thinking,
            reasoning_callback=on_reasoning_delta,
            step_callback=on_step,
            interim_assistant_callback=on_interim_assistant,
            tool_gen_callback=on_tool_generating,
            status_callback=on_status,
            enabled_toolsets=request.get("enabled_toolsets"),
            disabled_toolsets=request.get("disabled_toolsets"),
            skip_memory=bool(request.get("skip_memory", False)),
            skip_context_files=bool(request.get("skip_context_files", True)),
            ephemeral_system_prompt=system_prompt,
            max_iterations=int(request.get("max_iterations") or 90),
        )
        _apply_runtime_llm_config_to_agent(agent, runtime_llm_config)
        _configure_runtime_agent_protocol(agent)
        _AGENT_HOLDER["agent"] = agent
        llm_debug_hook_registrations = _register_runtime_llm_debug_hooks(
            run_id=run_id,
            request=request,
            system_prompt=system_prompt,
            emit_fn=emit,
        )

        emit({"event": "run.running", "run_id": run_id, "timestamp": time.time()})
        result = agent.run_conversation(
            user_message=request["message"],
            conversation_history=conversation_history,
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
        _unregister_runtime_llm_debug_hooks(llm_debug_hook_registrations)
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

def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _resolve_runtime_llm_config(request: dict[str, Any]) -> dict[str, Any]:
    llm_config = request.get("llm_config")
    if not isinstance(llm_config, dict):
        llm_config = {}

    model = (
        request.get("model")
        or llm_config.get("model")
        or llm_config.get("name")
        or llm_config.get("default")
        or ""
    )
    api_key = _first_present(request.get("api_key"), llm_config.get("api_key"), llm_config.get("apiKey"))
    base_url = _first_present(
        request.get("base_url"),
        request.get("baseURL"),
        llm_config.get("base_url"),
        llm_config.get("baseURL"),
    )
    provider = _normalize_agent_provider(
        _first_present(request.get("provider"), llm_config.get("provider")),
        base_url=base_url,
    )
    if base_url and api_key:
        provider = "custom"

    resolved = {
        "model": str(model or ""),
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
    }
    max_tokens = _coerce_positive_int(
        _first_present(
            request.get("max_tokens"),
            request.get("maxTokens"),
            llm_config.get("max_tokens"),
            llm_config.get("maxTokens"),
        )
    )
    if max_tokens > 0:
        resolved["max_tokens"] = max_tokens

    context_length = _coerce_positive_int(
        _first_present(
            request.get("context_length"),
            request.get("contextLength"),
            request.get("context_window"),
            request.get("contextWindow"),
            llm_config.get("context_length"),
            llm_config.get("contextLength"),
            llm_config.get("context_window"),
            llm_config.get("contextWindow"),
        )
    )
    if context_length > 0:
        resolved["context_length"] = context_length

    return resolved


def _apply_runtime_llm_config_to_agent(agent: Any, runtime_llm_config: dict[str, Any]) -> None:
    context_length = runtime_llm_config.get("context_length")
    if context_length is None:
        return

    setattr(agent, "_config_context_length", context_length)

    session_model_config = getattr(agent, "_session_init_model_config", None)
    if isinstance(session_model_config, dict):
        session_model_config["context_length"] = context_length

    compressor = getattr(agent, "context_compressor", None)
    update_model = getattr(compressor, "update_model", None)
    if not callable(update_model):
        return

    update_model(
        model=str(getattr(agent, "model", None) or runtime_llm_config.get("model") or ""),
        context_length=context_length,
        base_url=str(getattr(agent, "base_url", None) or runtime_llm_config.get("base_url") or ""),
        api_key=getattr(agent, "api_key", None) or runtime_llm_config.get("api_key") or "",
        provider=str(getattr(agent, "provider", None) or runtime_llm_config.get("provider") or ""),
        api_mode=str(getattr(agent, "api_mode", None) or runtime_llm_config.get("api_mode") or ""),
    )


def _normalize_agent_provider(provider: Any, *, base_url: Any = None) -> str | None:
    if provider is None:
        return None
    value = str(provider).strip().lower().replace("_", "-").replace(" ", "-")
    if not value:
        return None

    openai_compatible_aliases = {
        "openai-compatible",
        "openai-compat",
        "openai-compatible-api",
        "openai-compat-api",
        "openai-compatible-endpoint",
    }
    if value in openai_compatible_aliases:
        return "custom"

    # Hermes uses "openai-api" for its first-class OpenAI API provider and
    # "custom" for caller-supplied OpenAI-compatible endpoints.  Cloud model
    # configs commonly call both variants "openai"; normalize before passing
    # the value into AIAgent so platform/provider metadata remains canonical.
    if value == "openai":
        if isinstance(base_url, str) and base_url.strip():
            return "custom"
        return "openai-api"

    return value


def _base_url_host(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except Exception:
        return ""
    return parsed.netloc or parsed.path.split("/", 1)[0]


def _register_runtime_llm_debug_hooks(
    *,
    run_id: str,
    request: dict[str, Any],
    system_prompt: str | None,
    emit_fn,
) -> list[tuple[Any, str, Any]]:
    try:
        from hermes_cli.plugins import get_plugin_manager
    except Exception:
        return []

    manager = get_plugin_manager()
    registrations: list[tuple[Any, str, Any]] = []

    def on_pre_api_request(**kwargs: Any) -> None:
        event = _llm_request_metadata_event(
            kwargs,
            run_id=run_id,
            request=request,
            system_prompt=system_prompt,
        )
        if event:
            emit_fn(event)
            _log_llm_metadata_event(event, request=request)

    def on_post_api_request(**kwargs: Any) -> None:
        event = _llm_response_metadata_event(kwargs, run_id=run_id)
        if event:
            emit_fn(event)
            _log_llm_metadata_event(event, request=request)

    for hook_name, callback in (
        ("pre_api_request", on_pre_api_request),
        ("post_api_request", on_post_api_request),
    ):
        hooks = getattr(manager, "_hooks", None)
        if not isinstance(hooks, dict):
            return registrations
        hooks.setdefault(hook_name, []).append(callback)
        registrations.append((manager, hook_name, callback))
    return registrations


def _unregister_runtime_llm_debug_hooks(registrations: list[tuple[Any, str, Any]]) -> None:
    for manager, hook_name, callback in registrations:
        try:
            hooks = getattr(manager, "_hooks", None)
            callbacks = hooks.get(hook_name) if isinstance(hooks, dict) else None
            if isinstance(callbacks, list) and callback in callbacks:
                callbacks.remove(callback)
        except Exception:
            continue


def _llm_request_metadata_event(
    hook_kwargs: dict[str, Any],
    *,
    run_id: str,
    request: dict[str, Any],
    system_prompt: str | None,
) -> dict[str, Any]:
    body = _hook_request_body(hook_kwargs)
    tools = body.get("tools")
    if not isinstance(tools, list):
        tools = []

    prompt_text = system_prompt or ""
    skills = _normalize_string_list(request.get("skills"))
    return {
        "event": "llm.request.metadata",
        "run_id": run_id,
        "timestamp": time.time(),
        "api_request_id": str(hook_kwargs.get("api_request_id") or ""),
        "turn_id": str(hook_kwargs.get("turn_id") or ""),
        "api_call_count": _coerce_positive_int(hook_kwargs.get("api_call_count")),
        "provider": str(hook_kwargs.get("provider") or ""),
        "model": str(hook_kwargs.get("model") or ""),
        "base_url_host": _base_url_host(hook_kwargs.get("base_url")),
        "api_mode": str(hook_kwargs.get("api_mode") or ""),
        "message_count": _coerce_positive_int(hook_kwargs.get("message_count")),
        "request_message_count": _message_count_from_body(body),
        "request_char_count": _coerce_positive_int(hook_kwargs.get("request_char_count")),
        "approx_input_tokens": _coerce_positive_int(hook_kwargs.get("approx_input_tokens")),
        "max_tokens": _coerce_positive_int(hook_kwargs.get("max_tokens")),
        "skills_count": len(skills),
        "skill_names": skills,
        "skill_prompt_presence": {
            skill: bool(skill and skill in prompt_text)
            for skill in skills
        },
        "system_prompt_chars": len(prompt_text),
        "system_prompt_sha256": _sha256_hex(prompt_text) if prompt_text else "",
        "enabled_toolsets": _normalize_string_list(request.get("enabled_toolsets")),
        "disabled_toolsets": _normalize_string_list(request.get("disabled_toolsets")),
        "tool_count": _coerce_positive_int(hook_kwargs.get("tool_count")),
        "tools_count": len(tools),
        "tool_names": _tool_names_from_schemas(tools),
        "tool_choice": _summarize_tool_choice(body),
    }


def _llm_response_metadata_event(
    hook_kwargs: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    assistant_message = hook_kwargs.get("assistant_message")
    return {
        "event": "llm.response.metadata",
        "run_id": run_id,
        "timestamp": time.time(),
        "api_request_id": str(hook_kwargs.get("api_request_id") or ""),
        "turn_id": str(hook_kwargs.get("turn_id") or ""),
        "api_call_count": _coerce_positive_int(hook_kwargs.get("api_call_count")),
        "provider": str(hook_kwargs.get("provider") or ""),
        "model": str(hook_kwargs.get("model") or ""),
        "base_url_host": _base_url_host(hook_kwargs.get("base_url")),
        "api_mode": str(hook_kwargs.get("api_mode") or ""),
        "api_duration": _coerce_nonnegative_float(hook_kwargs.get("api_duration")),
        "finish_reason": str(hook_kwargs.get("finish_reason") or ""),
        "response_model": str(hook_kwargs.get("response_model") or ""),
        "message_count": _coerce_positive_int(hook_kwargs.get("message_count")),
        "assistant_content_chars": _coerce_positive_int(
            hook_kwargs.get("assistant_content_chars")
        ),
        "assistant_tool_calls_count": _coerce_positive_int(
            hook_kwargs.get("assistant_tool_call_count")
        ),
        "assistant_tool_names": _tool_names_from_calls(
            getattr(assistant_message, "tool_calls", None)
        ),
        "usage": hook_kwargs.get("usage") if isinstance(hook_kwargs.get("usage"), dict) else {},
    }


def _log_llm_metadata_event(event: dict[str, Any], *, request: dict[str, Any]) -> None:
    try:
        log_event(_llm_metadata_log_record(event, request=request))
    except Exception:
        return


def _llm_metadata_log_record(event: dict[str, Any], *, request: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "message": "runtime.llm.metadata",
        "event": str(event.get("event") or ""),
        "run_id": str(event.get("run_id") or ""),
        "session_id": str(request.get("session_id") or request.get("conversation_id") or ""),
        "user_id": str(request.get("user_id") or ""),
        "conversation_id": str(request.get("conversation_id") or ""),
        "timestamp": event.get("timestamp"),
        "api_request_id": str(event.get("api_request_id") or ""),
        "turn_id": str(event.get("turn_id") or ""),
        "api_call_count": _coerce_positive_int(event.get("api_call_count")),
        "provider": str(event.get("provider") or ""),
        "model": str(event.get("model") or ""),
        "base_url_host": str(event.get("base_url_host") or ""),
        "api_mode": str(event.get("api_mode") or ""),
    }

    if event.get("event") == "llm.request.metadata":
        record.update(
            {
                "skills_count": _coerce_positive_int(event.get("skills_count")),
                "skill_names": _normalize_string_list(event.get("skill_names")),
                "skill_prompt_presence": event.get("skill_prompt_presence")
                if isinstance(event.get("skill_prompt_presence"), dict)
                else {},
                "enabled_toolsets": _normalize_string_list(event.get("enabled_toolsets")),
                "disabled_toolsets": _normalize_string_list(event.get("disabled_toolsets")),
                "tools_count": _coerce_positive_int(event.get("tools_count")),
                "tool_names": _normalize_string_list(event.get("tool_names")),
                "tool_choice": str(event.get("tool_choice") or ""),
                "request_message_count": _coerce_positive_int(
                    event.get("request_message_count")
                ),
                "request_char_count": _coerce_positive_int(event.get("request_char_count")),
                "approx_input_tokens": _coerce_positive_int(event.get("approx_input_tokens")),
                "max_tokens": _coerce_positive_int(event.get("max_tokens")),
                "system_prompt_chars": _coerce_positive_int(event.get("system_prompt_chars")),
                "system_prompt_sha256": str(event.get("system_prompt_sha256") or ""),
            }
        )
    elif event.get("event") == "llm.response.metadata":
        record.update(
            {
                "finish_reason": str(event.get("finish_reason") or ""),
                "assistant_tool_calls_count": _coerce_positive_int(
                    event.get("assistant_tool_calls_count")
                ),
                "assistant_tool_names": _normalize_string_list(
                    event.get("assistant_tool_names")
                ),
                "assistant_content_chars": _coerce_positive_int(
                    event.get("assistant_content_chars")
                ),
                "response_model": str(event.get("response_model") or ""),
                "output_tokens": _usage_output_tokens(event.get("usage")),
            }
        )
    return record


def _usage_output_tokens(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return _coerce_positive_int(
        _first_present(
            value.get("completion_tokens"),
            value.get("output_tokens"),
            value.get("completionTokens"),
            value.get("outputTokens"),
        )
    )


def _hook_request_body(hook_kwargs: dict[str, Any]) -> dict[str, Any]:
    request_payload = hook_kwargs.get("request")
    if not isinstance(request_payload, dict):
        return {}
    body = request_payload.get("body")
    return body if isinstance(body, dict) else {}


def _message_count_from_body(body: dict[str, Any]) -> int:
    for key in ("messages", "input"):
        value = body.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _tool_names_from_schemas(tools: list[Any], *, limit: int = 32) -> list[str]:
    names: list[str] = []
    for item in tools:
        name = _tool_name_from_schema(item)
        if name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _tool_name_from_schema(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    function = value.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    for key in ("name", "tool_name"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def _tool_names_from_calls(value: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = _tool_name_from_call(item)
        if name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _tool_name_from_call(value: Any) -> str:
    function = getattr(value, "function", None)
    name = getattr(function, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    if isinstance(value, dict):
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"].strip()
        if isinstance(value.get("name"), str):
            return value["name"].strip()
    return ""


def _summarize_tool_choice(body: dict[str, Any]) -> str:
    if "tool_choice" not in body:
        return "omitted"
    value = body.get("tool_choice")
    if value is None:
        return "null"
    if isinstance(value, str):
        return value.strip() or "empty"
    if isinstance(value, dict):
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name.strip():
            return f"function:{name.strip()}"
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type.strip():
            return f"type:{item_type.strip()}"
        return "object"
    return type(value).__name__


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _coerce_nonnegative_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else 0.0
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return 0.0
        return parsed if parsed >= 0 else 0.0
    return 0.0


def _safe_status_event_text(value: Any, *, limit: int = 512) -> str:
    text = str(value or "")
    text = _redact_runtime_secret_values(text)
    text = _SENSITIVE_STATUS_VALUE_PATTERN.sub(r"\1\2<redacted>", text)
    text = _BEARER_STATUS_VALUE_PATTERN.sub("Bearer <redacted>", text)
    text = _INTERNAL_USER_PATH_PATTERN.sub("/opt/data/users/<redacted>", text)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _compression_event_from_status(
    kind: Any,
    message: Any,
    *,
    provider: Any = None,
    model: Any = None,
    base_url: Any = None,
) -> dict[str, Any] | None:
    text = str(message or "")
    normalized = text.lower()
    event_name = "context.compression.warning"
    reason = ""
    fallback = False
    abort = False
    summary_error = ""

    if "compression summary failed" in normalized:
        reason = "summary_failed"
        fallback = True
        match = _COMPRESSION_SUMMARY_ERROR_PATTERN.search(text)
        summary_error = match.group("error").strip() if match else ""
    elif "compression aborted" in normalized:
        event_name = "context.compression.failed"
        reason = "compression_aborted"
        abort = True
        match = _COMPRESSION_ABORT_ERROR_PATTERN.search(text)
        summary_error = match.group("error").strip() if match else ""
    elif "no auxiliary llm provider configured" in normalized:
        reason = "no_auxiliary_provider"
        fallback = True
        summary_error = "no auxiliary LLM provider configured"
    elif "session compressed" in normalized and "accuracy may degrade" in normalized:
        reason = "repeated_compression"
    else:
        return None

    event: dict[str, Any] = {
        "event": event_name,
        "reason": reason,
        "sourceEventKind": str(kind or ""),
        "message": _safe_status_event_text(text),
        "fallback": fallback,
        "abort": abort,
        "provider": str(provider or ""),
        "providerSource": "configured" if provider else "auto",
        "model": str(model or ""),
        "baseURLHost": _base_url_host(base_url),
    }
    if summary_error:
        event["summaryError"] = _safe_status_event_text(summary_error)
    return event


def _json_preview(value: Any, *, limit: int = 500) -> str:
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _approval_display_fields(data: dict[str, Any]) -> dict[str, Any]:
    command = data.get("command")
    summary = _safe_command_summary(command)
    fields: dict[str, Any] = {
        "tool_name": "terminal",
        "summary": summary,
        "action_summary": summary,
        "reason": summary,
        "risk": "requires_approval",
    }
    description = data.get("description")
    if isinstance(description, str) and description.strip():
        fields["approval_reason"] = description.strip()
    return fields


def _safe_tool_action_summary(tool_name: Any, args: Any) -> str:
    name = str(tool_name or "").strip()
    command = _extract_tool_command(args)
    if command:
        return _safe_command_summary(command)

    normalized = name.replace("_", " ").replace("-", " ").strip()
    if not normalized:
        return "Run tool"
    lower = normalized.lower()
    if "read" in lower:
        return f"Read data with {normalized}"
    if "search" in lower or "grep" in lower:
        return f"Search data with {normalized}"
    if "write" in lower or "edit" in lower or "delete" in lower:
        return f"Run {normalized} tool that may change data"
    return f"Run {normalized} tool"


def _safe_tool_command_preview(tool_name: Any, args: Any) -> str:
    if str(tool_name or "").strip().lower() != "terminal":
        return ""
    command = _extract_tool_command(args)
    if not command:
        return ""
    return _sanitize_command_preview(command, limit=500)


def _safe_command_summary(command: Any) -> str:
    if not isinstance(command, str) or not command.strip():
        return "Run terminal command"
    tokens = _split_command(command)
    tokens = _drop_command_environment_prefix(tokens)
    if not tokens:
        return "Run terminal command"

    first = tokens[0]
    if first == "kubectl":
        return _safe_kubectl_summary(tokens)
    kubectl_index = _find_command_token(tokens, "kubectl")
    if kubectl_index > 0:
        return _safe_kubectl_summary(tokens[kubectl_index:])
    if first in {"kbcli", "kb"}:
        return "Inspect KubeBlocks resources with kbcli"
    if first in {"bash", "sh", "zsh", "ksh"} and any(
        "c" in token.lstrip("-") for token in tokens[1:3]
    ):
        return "Run shell command that requires approval"
    if first in {"mongosh", "mysql", "psql", "sqlcmd"}:
        return f"Run database client command with {first}"
    return f"Run terminal command with {first}"


def _safe_kubectl_summary(tokens: list[str]) -> str:
    exec_index = _kubectl_find_verb(tokens, "exec")
    if exec_index > 0:
        namespace = _kubectl_namespace(tokens)
        pod = _kubectl_exec_pod(tokens, exec_index)
        detail = []
        if namespace:
            detail.append(f"namespace {namespace}")
        if pod:
            detail.append(f"pod {pod}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        return f"Run command inside Kubernetes workload with kubectl exec{suffix}"

    verb_index = _kubectl_first_verb_index(tokens)
    if verb_index < 0:
        return "Inspect Kubernetes resources with kubectl"
    verb = tokens[verb_index]
    if verb in {"get", "describe", "logs", "top"}:
        resource = (
            tokens[verb_index + 1]
            if len(tokens) > verb_index + 1 and not tokens[verb_index + 1].startswith("-")
            else "resources"
        )
        resource = _safe_kubectl_resource_name(resource)
        return f"Inspect Kubernetes {resource} with kubectl {verb}"
    if verb in {"config", "cluster-info", "version"}:
        return f"Inspect Kubernetes metadata with kubectl {verb}"
    return f"Run kubectl {verb} command"


def _find_command_token(tokens: list[str], command: str) -> int:
    for index, token in enumerate(tokens):
        if token == command:
            return index
    return -1


def _safe_kubectl_resource_name(resource: str) -> str:
    resource = resource.strip().strip(";")
    if not resource or resource.startswith("$"):
        return "resources"
    return resource


def _kubectl_find_verb(tokens: list[str], verb: str) -> int:
    for index, token in enumerate(tokens[1:], start=1):
        if token == verb:
            return index
    return -1


def _kubectl_first_verb_index(tokens: list[str]) -> int:
    flags_with_value = {"-n", "--namespace", "-c", "--container", "--context"}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in flags_with_value:
            index += 2
            continue
        if token.startswith("--namespace=") or token.startswith("--context="):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return index
    return -1


def _kubectl_namespace(tokens: list[str]) -> str:
    for index, token in enumerate(tokens):
        if token in {"-n", "--namespace"} and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("--namespace="):
            return token.split("=", 1)[1]
    return ""


def _kubectl_exec_pod(tokens: list[str], exec_index: int) -> str:
    index = exec_index + 1
    flags_with_value = {"-n", "--namespace", "-c", "--container", "--context"}
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return ""
        if token in flags_with_value:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return ""


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.strip().split()


def _extract_tool_command(args: Any) -> str:
    if isinstance(args, dict):
        return _extract_tool_command_from_mapping(args)
    if isinstance(args, str) and args.strip():
        text = args.strip()
        parsed = _parse_mapping_string(text)
        if isinstance(parsed, dict):
            command = _extract_tool_command_from_mapping(parsed)
            if command:
                return command
        command = _extract_loose_command_arg(text)
        if command:
            return command
        return text
    return ""


def _extract_tool_command_from_mapping(args: dict[Any, Any]) -> str:
    for key in ("command", "cmd", "shell_command"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_mapping_string(text: str) -> Any:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def _extract_loose_command_arg(text: str) -> str:
    match = _LOOSE_COMMAND_ARG_PATTERN.search(text)
    if not match:
        return ""
    value = match.group("value").strip()
    if not value:
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        try:
            parsed = ast.literal_eval(value)
            return parsed.strip() if isinstance(parsed, str) else ""
        except Exception:
            return value[1:-1].strip()
    return value.strip()


def _drop_command_environment_prefix(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens

    index = 0
    if tokens[index] == "export":
        index += 1
        while index < len(tokens) and _ENV_ASSIGNMENT_PATTERN.match(tokens[index]):
            index += 1
        if index < len(tokens) and tokens[index] == "&&":
            index += 1
        return tokens[index:]

    if tokens[index] == "env":
        index += 1

    while index < len(tokens) and _ENV_ASSIGNMENT_PATTERN.match(tokens[index]):
        index += 1
    return tokens[index:]


def _sanitize_command_preview(command: str, *, limit: int) -> str:
    tokens = _drop_command_environment_prefix(_split_command(command))
    text = " ".join(tokens) if tokens else command
    text = _redact_runtime_secret_values(text)
    text = _KUBECONFIG_ASSIGNMENT_PATTERN.sub(r"\1", text)
    text = _KUBECONFIG_FLAG_EQUAL_PATTERN.sub(" ", text)
    text = _KUBECONFIG_FLAG_VALUE_PATTERN.sub(" ", text)
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", text)
    text = _SENSITIVE_FLAG_EQUAL_PATTERN.sub(r"\1<redacted>", text)
    text = _SENSITIVE_FLAG_VALUE_PATTERN.sub(r"\1 <redacted>", text)
    text = _INTERNAL_USER_PATH_PATTERN.sub("/opt/data/users/<redacted>", text)
    return _bounded_preview(text, limit=limit)


def _safe_text_preview(text: str, *, limit: int) -> str:
    text = _redact_runtime_secret_values(text)
    text = _KUBECONFIG_ASSIGNMENT_PATTERN.sub(r"\1", text)
    text = _KUBECONFIG_FLAG_EQUAL_PATTERN.sub(" ", text)
    text = _KUBECONFIG_FLAG_VALUE_PATTERN.sub(" ", text)
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", text)
    text = _SENSITIVE_FLAG_EQUAL_PATTERN.sub(r"\1<redacted>", text)
    text = _SENSITIVE_FLAG_VALUE_PATTERN.sub(r"\1 <redacted>", text)
    text = _INTERNAL_USER_PATH_PATTERN.sub("/opt/data/users/<redacted>", text)
    return _bounded_preview(text, limit=limit)


def _safe_output_preview(text: str, *, limit: int) -> str:
    text = _redact_runtime_secret_values(text)
    text = _KUBECONFIG_ASSIGNMENT_PATTERN.sub(r"\1", text)
    text = _KUBECONFIG_FLAG_EQUAL_PATTERN.sub(" ", text)
    text = _KUBECONFIG_FLAG_VALUE_PATTERN.sub(" ", text)
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", text)
    text = _SENSITIVE_FLAG_EQUAL_PATTERN.sub(r"\1<redacted>", text)
    text = _SENSITIVE_FLAG_VALUE_PATTERN.sub(r"\1 <redacted>", text)
    text = _INTERNAL_USER_PATH_PATTERN.sub("/opt/data/users/<redacted>", text)
    return _bounded_output_preview(text, limit=limit)


def _redact_runtime_secret_values(text: str) -> str:
    if not text:
        return text
    try:
        from runtime_manager.ops_env import runtime_secret_values_from_env

        for secret in runtime_secret_values_from_env():
            text = text.replace(secret, "<redacted>")
    except Exception:
        pass
    return text


def _bounded_preview(text: str, *, limit: int) -> str:
    text = _WHITESPACE_PATTERN.sub(" ", str(text or "")).strip()
    if limit > 0 and len(text) > limit:
        return text[:limit] + "..."
    return text


def _bounded_output_preview(text: str, *, limit: int) -> str:
    text = str(text or "").strip()
    if limit > 0 and len(text) > limit:
        return text[:limit] + "..."
    return text


def _safe_tool_result_summary(result: Any) -> str:
    size = _serialized_size(result)
    if _tool_result_has_error(result):
        return f"Tool failed; output size {size} bytes"
    return f"Tool completed; output size {size} bytes"


def _safe_tool_result_fields(
    result: Any,
    *,
    run_id: Any = None,
    session_id: Any = None,
    tool_call_id: Any = None,
    tool_name: Any = None,
    seen_artifact_ids: set[str] | None = None,
) -> dict[str, Any]:
    parsed = _parse_tool_result(result)
    has_error = _tool_result_has_error(parsed)
    fields: dict[str, Any] = {
        "result_preview": _safe_tool_result_summary(result),
        "error": has_error,
    }
    if not isinstance(parsed, dict):
        return fields

    exit_code = _first_value(parsed, "exit_code", "exitCode", "returncode", "return_code")
    if exit_code is not None:
        fields["exit_code"] = exit_code
        fields["exitCode"] = exit_code

    stdout = _first_text(parsed, "output", "stdout")
    reported_output_bytes = _coerce_positive_int(_first_value(parsed, "output_bytes", "outputBytes"))
    reported_artifact = _artifact_metadata_from_mapping(parsed.get("artifact"))
    reported_artifacts = _artifact_list_from_value(parsed.get("artifacts"))
    if reported_artifact and reported_artifact not in reported_artifacts:
        reported_artifacts.insert(0, reported_artifact)
    reported_artifact_ids = {
        artifact["artifactId"]
        for artifact in reported_artifacts
        if isinstance(artifact.get("artifactId"), str)
    }
    discovered_artifacts = _discover_report_artifacts_from_dir(
        _artifact_dir_from_env(),
        seen_artifact_ids=reported_artifact_ids | (seen_artifact_ids or set()),
    )
    if seen_artifact_ids is not None:
        seen_artifact_ids.update(artifact["artifactId"] for artifact in discovered_artifacts)
    artifacts = reported_artifacts + discovered_artifacts
    if artifacts:
        fields["artifact"] = artifacts[0]
        fields["artifacts"] = artifacts

    output_bytes = 0
    if stdout:
        fields["stdout_preview"] = _safe_output_preview(stdout, limit=_TOOL_OUTPUT_PREVIEW_LIMIT)
        fields["stdoutPreview"] = fields["stdout_preview"]
        output_bytes = reported_output_bytes or len(stdout.encode("utf-8", errors="replace"))
        fields["output_bytes"] = output_bytes
        fields["outputBytes"] = output_bytes
        if _truthy_result_value(parsed.get("truncated")) or output_bytes > _TOOL_OUTPUT_PREVIEW_LIMIT:
            fields["truncated"] = True
            if not has_error:
                fields["partial"] = True
                fields["warningReason"] = "output_truncated"
                fields["warning_reason"] = "output_truncated"
                if "artifact" not in fields and isinstance(parsed.get("artifactUnavailableReason"), str):
                    fields["artifactUnavailableReason"] = parsed["artifactUnavailableReason"]

    stderr = _first_string(parsed, "stderr")
    error = parsed.get("error")
    if not stderr and _truthy_result_value(error) and not _output_truncation_warning(parsed, stdout=stdout):
        stderr = str(error)
    if stderr:
        fields["stderr_preview"] = _safe_output_preview(stderr, limit=_TOOL_OUTPUT_PREVIEW_LIMIT)
        fields["stderrPreview"] = fields["stderr_preview"]

    return fields


def _artifact_dir_from_env() -> Path | None:
    raw = os.environ.get("HERMES_ARTIFACT_DIR", "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _discover_report_artifacts_from_dir(
    artifact_dir: Path | None,
    *,
    seen_artifact_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if artifact_dir is None:
        return []
    try:
        from runtime_manager.artifacts import discover_artifacts

        return discover_artifacts(artifact_dir, seen_artifact_ids=seen_artifact_ids)
    except Exception:
        return []


def _parse_tool_result(result: Any) -> Any:
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return result
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                continue
    return result


def _first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_string(mapping: dict[str, Any], *keys: str) -> str:
    value = _first_value(mapping, *keys)
    if isinstance(value, str):
        return value.strip()
    return ""


def _first_text(mapping: dict[str, Any], *keys: str) -> str:
    value = _first_value(mapping, *keys)
    if isinstance(value, str):
        return value
    return ""


def _coerce_positive_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else 0
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return 0
        return parsed if parsed > 0 else 0
    return 0


def _artifact_metadata_from_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    artifact_id = value.get("artifactId")
    file_name = value.get("fileName")
    size_bytes = _coerce_positive_int(value.get("sizeBytes"))
    mime_type = value.get("mimeType")
    sha256 = value.get("sha256")
    if not all(isinstance(item, str) and item.strip() for item in (artifact_id, file_name, mime_type, sha256)):
        return None
    if not size_bytes:
        return None
    if "/" in artifact_id or "\\" in artifact_id:
        return None
    artifact = {
        "artifactId": artifact_id.strip(),
        "fileName": file_name.strip(),
        "sizeBytes": size_bytes,
        "mimeType": mime_type.strip(),
        "sha256": sha256.strip(),
    }
    for key in ("kind", "source", "summary"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            artifact[key] = item.strip()
    for key in ("canBrowse", "canDownload"):
        item = value.get(key)
        if isinstance(item, bool):
            artifact[key] = item
    return artifact


def _artifact_list_from_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    artifacts = []
    for item in value:
        artifact = _artifact_metadata_from_mapping(item)
        if artifact:
            artifacts.append(artifact)
    return artifacts


def _serialized_size(value: Any) -> int:
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return len(text.encode("utf-8", errors="replace"))


def _summarize_previous_tools(
    previous_tools: list[Any] | None,
    *,
    max_tools: int = 8,
) -> list[dict[str, Any]]:
    """Return JSONL-safe tool history summaries for agent.step events.

    AIAgent passes full previous tool results to step_callback. For kubectl
    diagnostics those results can easily exceed asyncio's default line limit
    when serialized as one stdout JSONL event. Runtime Manager only needs
    progress context here, not full tool outputs, so keep bounded display
    summaries and byte counts.
    """
    summaries: list[dict[str, Any]] = []
    for item in (previous_tools or [])[:max_tools]:
        if isinstance(item, dict):
            summary: dict[str, Any] = {}
            name = item.get("name") or item.get("tool") or item.get("tool_name")
            if name is not None:
                summary["name"] = str(name)
            if "arguments" in item:
                summary["action_summary"] = _safe_tool_action_summary(name, item.get("arguments"))
                summary["arguments_bytes"] = _serialized_size(item.get("arguments"))
            if "result" in item:
                summary["result_summary"] = _safe_tool_result_summary(item.get("result"))
                summary["result_bytes"] = _serialized_size(item.get("result"))
            if not summary:
                summary["summary"] = "Tool history item"
                summary["bytes"] = _serialized_size(item)
            summaries.append(summary)
        else:
            summaries.append(
                {
                    "summary": "Tool history item",
                    "bytes": _serialized_size(item),
                }
            )
    if previous_tools and len(previous_tools) > max_tools:
        summaries.append(
            {
                "truncated": True,
                "remaining_count": len(previous_tools) - max_tools,
            }
        )
    return summaries


def _tool_result_has_error(value: Any) -> bool:
    value = _parse_tool_result(value)
    if isinstance(value, dict):
        if _truthy_result_value(value.get("is_error")):
            return True
        for key in ("exit_code", "exitCode", "returncode", "return_code"):
            if _exit_code_failed(value.get(key)):
                return True
        status = value.get("status")
        if isinstance(status, str) and status.strip().lower() in {
            "error",
            "failed",
            "failure",
            "blocked",
            "timed_out",
            "timeout",
        }:
            return True
        stdout = _first_text(value, "output", "stdout")
        if _truthy_result_value(value.get("error")):
            return not _output_truncation_warning(value, stdout=stdout)
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
        except Exception:
            return False
        return _tool_result_has_error(parsed)
    return False


def _output_truncation_warning(mapping: dict[str, Any], *, stdout: str) -> bool:
    if not stdout:
        return False
    if not _exit_code_success(mapping):
        return False
    error = mapping.get("error")
    if not _truthy_result_value(error):
        return False
    return _is_output_truncation_notice(str(error))


def _exit_code_success(mapping: dict[str, Any]) -> bool:
    exit_code_seen = False
    for key in ("exit_code", "exitCode", "returncode", "return_code"):
        if key not in mapping:
            continue
        exit_code_seen = True
        if _exit_code_failed(mapping.get(key)):
            return False
    return exit_code_seen


def _is_output_truncation_notice(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _OUTPUT_TRUNCATION_MARKERS)


def _truthy_result_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "0", "false", "none", "null", "nil", "ok", "success"}
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def _exit_code_failed(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            return int(stripped) != 0
        except ValueError:
            return False
    return False


def _compose_effective_system_prompt(
    request: dict[str, Any],
    *,
    session_id: str,
    skill_prompt_builder=None,
) -> str | None:
    prompt_parts: list[str] = []
    base_prompt = request.get("system_prompt")
    if isinstance(base_prompt, str) and base_prompt.strip():
        prompt_parts.append(base_prompt.strip())

    skills = _normalize_string_list(request.get("skills"))
    if skills:
        if skill_prompt_builder is None:
            from agent.skill_commands import build_preloaded_skills_prompt

            skill_prompt_builder = build_preloaded_skills_prompt
        skill_prompt, _loaded_skills, missing_skills = skill_prompt_builder(skills, task_id=session_id)
        if missing_skills:
            raise RuntimeError(f"missing Runtime Manager skills: {', '.join(missing_skills)}")
        if skill_prompt:
            prompt_parts.append(str(skill_prompt).strip())

    if not prompt_parts:
        return None
    return "\n\n".join(prompt_parts)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
