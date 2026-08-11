import asyncio
import json
import os
import stat

import pytest

from runtime_manager.home_resolver import UserHomeResolver
from runtime_manager.registry import RunRegistry


@pytest.mark.asyncio
async def test_run_registry_updates_terminal_state_and_replays_to_subscriber():
    registry = RunRegistry()
    handle = registry.create(
        run_id="run_1",
        user_id="u1",
        conversation_id="c1",
        session_id="s1",
        model="m1",
    )
    queue = handle.subscribe()
    handle.publish({"event": "run.running", "run_id": "run_1", "timestamp": 1.0})
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["event"] == "run.running"
    assert handle.status == "running"

    handle.publish({
        "event": "run.completed",
        "run_id": "run_1",
        "timestamp": 2.0,
        "output": "done",
        "usage": {"total_tokens": 3},
    })
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["event"] == "run.completed"
    assert handle.status == "completed"
    assert handle.output == "done"
    assert handle.snapshot()["usage"]["total_tokens"] == 3


def test_user_home_resolver_creates_bootstrap_dirs(tmp_path):
    resolver = UserHomeResolver(tmp_path)
    home = resolver.resolve("user-1")
    assert home == tmp_path / "user-1"
    for name in ("home", "workspace", "sessions", "memories", "skills", "logs"):
        assert (home / name).is_dir()
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "home").stat().st_mode) == 0o700


def test_user_home_resolver_rejects_invalid_ids(tmp_path):
    resolver = UserHomeResolver(tmp_path)
    with pytest.raises(ValueError):
        resolver.resolve("../escape")


def test_runtime_worker_uses_hermes_api_server_platform():
    from runtime_manager.worker_main import HERMES_RUNTIME_PLATFORM

    assert HERMES_RUNTIME_PLATFORM == "api_server"


def test_runtime_worker_prefers_native_session_history_over_request_history():
    from runtime_manager.worker_main import _load_runtime_conversation_history

    class DB:
        def __init__(self):
            self.calls = []

        def get_messages_as_conversation(self, session_id, *, repair_alternation=False):
            self.calls.append((session_id, repair_alternation))
            return [
                {"role": "session_meta", "content": "{}"},
                {"role": "user", "content": "native previous question"},
                {
                    "role": "assistant",
                    "content": "native answer",
                    "tool_calls": [{"id": "call_1", "type": "function"}],
                },
            ]

    db = DB()

    history = _load_runtime_conversation_history(
        db,
        "conv-1",
        [{"role": "user", "content": "cloud display history"}],
    )

    assert db.calls == [("conv-1", True)]
    assert history == [
        {"role": "user", "content": "native previous question"},
        {
            "role": "assistant",
            "content": "native answer",
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
    ]


def test_runtime_worker_falls_back_to_request_history_when_native_session_empty():
    from runtime_manager.worker_main import _load_runtime_conversation_history

    class DB:
        def get_messages_as_conversation(self, session_id, *, repair_alternation=False):
            return []

    request_history = [{"role": "user", "content": "legacy caller history"}]

    assert _load_runtime_conversation_history(DB(), "conv-empty", request_history) == request_history


def test_runtime_worker_normalizes_cloud_provider_aliases_to_hermes_names():
    from runtime_manager.worker_main import _normalize_agent_provider

    assert _normalize_agent_provider("openai-compatible", base_url="https://models.example/v1") == "custom"
    assert _normalize_agent_provider("openai_compat", base_url="https://models.example/v1") == "custom"
    assert _normalize_agent_provider("openai", base_url="https://api.openai.com/v1") == "custom"
    assert _normalize_agent_provider("openai") == "openai-api"
    assert _normalize_agent_provider("qwen-oauth") == "qwen-oauth"


def test_runtime_worker_resolves_cloud_llm_config_for_agent_and_auxiliary():
    from runtime_manager.worker_main import _resolve_runtime_llm_config

    resolved = _resolve_runtime_llm_config(
        {
            "llm_config": {
                "provider": "openai-compatible",
                "model": "qwen3.6-35b-a3b",
                "baseURL": "https://models.example/v1",
                "apiKey": "secret-key",
            }
        }
    )

    assert resolved == {
        "model": "qwen3.6-35b-a3b",
        "provider": "custom",
        "base_url": "https://models.example/v1",
        "api_key": "secret-key",
    }


def test_runtime_worker_treats_named_cloud_provider_with_endpoint_as_custom():
    from runtime_manager.worker_main import _resolve_runtime_llm_config

    resolved = _resolve_runtime_llm_config(
        {
            "llm_config": {
                "provider": "Qwen",
                "model": "qwen3.6-35b-a3b",
                "baseURL": "https://models.example/v1",
                "apiKey": "secret-key",
            }
        }
    )

    assert resolved == {
        "model": "qwen3.6-35b-a3b",
        "provider": "custom",
        "base_url": "https://models.example/v1",
        "api_key": "secret-key",
    }


def test_runtime_worker_resolves_model_limit_entries_from_llm_config():
    from runtime_manager.worker_main import _resolve_runtime_llm_config

    resolved = _resolve_runtime_llm_config(
        {
            "llm_config": {
                "provider": "Qwen",
                "model": "qwen3.6-35b-a3b",
                "baseURL": "https://models.example/v1",
                "apiKey": "secret-key",
                "max_tokens": "8192",
                "context_length": "131072",
            }
        }
    )

    assert resolved["max_tokens"] == 8192
    assert resolved["context_length"] == 131072


def test_runtime_worker_applies_context_length_to_agent_compressor():
    from runtime_manager.worker_main import _apply_runtime_llm_config_to_agent

    class Compressor:
        def __init__(self):
            self.update_call = None

        def update_model(self, **kwargs):
            self.update_call = kwargs

    class Agent:
        model = "qwen3.6-35b-a3b"
        provider = "custom"
        base_url = "https://models.example/v1"
        api_key = "secret-key"
        api_mode = ""

        def __init__(self):
            self.context_compressor = Compressor()
            self._session_init_model_config = {}

    agent = Agent()

    _apply_runtime_llm_config_to_agent(
        agent,
        {
            "context_length": 131072,
        },
    )

    assert agent._config_context_length == 131072
    assert agent._session_init_model_config["context_length"] == 131072
    assert agent.context_compressor.update_call == {
        "model": "qwen3.6-35b-a3b",
        "context_length": 131072,
        "base_url": "https://models.example/v1",
        "api_key": "secret-key",
        "provider": "custom",
        "api_mode": "",
    }


def test_runtime_ops_env_extracts_only_apiserver_allowlist():
    from runtime_manager.ops_env import extract_apiserver_ops_env

    extracted = extract_apiserver_ops_env(
        {
            "opsEnv": {
                "APISERVER_USER_TYPE": "frontend",
                "APISERVER_OPS_ENDPOINT": "http://apiserver.kb-cloud:8080",
                "APISERVER_USER_TOKEN": "secret-token",
                "OPENAI_API_KEY": "must-not-forward",
            },
            "metadata": {
                "opsEnv": {
                    "APISERVER_USER_TYPE": "metadata-value",
                    "APISERVER_USER_TOKEN": "",
                }
            },
        }
    )

    assert extracted == {
        "APISERVER_USER_TYPE": "metadata-value",
        "APISERVER_OPS_ENDPOINT": "http://apiserver.kb-cloud:8080",
        "APISERVER_USER_TOKEN": "secret-token",
    }


def test_runtime_ops_env_reapplies_after_profile_bootstrap(monkeypatch):
    from runtime_manager.ops_env import apply_apiserver_ops_env

    monkeypatch.setenv("APISERVER_USER_TOKEN", "stale-profile-token")
    applied = apply_apiserver_ops_env(
        {
            "APISERVER_USER_TYPE": "admin",
            "APISERVER_OPS_ENDPOINT": "http://internal-apiserver",
            "APISERVER_USER_TOKEN": "fresh-run-token",
            "UNSAFE_ENV": "ignored",
        }
    )

    assert applied == {
        "APISERVER_USER_TYPE": "admin",
        "APISERVER_OPS_ENDPOINT": "http://internal-apiserver",
        "APISERVER_USER_TOKEN": "fresh-run-token",
    }
    assert os.environ["APISERVER_USER_TOKEN"] == "fresh-run-token"
    assert "UNSAFE_ENV" not in os.environ


def test_runtime_worker_llm_request_metadata_is_safe_and_actionable():
    from runtime_manager.worker_main import _llm_request_metadata_event

    event = _llm_request_metadata_event(
        {
            "api_request_id": "turn-1:api:1",
            "turn_id": "turn-1",
            "api_call_count": 1,
            "provider": "custom",
            "model": "qwen3.6-35b-a3b",
            "base_url": "https://models.example/v1",
            "api_mode": "chat_completions",
            "message_count": 3,
            "request_char_count": 12345,
            "approx_input_tokens": 6789,
            "max_tokens": 8192,
            "tool_count": 6,
            "request": {
                "body": {
                    "messages": [
                        {"role": "system", "content": "SECRET PROMPT"},
                        {"role": "user", "content": "diagnose"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "parameters": {"properties": {"command": {"description": "SECRET SCHEMA"}}},
                            },
                        },
                        {"type": "function", "function": {"name": "read_file"}},
                    ],
                    "tool_choice": "auto",
                    "api_key": "secret-key",
                }
            },
        },
        run_id="run-1",
        request={
            "skills": ["dba-diagnose-oracle"],
            "enabled_toolsets": ["terminal", "file"],
        },
        system_prompt="BASE\n\ndba-diagnose-oracle\n\nSECRET PROMPT",
    )

    assert event["event"] == "llm.request.metadata"
    assert event["run_id"] == "run-1"
    assert event["skills_count"] == 1
    assert event["skill_names"] == ["dba-diagnose-oracle"]
    assert event["skill_prompt_presence"] == {"dba-diagnose-oracle": True}
    assert event["system_prompt_chars"] > 0
    assert len(event["system_prompt_sha256"]) == 64
    assert event["tools_count"] == 2
    assert event["tool_names"] == ["terminal", "read_file"]
    assert event["tool_choice"] == "auto"
    assert event["enabled_toolsets"] == ["terminal", "file"]
    encoded = json.dumps(event, ensure_ascii=False)
    assert "SECRET PROMPT" not in encoded
    assert "SECRET SCHEMA" not in encoded
    assert "secret-key" not in encoded


def test_runtime_worker_llm_request_metadata_log_record_is_safe_and_greppable():
    from runtime_manager.worker_main import (
        _llm_metadata_log_record,
        _llm_request_metadata_event,
    )

    event = _llm_request_metadata_event(
        {
            "api_request_id": "turn-1:api:1",
            "turn_id": "turn-1",
            "api_call_count": 1,
            "provider": "custom",
            "model": "qwen3.6-35b-a3b",
            "base_url": "https://models.example/v1",
            "request_char_count": 12345,
            "approx_input_tokens": 6789,
            "request": {
                "body": {
                    "messages": [{"role": "user", "content": "SECRET USER INPUT"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "parameters": {"properties": {"command": {"description": "SECRET SCHEMA"}}},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "api_key": "secret-key",
                }
            },
        },
        run_id="run-1",
        request={
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "session_id": "session-1",
            "message": "SECRET REQUEST",
            "skills": ["dba-diagnose-oracle"],
            "enabled_toolsets": ["terminal", "file"],
        },
        system_prompt="BASE\n\ndba-diagnose-oracle\n\nSECRET PROMPT",
    )

    record = _llm_metadata_log_record(
        event,
        request={
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "session_id": "session-1",
            "message": "SECRET REQUEST",
        },
    )

    assert record["message"] == "runtime.llm.metadata"
    assert record["event"] == "llm.request.metadata"
    assert record["run_id"] == "run-1"
    assert record["session_id"] == "session-1"
    assert record["user_id"] == "user-1"
    assert record["conversation_id"] == "conv-1"
    assert record["skill_names"] == ["dba-diagnose-oracle"]
    assert record["tools_count"] == 1
    assert record["tool_names"] == ["terminal"]
    assert record["tool_choice"] == "auto"

    encoded = json.dumps(record, ensure_ascii=False)
    assert "SECRET USER INPUT" not in encoded
    assert "SECRET REQUEST" not in encoded
    assert "SECRET PROMPT" not in encoded
    assert "SECRET SCHEMA" not in encoded
    assert "secret-key" not in encoded


def test_runtime_worker_llm_response_metadata_is_safe_and_actionable():
    from types import SimpleNamespace

    from runtime_manager.worker_main import _llm_response_metadata_event

    tool_call = SimpleNamespace(function=SimpleNamespace(name="terminal"))
    event = _llm_response_metadata_event(
        {
            "api_request_id": "turn-1:api:1",
            "turn_id": "turn-1",
            "api_call_count": 1,
            "provider": "custom",
            "model": "qwen3.6-35b-a3b",
            "base_url": "https://models.example/v1",
            "api_mode": "chat_completions",
            "api_duration": 1.25,
            "finish_reason": "tool_calls",
            "response_model": "qwen3.6-35b-a3b",
            "message_count": 3,
            "assistant_content_chars": 0,
            "assistant_tool_call_count": 1,
            "assistant_message": SimpleNamespace(tool_calls=[tool_call]),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
        run_id="run-1",
    )

    assert event["event"] == "llm.response.metadata"
    assert event["run_id"] == "run-1"
    assert event["finish_reason"] == "tool_calls"
    assert event["assistant_tool_calls_count"] == 1
    assert event["assistant_tool_names"] == ["terminal"]
    assert event["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}


def test_runtime_worker_llm_response_metadata_log_record_is_safe_and_greppable():
    from types import SimpleNamespace

    from runtime_manager.worker_main import (
        _llm_metadata_log_record,
        _llm_response_metadata_event,
    )

    tool_call = SimpleNamespace(function=SimpleNamespace(name="terminal"))
    event = _llm_response_metadata_event(
        {
            "api_request_id": "turn-1:api:1",
            "turn_id": "turn-1",
            "api_call_count": 1,
            "provider": "custom",
            "model": "qwen3.6-35b-a3b",
            "base_url": "https://models.example/v1",
            "api_mode": "chat_completions",
            "finish_reason": "tool_calls",
            "response_model": "qwen3.6-35b-a3b",
            "assistant_content_chars": 0,
            "assistant_tool_call_count": 1,
            "assistant_message": SimpleNamespace(tool_calls=[tool_call]),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
        run_id="run-1",
    )

    record = _llm_metadata_log_record(
        event,
        request={
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "session_id": "session-1",
            "message": "SECRET REQUEST",
        },
    )

    assert record["message"] == "runtime.llm.metadata"
    assert record["event"] == "llm.response.metadata"
    assert record["run_id"] == "run-1"
    assert record["session_id"] == "session-1"
    assert record["finish_reason"] == "tool_calls"
    assert record["assistant_tool_calls_count"] == 1
    assert record["assistant_tool_names"] == ["terminal"]
    assert record["output_tokens"] == 20
    assert "SECRET REQUEST" not in json.dumps(record, ensure_ascii=False)


def test_runtime_worker_projects_compression_status_as_structured_event():
    from runtime_manager.worker_main import _compression_event_from_status

    event = _compression_event_from_status(
        "warn",
        "⚠ Compression summary failed: no auxiliary LLM provider configured "
        "api_key=secret-token. Inserted a fallback context marker.",
        provider="custom",
        model="qwen3.6-35b-a3b",
        base_url="https://models.example/v1",
    )

    assert event is not None
    assert event["event"] == "context.compression.warning"
    assert event["reason"] == "summary_failed"
    assert event["fallback"] is True
    assert event["abort"] is False
    assert event["provider"] == "custom"
    assert event["model"] == "qwen3.6-35b-a3b"
    assert event["baseURLHost"] == "models.example"
    assert "secret-token" not in json.dumps(event, ensure_ascii=False)


def test_runtime_worker_tool_event_helpers_are_json_safe():
    from runtime_manager.worker_main import (
        _approval_display_fields,
        _json_preview,
        _safe_tool_command_preview,
        _safe_tool_action_summary,
        _safe_tool_result_fields,
        _safe_tool_result_summary,
        _summarize_previous_tools,
        _tool_result_has_error,
    )

    assert _json_preview({"tool": "kubectl", "args": ["get", "pods"]}) == (
        '{"tool": "kubectl", "args": ["get", "pods"]}'
    )
    assert _json_preview("x" * 520).endswith("...")
    assert _tool_result_has_error({"error": "boom"})
    assert _tool_result_has_error('{"error":"boom"}')
    assert _tool_result_has_error({"is_error": True})
    assert _tool_result_has_error({"exit_code": 1})
    assert _tool_result_has_error({"status": "timed_out"})
    assert not _tool_result_has_error({"error": None, "exit_code": 0})
    assert not _tool_result_has_error({"error": "", "exit_code": 0})
    assert not _tool_result_has_error({"error": False, "exit_code": 0})
    assert not _tool_result_has_error('{"error": null, "exit_code": 0}')
    assert not _tool_result_has_error({"ok": True})
    assert not _tool_result_has_error("plain text result")
    assert _safe_tool_result_summary({"output": "ok", "error": None, "exit_code": 0}).startswith(
        "Tool completed;"
    )
    assert _safe_tool_result_summary({"output": "", "error": "boom", "exit_code": 1}).startswith(
        "Tool failed;"
    )

    summary = _summarize_previous_tools(
        [
            {
                "name": "terminal",
                "arguments": {"command": "kubectl get pods -A"},
                "result": "x" * 10000,
            }
        ]
    )
    assert len(json.dumps(summary, ensure_ascii=False)) < 2000
    assert summary[0]["name"] == "terminal"
    assert summary[0]["action_summary"] == "Inspect Kubernetes pods with kubectl get"
    assert summary[0]["result_bytes"] == 10000
    assert summary[0]["result_summary"] == "Tool completed; output size 10000 bytes"
    assert "x" * 1000 not in json.dumps(summary, ensure_ascii=False)

    kubectl_exec_summary = _safe_tool_action_summary(
        "terminal",
        {
            "command": (
                "kubectl exec -n kb-cloud apecloud-pg-0 -- "
                "bash -c 'ps aux && ss -lntp && cat /var/lib/kubeconfig'"
            )
        },
    )
    assert kubectl_exec_summary == (
        "Run command inside Kubernetes workload with kubectl exec "
        "(namespace kb-cloud, pod apecloud-pg-0)"
    )
    assert "ps aux" not in kubectl_exec_summary
    assert "/var/lib" not in kubectl_exec_summary

    assert (
        _safe_tool_action_summary("terminal", '{"command":"kubectl describe pod demo-0 -n kb-cloud"}')
        == "Inspect Kubernetes pod with kubectl describe"
    )
    assert (
        _safe_tool_action_summary("terminal", "{command: 'kubectl logs demo-0 -n kb-cloud'}")
        == "Inspect Kubernetes demo-0 with kubectl logs"
    )
    assert (
        _safe_tool_action_summary(
            "terminal",
            "{'command': 'KUBECONFIG=/opt/data/users/u1/kubeconfig "
            "kubectl get pvc -n kb-cloud'}",
        )
        == "Inspect Kubernetes pvc with kubectl get"
    )
    assert (
        _safe_tool_action_summary(
            "terminal",
            {"command": "for p in demo-0 demo-1; do kubectl logs $p -n kb-cloud; done"},
        )
        == "Inspect Kubernetes resources with kubectl logs"
    )
    assert (
        _safe_tool_action_summary(
            "terminal",
            {"command": "echo checking events && kubectl get events -n kb-cloud"},
        )
        == "Inspect Kubernetes events with kubectl get"
    )
    command_preview = _safe_tool_command_preview(
        "terminal",
        {
            "command": (
                "KUBECONFIG=/opt/data/users/u1/kubeconfig kubectl get pods -n kb-cloud "
                "--token super-secret"
            )
        },
    )
    assert command_preview == "kubectl get pods -n kb-cloud --token <redacted>"
    assert "KUBECONFIG" not in command_preview
    assert "/opt/data/users" not in command_preview
    assert "super-secret" not in command_preview

    result_fields = _safe_tool_result_fields(
        json.dumps(
            {
                "output": (
                    "NAME     READY   STATUS\n"
                    "demo-0   1/1     Running TOKEN=secret-token\n"
                    "config   /opt/data/users/u1/kubeconfig"
                ),
                "stderr": "warning:\tSECRET=stderr-secret\nnext line",
                "exit_code": 0,
                "error": None,
            }
        )
    )
    assert result_fields["error"] is False
    assert result_fields["exit_code"] == 0
    assert result_fields["stdout_preview"] == (
        "NAME     READY   STATUS\n"
        "demo-0   1/1     Running TOKEN=<redacted>\n"
        "config   /opt/data/users/<redacted>"
    )
    assert result_fields["stderr_preview"] == "warning:\tSECRET=<redacted>\nnext line"
    assert "\n" in result_fields["stdout_preview"]
    assert "NAME     READY" in result_fields["stdout_preview"]
    assert "secret-token" not in json.dumps(result_fields, ensure_ascii=False)
    assert "stderr-secret" not in json.dumps(result_fields, ensure_ascii=False)

    os.environ["APISERVER_USER_TOKEN"] = "ops-token-secret"
    try:
        result_fields = _safe_tool_result_fields(
            {
                "output": "token value: ops-token-secret",
                "stderr": "authorization failed for ops-token-secret",
                "exit_code": 1,
                "error": "ops-token-secret",
            }
        )
    finally:
        os.environ.pop("APISERVER_USER_TOKEN", None)
    encoded = json.dumps(result_fields, ensure_ascii=False)
    assert "ops-token-secret" not in encoded
    assert encoded.count("<redacted>") >= 2

    approval_fields = _approval_display_fields(
        {
            "command": "bash -c 'cat /secret/path && kubectl get pods'",
            "description": "shell command via -c/-lc flag",
        }
    )
    assert approval_fields["tool_name"] == "terminal"
    assert approval_fields["risk"] == "requires_approval"
    assert approval_fields["approval_reason"] == "shell command via -c/-lc flag"
    assert approval_fields["summary"] == "Run shell command that requires approval"
    assert approval_fields["reason"] == "Run shell command that requires approval"
    assert "/secret/path" not in json.dumps(approval_fields, ensure_ascii=False)


def test_runtime_worker_long_success_stdout_becomes_warning_without_auto_artifact(tmp_path, monkeypatch):
    from runtime_manager.worker_main import _safe_tool_result_fields

    hermes_home = tmp_path / "user-1"
    artifact_dir = hermes_home / "sessions" / "session_1.artifacts" / "run_1"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_ARTIFACT_DIR", str(artifact_dir))
    stdout = "<html><body>\n" + ("AWR report row\n" * 160) + "</body></html>\n"

    result_fields = _safe_tool_result_fields(
        {
            "output": stdout,
            "exit_code": 0,
            "error": "output too long: original output hidden",
        },
        run_id="run/1",
        session_id="../session 1",
        tool_call_id="tool/1",
        tool_name="terminal",
    )

    assert result_fields["error"] is False
    assert result_fields["exit_code"] == 0
    assert result_fields["truncated"] is True
    assert result_fields["warningReason"] == "output_truncated"
    assert result_fields["output_bytes"] == len(stdout.encode("utf-8"))
    assert result_fields["stdout_preview"].startswith("<html><body>")
    assert "stderr_preview" not in result_fields
    assert "AWR report row\n" * 120 not in json.dumps(result_fields, ensure_ascii=False)
    assert "artifact" not in result_fields
    assert "artifactUnavailableReason" not in result_fields
    assert not any(artifact_dir.rglob("*"))


def test_runtime_worker_prefers_terminal_artifact_metadata_over_rearchiving(tmp_path, monkeypatch):
    from runtime_manager.worker_main import _safe_tool_result_fields

    hermes_home = tmp_path / "user-1"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    preview = "head\n... [OUTPUT TRUNCATED - 100 chars omitted out of 200 total] ...\ntail"
    terminal_artifact = {
        "artifactId": "terminal-call-abc123",
        "fileName": "terminal-call-abc123.txt",
        "sizeBytes": 200,
        "mimeType": "text/plain; charset=utf-8",
        "sha256": "f" * 64,
    }

    result_fields = _safe_tool_result_fields(
        {
            "output": preview,
            "exit_code": 0,
            "error": None,
            "truncated": True,
            "output_bytes": 200,
            "artifact": terminal_artifact,
            "artifactUnavailableReason": "output_artifact_unavailable",
        },
        run_id="run-1",
        session_id="session-1",
        tool_call_id="tool-1",
        tool_name="terminal",
    )

    assert result_fields["error"] is False
    assert result_fields["truncated"] is True
    assert result_fields["warningReason"] == "output_truncated"
    assert result_fields["output_bytes"] == 200
    assert result_fields["artifact"] == terminal_artifact
    assert result_fields["artifacts"] == [terminal_artifact]
    assert "artifactUnavailableReason" not in result_fields
    assert not (hermes_home / "sessions").exists()


def test_runtime_manager_session_cleanup_removes_output_artifacts(tmp_path):
    from runtime_manager.manager import _remove_session_files

    sessions_dir = tmp_path / "sessions"
    artifacts_dir = sessions_dir / "conv-1.artifacts" / "run-1"
    artifacts_dir.mkdir(parents=True)
    (sessions_dir / "conv-1.json").write_text("{}", encoding="utf-8")
    (artifacts_dir / "terminal-tool-output.txt").write_text("full output", encoding="utf-8")

    assert _remove_session_files(sessions_dir, "conv-1")
    assert not (sessions_dir / "conv-1.json").exists()
    assert not (sessions_dir / "conv-1.artifacts").exists()


def test_session_db_cleanup_removes_output_artifacts(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    artifacts_dir = sessions_dir / "conv-1.artifacts" / "run-1"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "report.html").write_text("<html></html>", encoding="utf-8")

    session_id = db.create_session("conv-1", "test")
    assert session_id == "conv-1"
    assert db.delete_session("conv-1", sessions_dir=sessions_dir)
    assert not (sessions_dir / "conv-1.artifacts").exists()


def test_runtime_worker_long_nonzero_exit_remains_error(tmp_path, monkeypatch):
    from runtime_manager.worker_main import _safe_tool_result_fields

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "user-1"))
    result_fields = _safe_tool_result_fields(
        {
            "stdout": "still useful\n" * 200,
            "exitCode": 1,
            "error": "command failed",
        },
        run_id="run-1",
        session_id="session-1",
        tool_call_id="tool-1",
        tool_name="terminal",
    )

    assert result_fields["error"] is True
    assert result_fields["exit_code"] == 1
    assert result_fields["stderr_preview"] == "command failed"
    assert "warningReason" not in result_fields
    assert "artifact" not in result_fields


def test_runtime_worker_output_truncation_without_stdout_remains_error(tmp_path, monkeypatch):
    from runtime_manager.worker_main import _safe_tool_result_fields

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "user-1"))
    result_fields = _safe_tool_result_fields(
        {
            "output": "",
            "exit_code": 0,
            "error": "output too long: preview unavailable",
        },
        run_id="run-1",
        session_id="session-1",
        tool_call_id="tool-1",
        tool_name="terminal",
    )

    assert result_fields["error"] is True
    assert result_fields["stderr_preview"] == "output too long: preview unavailable"
    assert "warningReason" not in result_fields
    assert "artifact" not in result_fields


def test_runtime_worker_summarizes_stringified_terminal_arguments():
    from runtime_manager.worker_main import _summarize_previous_tools

    summary = _summarize_previous_tools(
        [
            {
                "name": "terminal",
                "arguments": "{command: 'kubectl get events -n kb-cloud'}",
                "result": "event output",
            },
            {
                "name": "terminal",
                "arguments": '{"command":"kubectl describe pvc data-demo-0 -n kb-cloud"}',
                "result": "pvc output",
            },
        ]
    )

    assert summary[0]["action_summary"] == "Inspect Kubernetes events with kubectl get"
    assert summary[1]["action_summary"] == "Inspect Kubernetes pvc with kubectl describe"
    serialized = json.dumps(summary, ensure_ascii=False)
    assert "Run terminal command with" not in serialized
    assert "KUBECONFIG" not in serialized


def test_runtime_manager_requires_api_key_unless_explicitly_allowed(tmp_path, monkeypatch):
    from runtime_manager.manager import RuntimeManager

    monkeypatch.delenv("RUNTIME_MANAGER_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("RUNTIME_MANAGER_INSECURE_ALLOW_UNAUTHENTICATED", raising=False)
    manager = RuntimeManager(users_root=tmp_path / "users", api_key="")
    assert not manager.authorize(None)

    monkeypatch.setenv("RUNTIME_MANAGER_INSECURE_ALLOW_UNAUTHENTICATED", "true")
    manager = RuntimeManager(users_root=tmp_path / "users", api_key="")
    assert manager.authorize(None)


@pytest.mark.asyncio
async def test_runtime_manager_runs_fake_worker_approval_flow(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.running', 'run_id': run_id, 'timestamp': time.time()}), flush=True)",
                "print(json.dumps({'event': 'approval.request', 'run_id': run_id, 'timestamp': time.time(), 'choices': ['once', 'session', 'always', 'deny']}), flush=True)",
                "for line in sys.stdin:",
                "    cmd = json.loads(line)",
                "    if cmd.get('type') == 'approval':",
                "        print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': cmd.get('choice'), 'usage': {}}), flush=True)",
                "        break",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "waiting_for_approval":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "waiting_for_approval"

    await manager.approve_run(handle.run_id, choice="once")

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "completed"
    assert handle.output == "once"
    assert (tmp_path / "users" / "user-1" / "sessions").is_dir()
    assert (tmp_path / "users" / "user-1" / "home").is_dir()


@pytest.mark.asyncio
async def test_runtime_manager_fails_run_when_worker_stdout_line_exceeds_limit(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.running', 'run_id': run_id, 'timestamp': time.time()}), flush=True)",
                "print(json.dumps({'event': 'agent.step', 'run_id': run_id, 'timestamp': time.time(), 'previous_tools': [{'name': 'terminal', 'result': 'x' * (80 * 1024)}]}), flush=True)",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )
    try:
        for _ in range(100):
            if handle.status == "failed":
                break
            await asyncio.sleep(0.02)
        assert handle.status == "failed"
        assert "stdout" in (handle.error or "")
    finally:
        proc = handle.process
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.asyncio
async def test_runtime_manager_forwards_per_run_llm_config_to_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'model': req.get('model'), 'provider': req.get('provider'), 'base_url': req.get('base_url'), 'api_key': req.get('api_key')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4.1",
                "base_url": "https://models.example/v1",
                "api_key": "sk-test",
            },
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    assert handle.model == "gpt-4.1"
    output = json.loads(handle.output)
    assert output == {
        "model": "gpt-4.1",
        "provider": "openai",
        "base_url": "https://models.example/v1",
        "api_key": "sk-test",
    }


@pytest.mark.asyncio
async def test_runtime_manager_defaults_worker_toolsets_to_terminal_file_and_skills(
    tmp_path, monkeypatch
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'enabled_toolsets': req.get('enabled_toolsets'), 'disabled_toolsets': req.get('disabled_toolsets'), 'max_iterations': req.get('max_iterations')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_ENABLED_TOOLSETS", raising=False)
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {
        "enabled_toolsets": ["terminal", "file", "skills"],
        "disabled_toolsets": None,
        "max_iterations": 20,
    }


@pytest.mark.asyncio
async def test_runtime_manager_ignores_legacy_default_enabled_toolsets_env(
    tmp_path, monkeypatch
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'enabled_toolsets': req.get('enabled_toolsets')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.setenv("RUNTIME_MANAGER_DEFAULT_ENABLED_TOOLSETS", "terminal,file")
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {"enabled_toolsets": ["terminal", "file", "skills"]}


@pytest.mark.asyncio
async def test_runtime_manager_sets_worker_profile_env_to_user_workspace(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, os, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'hermes_home': os.environ.get('HERMES_HOME'), 'home': os.environ.get('HOME'), 'terminal_cwd': os.environ.get('TERMINAL_CWD')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    user_home = tmp_path / "users" / "user-1"
    assert json.loads(handle.output) == {
        "hermes_home": str(user_home),
        "home": str(user_home / "home"),
        "terminal_cwd": str(user_home / "workspace"),
    }


@pytest.mark.asyncio
async def test_runtime_manager_forces_tirith_disabled_for_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, os, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'tirith_enabled': os.environ.get('TIRITH_ENABLED')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.setenv("TIRITH_ENABLED", "true")
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    assert json.loads(handle.output) == {"tirith_enabled": "false"}


@pytest.mark.asyncio
async def test_runtime_manager_forwards_apiserver_ops_env_to_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, os, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'user_type': os.environ.get('APISERVER_USER_TYPE'), 'endpoint': os.environ.get('APISERVER_OPS_ENDPOINT'), 'token': os.environ.get('APISERVER_USER_TOKEN'), 'unsafe': os.environ.get('OPENAI_API_KEY'), 'request_ops_env': req.get('ops_env')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
            "opsEnv": {
                "APISERVER_USER_TYPE": "frontend",
                "APISERVER_OPS_ENDPOINT": "http://apiserver.internal",
                "APISERVER_USER_TOKEN": "ops-token",
                "OPENAI_API_KEY": "must-not-forward",
            },
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    assert json.loads(handle.output) == {
        "user_type": "frontend",
        "endpoint": "http://apiserver.internal",
        "token": "ops-token",
        "unsafe": None,
        "request_ops_env": {
            "APISERVER_USER_TYPE": "frontend",
            "APISERVER_OPS_ENDPOINT": "http://apiserver.internal",
            "APISERVER_USER_TOKEN": "ops-token",
        },
    }


@pytest.mark.asyncio
async def test_runtime_manager_does_not_inject_prompt_or_skills_without_default_profile(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'system_prompt': req.get('system_prompt'), 'skills': req.get('skills')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_SYSTEM_PROMPT_FILE", raising=False)
    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_PROFILE_DIR", raising=False)
    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_SKILLS", raising=False)
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
            "system_prompt": "Current message context: cluster=a",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {
        "system_prompt": "Current message context: cluster=a",
        "skills": [],
    }
    assert not (tmp_path / "users" / "user-1" / "asset-version.json").exists()


@pytest.mark.asyncio
async def test_runtime_manager_ignores_legacy_default_skills_env(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'skills': req.get('skills')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.setenv("RUNTIME_MANAGER_DEFAULT_SKILLS", "dba-diagnose-redis")
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    assert json.loads(handle.output) == {"skills": []}


def test_worker_composes_system_prompt_with_preloaded_skills():
    from runtime_manager.worker_main import _compose_effective_system_prompt

    def fake_skill_loader(skills, task_id=None):
        assert skills == ["kubeblocks-k8s-diagnosis"]
        assert task_id == "conv-1"
        return "SKILL PROMPT", ["kubeblocks-k8s-diagnosis"], []

    prompt = _compose_effective_system_prompt(
        {
            "system_prompt": "BASE PROMPT",
            "skills": ["kubeblocks-k8s-diagnosis"],
        },
        session_id="conv-1",
        skill_prompt_builder=fake_skill_loader,
    )

    assert prompt == "BASE PROMPT\n\nSKILL PROMPT"


def test_worker_fails_fast_when_requested_skill_is_missing():
    from runtime_manager.worker_main import _compose_effective_system_prompt

    def fake_skill_loader(skills, task_id=None):
        return "", [], ["missing-skill"]

    with pytest.raises(RuntimeError, match="missing Runtime Manager skills"):
        _compose_effective_system_prompt(
            {"skills": ["missing-skill"]},
            session_id="conv-1",
            skill_prompt_builder=fake_skill_loader,
        )


@pytest.mark.asyncio
async def test_runtime_manager_copies_default_profile_assets_without_preloading_manifest_skills(
    tmp_path, monkeypatch
):
    assets = tmp_path / "cloud-assets"
    (assets / "skills" / "custom-diagnosis").mkdir(parents=True)
    (assets / "system-prompt.md").write_text("CLOUD MAINTAINED PROMPT", encoding="utf-8")
    (assets / "manifest.yaml").write_text(
        "\n".join(
            [
                "version: v9",
                "systemPrompt: system-prompt.md",
                "skills:",
                "  - path: skills/custom-diagnosis",
                "    name: custom-diagnosis",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    (assets / "skills" / "custom-diagnosis" / "SKILL.md").write_text(
        "---\nname: custom-diagnosis\n---\n# Custom diagnosis skill\n",
        encoding="utf-8",
    )

    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'system_prompt': req.get('system_prompt'), 'skills': req.get('skills')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.setenv("RUNTIME_MANAGER_DEFAULT_PROFILE_DIR", str(assets))
    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_SKILLS", raising=False)
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {
        "system_prompt": "CLOUD MAINTAINED PROMPT",
        "skills": [],
    }
    assert (
        tmp_path
        / "users"
        / "user-1"
        / "skills"
        / "custom-diagnosis"
        / "SKILL.md"
    ).is_file()
    assert not (tmp_path / "users" / "user-1" / "asset-version.json").exists()


@pytest.mark.asyncio
async def test_runtime_manager_syncs_nested_skill_assets_without_preloading_manifest_skills(
    tmp_path, monkeypatch
):
    assets = tmp_path / "cloud-assets"
    skill_dir = assets / "skills" / "diagnosis" / "custom-diagnosis"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: custom-diagnosis\n---\n# Custom diagnosis skill\n",
        encoding="utf-8",
    )
    (skill_dir / "references" / "runbook.md").write_text(
        "# Runbook\nUse kubectl describe first.\n",
        encoding="utf-8",
    )
    (assets / "manifest.yaml").write_text(
        "\n".join(
            [
                "skills:",
                "  - path: skills/diagnosis/custom-diagnosis",
                "    name: custom-diagnosis",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'skills': req.get('skills')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.setenv("RUNTIME_MANAGER_DEFAULT_PROFILE_DIR", str(assets))
    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_SKILLS", raising=False)
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    assert json.loads(handle.output) == {"skills": []}
    copied_skill_dir = (
        tmp_path
        / "users"
        / "user-1"
        / "skills"
        / "diagnosis"
        / "custom-diagnosis"
    )
    assert (copied_skill_dir / "SKILL.md").is_file()
    assert (copied_skill_dir / "references" / "runbook.md").read_text(
        encoding="utf-8"
    ) == "# Runbook\nUse kubectl describe first.\n"


@pytest.mark.asyncio
async def test_runtime_manager_payload_toolsets_override_default(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'enabled_toolsets': req.get('enabled_toolsets'), 'disabled_toolsets': req.get('disabled_toolsets'), 'max_iterations': req.get('max_iterations')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import json
    import sys

    monkeypatch.delenv("RUNTIME_MANAGER_DEFAULT_ENABLED_TOOLSETS", raising=False)
    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
            "enabled_toolsets": ["web"],
            "disabled_toolsets": "browser,tts",
            "max_iterations": 7,
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {
        "enabled_toolsets": ["web"],
        "disabled_toolsets": ["browser", "tts"],
        "max_iterations": 7,
    }


@pytest.mark.asyncio
async def test_runtime_manager_rejects_approval_when_not_waiting(tmp_path):
    from runtime_manager.manager import RuntimeManager

    manager = RuntimeManager(users_root=tmp_path / "users")
    handle = manager.registry.create(
        run_id="run-1",
        user_id="user-1",
        conversation_id="conv-1",
        session_id="conv-1",
        model="openai/test",
    )
    handle.publish({"event": "run.running", "run_id": "run-1", "timestamp": 1.0})

    with pytest.raises(RuntimeError):
        await manager.approve_run("run-1", choice="once")


@pytest.mark.asyncio
async def test_runtime_manager_rejects_second_active_conversation_run(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.running', 'run_id': run_id, 'timestamp': time.time()}), flush=True)",
                "for _ in sys.stdin:",
                "    break",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    with pytest.raises(RuntimeError):
        await manager.start_run(
            {
                "user_id": "user-1",
                "conversation_id": "conv-1",
                "message": "again",
                "model": "openai/test",
            }
        )

    await manager.stop_run(handle.run_id)
    for _ in range(100):
        if handle.status == "cancelled":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "cancelled"


def test_run_registry_caps_replay_events_and_prunes_terminal_runs():
    registry = RunRegistry(max_events_per_run=2, completed_run_ttl_seconds=10)
    handle = registry.create(
        run_id="run-1",
        user_id="user-1",
        conversation_id="conv-1",
        session_id="conv-1",
        model="openai/test",
    )
    handle.publish({"event": "run.running", "run_id": "run-1", "timestamp": 1.0})
    handle.publish({"event": "message.delta", "run_id": "run-1", "timestamp": 2.0, "delta": "a"})
    handle.publish({"event": "run.completed", "run_id": "run-1", "timestamp": 3.0, "output": "done"})

    assert [event["event"] for event in handle.events] == ["message.delta", "run.completed"]
    assert [event["event_id"] for event in handle.events] == [2, 3]
    assert registry.prune(now=20.0) == 1
    assert registry.get("run-1") is None


def test_profile_environment_bridges_home_and_config(tmp_path, monkeypatch):
    from runtime_manager.bootstrap import load_profile_environment

    hermes_home = tmp_path / "user-1"
    (hermes_home / "home").mkdir(parents=True)
    (hermes_home / ".env").write_text("TERMINAL_TIMEOUT=12\n", encoding="utf-8")
    (hermes_home / "config.yaml").write_text(
        "\n".join(
            [
                "terminal:",
                "  backend: local",
                f"  cwd: {hermes_home / 'workspace'}",
                "auxiliary:",
                "  approval:",
                "    provider: openai",
                "    model: gpt-4.1",
                "agent:",
                "  max_turns: 7",
                "timezone: Asia/Shanghai",
                "security:",
                "  redact_secrets: true",
            ]
        ),
        encoding="utf-8",
    )

    for key in (
        "HERMES_HOME",
        "HOME",
        "TERMINAL_ENV",
        "TERMINAL_CWD",
        "TERMINAL_TIMEOUT",
        "AUXILIARY_APPROVAL_PROVIDER",
        "AUXILIARY_APPROVAL_MODEL",
        "HERMES_MAX_ITERATIONS",
        "HERMES_TIMEZONE",
        "HERMES_REDACT_SECRETS",
    ):
        monkeypatch.delenv(key, raising=False)

    load_profile_environment(hermes_home)

    assert os.environ["HERMES_HOME"] == str(hermes_home.resolve())
    assert os.environ["HOME"] == str((hermes_home / "home").resolve())
    assert os.environ["TERMINAL_ENV"] == "local"
    assert os.environ["TERMINAL_CWD"] == str(hermes_home / "workspace")
    assert os.environ["TERMINAL_TIMEOUT"] == "12"
    assert os.environ["AUXILIARY_APPROVAL_PROVIDER"] == "openai"
    assert os.environ["AUXILIARY_APPROVAL_MODEL"] == "gpt-4.1"
    assert os.environ["HERMES_MAX_ITERATIONS"] == "7"
    assert os.environ["HERMES_TIMEZONE"] == "Asia/Shanghai"
    assert os.environ["HERMES_REDACT_SECRETS"] == "true"


def test_profile_environment_defaults_terminal_cwd_to_workspace(tmp_path, monkeypatch):
    from runtime_manager.bootstrap import load_profile_environment

    hermes_home = tmp_path / "user-1"
    for key in ("HERMES_HOME", "HOME", "TERMINAL_CWD"):
        monkeypatch.delenv(key, raising=False)

    load_profile_environment(hermes_home)

    assert os.environ["HERMES_HOME"] == str(hermes_home.resolve())
    assert os.environ["HOME"] == str((hermes_home / "home").resolve())
    assert os.environ["TERMINAL_CWD"] == str((hermes_home / "workspace").resolve())
    assert (hermes_home / "workspace").is_dir()


def test_profile_environment_does_not_allow_dotenv_to_escape_profile(tmp_path, monkeypatch):
    from runtime_manager.bootstrap import load_profile_environment

    hermes_home = tmp_path / "user-1"
    outside = tmp_path / "outside"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "HERMES_HOME=/tmp/escape",
                "HOME=/tmp/escape-home",
                f"TERMINAL_CWD={outside}",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("HERMES_HOME", "HOME", "TERMINAL_CWD"):
        monkeypatch.delenv(key, raising=False)

    load_profile_environment(hermes_home)

    assert os.environ["HERMES_HOME"] == str(hermes_home.resolve())
    assert os.environ["HOME"] == str((hermes_home / "home").resolve())
    assert os.environ["TERMINAL_CWD"] == str((hermes_home / "workspace").resolve())
    assert not outside.exists()


def test_profile_environment_rejects_terminal_cwd_outside_home(tmp_path, monkeypatch):
    from runtime_manager.bootstrap import load_profile_environment

    hermes_home = tmp_path / "user-1"
    outside = tmp_path / "outside"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "\n".join(
            [
                "terminal:",
                f"  cwd: {outside}",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("HERMES_HOME", "HOME", "TERMINAL_CWD"):
        monkeypatch.delenv(key, raising=False)

    load_profile_environment(hermes_home)

    assert os.environ["TERMINAL_CWD"] == str((hermes_home / "workspace").resolve())
    assert not outside.exists()


@pytest.mark.asyncio
async def test_sse_stream_uses_event_ids_and_last_event_id_cursor():
    from runtime_manager.app import _sse_stream

    registry = RunRegistry()
    handle = registry.create(
        run_id="run-1",
        user_id="user-1",
        conversation_id="conv-1",
        session_id="conv-1",
        model="openai/test",
    )
    handle.publish({"event": "run.running", "run_id": "run-1", "timestamp": 1.0})
    handle.publish({"event": "run.completed", "run_id": "run-1", "timestamp": 2.0, "output": "done"})

    stream = _sse_stream(handle, last_event_id=1)
    try:
        chunk = await asyncio.wait_for(stream.__anext__(), timeout=1)
    finally:
        await stream.aclose()

    assert chunk.startswith("id: 2\n")
    assert '"event": "run.completed"' in chunk
    assert '"event_id": 2' in chunk


@pytest.mark.asyncio
async def test_runtime_manager_stop_transitions_to_cancelled(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.running', 'run_id': run_id, 'timestamp': time.time()}), flush=True)",
                "for line in sys.stdin:",
                "    cmd = json.loads(line)",
                "    if cmd.get('type') == 'stop':",
                "        break",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import sys

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
        }
    )

    for _ in range(100):
        if handle.status == "running":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "running"

    await manager.stop_run(handle.run_id)
    for _ in range(100):
        if handle.status == "cancelled":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "cancelled"
