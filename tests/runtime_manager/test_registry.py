import asyncio
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
    for name in ("home", "sessions", "memories", "skills", "logs"):
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


def test_runtime_worker_normalizes_cloud_provider_aliases_to_hermes_names():
    from runtime_manager.worker_main import _normalize_agent_provider

    assert _normalize_agent_provider("openai-compatible", base_url="https://models.example/v1") == "custom"
    assert _normalize_agent_provider("openai_compat", base_url="https://models.example/v1") == "custom"
    assert _normalize_agent_provider("openai", base_url="https://api.openai.com/v1") == "custom"
    assert _normalize_agent_provider("openai") == "openai-api"
    assert _normalize_agent_provider("qwen-oauth") == "qwen-oauth"


def test_runtime_worker_tool_event_helpers_are_json_safe():
    from runtime_manager.worker_main import _json_preview, _tool_result_has_error

    assert _json_preview({"tool": "kubectl", "args": ["get", "pods"]}) == (
        '{"tool": "kubectl", "args": ["get", "pods"]}'
    )
    assert _json_preview("x" * 520).endswith("...")
    assert _tool_result_has_error({"error": "boom"})
    assert _tool_result_has_error('{"error":"boom"}')
    assert _tool_result_has_error({"is_error": True})
    assert not _tool_result_has_error({"ok": True})
    assert not _tool_result_has_error("plain text result")


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

    for _ in range(50):
        if handle.status == "waiting_for_approval":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "waiting_for_approval"

    await manager.approve_run(handle.run_id, choice="once")

    for _ in range(50):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "completed"
    assert handle.output == "once"
    assert (tmp_path / "users" / "user-1" / "sessions").is_dir()
    assert (tmp_path / "users" / "user-1" / "home").is_dir()


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

    for _ in range(50):
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
async def test_runtime_manager_defaults_worker_toolsets_to_terminal_file(tmp_path, monkeypatch):
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

    for _ in range(50):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {
        "enabled_toolsets": ["terminal", "file"],
        "disabled_toolsets": None,
        "max_iterations": 20,
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

    for _ in range(50):
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
async def test_runtime_manager_uses_external_default_profile_assets(tmp_path, monkeypatch):
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

    for _ in range(50):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output == {
        "system_prompt": "CLOUD MAINTAINED PROMPT",
        "skills": ["custom-diagnosis"],
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
async def test_runtime_manager_syncs_full_nested_skill_directories(tmp_path, monkeypatch):
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

    for _ in range(50):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    assert json.loads(handle.output) == {"skills": ["custom-diagnosis"]}
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
            "enabled_toolsets": ["web"],
            "disabled_toolsets": "browser,tts",
            "max_iterations": 7,
        }
    )

    for _ in range(50):
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
    for _ in range(50):
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

    for _ in range(50):
        if handle.status == "running":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "running"

    await manager.stop_run(handle.run_id)
    for _ in range(50):
        if handle.status == "cancelled":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "cancelled"
