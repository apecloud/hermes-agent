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
