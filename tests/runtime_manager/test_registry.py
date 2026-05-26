import asyncio

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
    for name in ("sessions", "memories", "skills", "logs"):
        assert (home / name).is_dir()


def test_user_home_resolver_rejects_invalid_ids(tmp_path):
    resolver = UserHomeResolver(tmp_path)
    with pytest.raises(ValueError):
        resolver.resolve("../escape")


def test_runtime_manager_requires_api_key_unless_explicitly_allowed(tmp_path, monkeypatch):
    from runtime_manager.manager import RuntimeManager

    monkeypatch.delenv("RUNTIME_MANAGER_ALLOW_UNAUTHENTICATED", raising=False)
    manager = RuntimeManager(users_root=tmp_path / "users", api_key="")
    assert not manager.authorize(None)

    monkeypatch.setenv("RUNTIME_MANAGER_ALLOW_UNAUTHENTICATED", "true")
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
