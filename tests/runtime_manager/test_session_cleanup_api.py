from __future__ import annotations

from fastapi.testclient import TestClient

from hermes_state import SessionDB
from runtime_manager.app import create_app


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-secret"}


def _seed_session(users_root, *, user_id: str, session_id: str) -> None:
    user_home = users_root / user_id
    sessions_dir = user_home / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    db = SessionDB(user_home / "state.db")
    try:
        db.create_session(session_id, "cloud", user_id=user_id)
        db.append_message(session_id, "user", "hello")
    finally:
        db.close()
    (sessions_dir / f"{session_id}.json").write_text("{}", encoding="utf-8")
    (sessions_dir / f"{session_id}.jsonl").write_text("{}\n", encoding="utf-8")
    (sessions_dir / f"request_dump_{session_id}_0001.json").write_text("{}", encoding="utf-8")


def _session_db(users_root, *, user_id: str) -> SessionDB:
    return SessionDB(users_root / user_id / "state.db")


def test_runtime_manager_deletes_exact_user_session_and_transcripts(tmp_path):
    users_root = tmp_path / "users"
    user_id = "cloud-user"
    session_id = "conv-123"
    _seed_session(users_root, user_id=user_id, session_id=session_id)

    app = create_app(users_root=users_root, api_key="test-secret")
    app.state.runtime_manager.manager.registry.create(
        run_id="run-completed",
        user_id=user_id,
        conversation_id=session_id,
        session_id=session_id,
        model="test-model",
    ).publish({"event": "run.completed", "run_id": "run-completed"})
    client = TestClient(app)

    response = client.delete(
        f"/agent/sessions/{session_id}",
        params={"user_id": user_id},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "object": "runtime_manager.session_cleanup",
        "user_id": user_id,
        "session_id": session_id,
        "deleted": True,
    }
    db = _session_db(users_root, user_id=user_id)
    try:
        assert db.get_session(session_id) is None
        assert db.get_messages(session_id) == []
    finally:
        db.close()
    sessions_dir = users_root / user_id / "sessions"
    assert not (sessions_dir / f"{session_id}.json").exists()
    assert not (sessions_dir / f"{session_id}.jsonl").exists()
    assert not list(sessions_dir.glob(f"request_dump_{session_id}_*.json"))
    assert app.state.runtime_manager.manager.registry.get("run-completed") is None

    second_response = client.delete(
        f"/agent/sessions/{session_id}",
        params={"user_id": user_id},
        headers=_auth_headers(),
    )

    assert second_response.status_code == 200
    assert second_response.json()["deleted"] is False


def test_runtime_manager_session_cleanup_requires_auth(tmp_path):
    app = create_app(users_root=tmp_path / "users", api_key="test-secret")
    client = TestClient(app)

    response = client.delete(
        "/agent/sessions/conv-123",
        params={"user_id": "cloud-user"},
    )

    assert response.status_code == 401


def test_runtime_manager_refuses_to_delete_active_session(tmp_path):
    users_root = tmp_path / "users"
    user_id = "cloud-user"
    session_id = "conv-active"
    _seed_session(users_root, user_id=user_id, session_id=session_id)

    app = create_app(users_root=users_root, api_key="test-secret")
    app.state.runtime_manager.manager.registry.create(
        run_id="run-active",
        user_id=user_id,
        conversation_id=session_id,
        session_id=session_id,
        model="test-model",
    ).publish({"event": "run.running", "run_id": "run-active"})
    client = TestClient(app)

    response = client.delete(
        f"/agent/sessions/{session_id}",
        params={"user_id": user_id},
        headers=_auth_headers(),
    )

    assert response.status_code == 409
    db = _session_db(users_root, user_id=user_id)
    try:
        assert db.get_session(session_id) is not None
        assert len(db.get_messages(session_id)) == 1
    finally:
        db.close()
