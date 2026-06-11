import asyncio
import hashlib
import json
import sys

from fastapi.testclient import TestClient
import pytest

from runtime_manager.app import create_app


def test_report_artifact_metadata_is_discovered_from_controlled_dir(tmp_path):
    from runtime_manager.worker_main import _discover_report_artifacts_from_dir

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    report = artifact_dir / "awr report.html"
    content = b"<html><body>AWR report</body></html>"
    report.write_bytes(content)

    artifacts = _discover_report_artifacts_from_dir(artifact_dir, seen_artifact_ids=set())

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["fileName"] == "awr report.html"
    assert artifact["sizeBytes"] == len(content)
    assert artifact["mimeType"] == "text/html; charset=utf-8"
    assert artifact["sha256"] == hashlib.sha256(content).hexdigest()
    assert artifact["kind"] == "report"
    assert artifact["source"] == "tool"
    assert artifact["canDownload"] is True
    assert artifact["canBrowse"] is True
    assert "path" not in json.dumps(artifact, ensure_ascii=False).lower()


def test_report_artifact_snapshot_updates_same_path_without_new_asset(tmp_path):
    from runtime_manager.artifacts import discover_artifacts, find_artifact_file

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    report = artifact_dir / "analysis.txt"
    report.write_text("first report", encoding="utf-8")

    first = discover_artifacts(artifact_dir, seen_artifact_ids=set())[0]
    first_id = first["artifactId"]
    report.write_text("second report", encoding="utf-8")
    second = discover_artifacts(artifact_dir, seen_artifact_ids={first_id})

    assert second == []
    first_path, first_metadata = find_artifact_file(artifact_dir, first_id)
    assert first_path.read_text(encoding="utf-8") == "second report"
    assert first_metadata["artifactId"] == first_id
    assert first_metadata["sha256"] == hashlib.sha256(b"second report").hexdigest()


def test_report_artifact_discovery_ignores_published_snapshots(tmp_path):
    from runtime_manager.artifacts import discover_artifacts

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    report = artifact_dir / "analysis.txt"
    report.write_text("first report", encoding="utf-8")
    discover_artifacts(artifact_dir, seen_artifact_ids=set())

    discovered = discover_artifacts(artifact_dir, seen_artifact_ids=set())

    assert [artifact["fileName"] for artifact in discovered] == ["analysis.txt"]


def test_runtime_worker_adds_report_artifact_without_stdout_truncation(tmp_path, monkeypatch):
    from runtime_manager.worker_main import _safe_tool_result_fields

    artifact_dir = tmp_path / "run-artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "diagnosis.txt").write_text("full report", encoding="utf-8")
    monkeypatch.setenv("HERMES_ARTIFACT_DIR", str(artifact_dir))

    fields = _safe_tool_result_fields(
        {"output": "report generated\n", "exit_code": 0, "error": None},
        run_id="run-1",
        session_id="conv-1",
        tool_call_id="tool-1",
        tool_name="terminal",
    )
    assert fields["error"] is False
    assert "warningReason" not in fields
    assert fields["artifact"]["fileName"] == "diagnosis.txt"
    assert fields["artifact"]["kind"] == "report"


def test_runtime_manager_serves_report_artifact_by_user_session_run(tmp_path):
    from runtime_manager.artifacts import metadata_for_artifact_file

    user_home = tmp_path / "user-1"
    artifact_dir = user_home / "sessions" / "conv-1.artifacts" / "run-1"
    artifact_dir.mkdir(parents=True)
    report = artifact_dir / "awr.html"
    content = b"<html><body>AWR</body></html>"
    report.write_bytes(content)
    metadata = metadata_for_artifact_file(report, artifact_dir)

    app = create_app(users_root=tmp_path, api_key="secret")
    client = TestClient(app)

    response = client.get(
        f"/agent/sessions/conv-1/artifacts/{metadata['artifactId']}",
        params={"user_id": "user-1", "run_id": "run-1"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"].startswith("text/html")
    assert "attachment" in response.headers["content-disposition"]
    assert "awr.html" in response.headers["content-disposition"]


def test_runtime_manager_serves_latest_published_report_artifact_after_source_overwrite(tmp_path):
    from runtime_manager.artifacts import discover_artifacts

    user_home = tmp_path / "user-1"
    artifact_dir = user_home / "sessions" / "conv-1.artifacts" / "run-1"
    artifact_dir.mkdir(parents=True)
    report = artifact_dir / "analysis.txt"
    report.write_text("first report", encoding="utf-8")
    metadata = discover_artifacts(artifact_dir, seen_artifact_ids=set())[0]
    report.write_text("second report", encoding="utf-8")
    assert discover_artifacts(artifact_dir, seen_artifact_ids={metadata["artifactId"]}) == []

    app = create_app(users_root=tmp_path, api_key="secret")
    client = TestClient(app)

    response = client.get(
        f"/agent/sessions/conv-1/artifacts/{metadata['artifactId']}",
        params={"user_id": "user-1", "run_id": "run-1"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.content == b"second report"


def test_runtime_manager_rejects_artifact_path_escape(tmp_path):
    from runtime_manager.artifacts import metadata_for_artifact_file

    user_home = tmp_path / "user-1"
    artifact_dir = user_home / "sessions" / "conv-1.artifacts" / "run-1"
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside.html"
    outside.write_text("secret", encoding="utf-8")
    symlink = artifact_dir / "outside.html"
    symlink.symlink_to(outside)

    app = create_app(users_root=tmp_path, api_key="secret")
    client = TestClient(app)
    artifact_id = metadata_for_artifact_file(outside, tmp_path)["artifactId"]

    response = client.get(
        f"/agent/sessions/conv-1/artifacts/{artifact_id}",
        params={"user_id": "user-1", "run_id": "run-1"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 404


def test_file_tool_paths_can_use_hermes_artifact_dir_env(tmp_path, monkeypatch):
    from tools.file_tools import _resolve_path_for_task

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setenv("HERMES_ARTIFACT_DIR", str(artifact_dir))

    resolved = _resolve_path_for_task("$HERMES_ARTIFACT_DIR/report.md")

    assert resolved == artifact_dir / "report.md"


@pytest.mark.asyncio
async def test_runtime_manager_provisions_per_run_artifact_dir(tmp_path):
    from runtime_manager.manager import RuntimeManager

    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, os, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "artifact_dir = os.environ.get('HERMES_ARTIFACT_DIR')",
                "print(json.dumps({'event': 'run.completed', 'run_id': req['run_id'], 'timestamp': time.time(), 'output': json.dumps({'env_artifact_dir': artifact_dir, 'request_artifact_dir': req.get('artifact_dir')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "generate report",
        }
    )
    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)
    assert handle.status == "completed"
    output = json.loads(handle.output)
    artifact_dir = output["env_artifact_dir"]
    assert artifact_dir == output["request_artifact_dir"]
    assert artifact_dir.endswith(f"conv-1.artifacts/{handle.run_id}")
    assert (tmp_path / "users" / "user-1" / "sessions" / "conv-1.artifacts" / handle.run_id).is_dir()
