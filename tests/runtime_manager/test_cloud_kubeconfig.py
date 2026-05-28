import asyncio
import base64
import json
import stat
import subprocess
from dataclasses import dataclass

import pytest

from runtime_manager.cloud_kubeconfig import (
    CloudKubeconfigResolver,
    CloudMetaConfig,
    build_cluster_context_prompt,
    normalize_cluster_contexts,
    _run_command,
)


@dataclass
class FakeResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def test_normalize_cluster_contexts_accepts_cloud_context_shapes():
    contexts = normalize_cluster_contexts(
        [
            {
                "orgId": "org-1",
                "environmentName": "kb10",
                "clusterName": "orders",
                "namespace": "default",
            },
            {"org_id": "org-1", "environment_name": "kb10", "namespace": "default"},
            "kb11",
            {"missing": "ignored"},
        ]
    )

    assert [context.environment_name for context in contexts] == ["kb10", "kb11"]
    assert contexts[0].display_name == "orders"
    assert contexts[0].org_id == "org-1"
    assert contexts[0].namespace == "default"


def test_cloud_kubeconfig_resolver_queries_psql_and_writes_file_under_user_home(tmp_path):
    calls = []
    kubeconfig = "apiVersion: v1\nclusters:\n- name: kb10\n"
    encoded_kubeconfig = base64.b64encode(kubeconfig.encode("utf-8")).decode("ascii")

    def runner(cmd, *, timeout):
        calls.append((cmd, timeout))
        return FakeResult(stdout=encoded_kubeconfig)

    resolver = CloudKubeconfigResolver(
        CloudMetaConfig(
            namespace="kb-cloud",
            pg_pod_name="apecloud-pg-0",
            pg_container="postgres",
            database="kubeblockscloud",
            environment_table="admin_environment",
            kubectl="kubectl",
            query_timeout_seconds=9,
        ),
        runner=runner,
    )

    prepared = resolver.prepare_contexts(
        tmp_path / "user-home",
        [{"orgId": "org-1", "environmentName": "kb10", "clusterName": "orders"}],
    )

    assert len(prepared) == 1
    path = tmp_path / "user-home" / "kubeconfigs" / "org-1" / "kb10.yaml"
    assert prepared[0] == {
        "environment_name": "kb10",
        "display_name": "orders",
        "cluster_name": "orders",
        "namespace": "",
        "org_id": "org-1",
        "kubeconfig_path": str(path),
        "source": "cloud_meta_pg",
    }
    assert path.read_text(encoding="utf-8") == kubeconfig
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert calls == [
        (
            [
                "kubectl",
                "exec",
                "-n",
                "kb-cloud",
                "apecloud-pg-0",
                "-c",
                "postgres",
                "--",
                "psql",
                "-d",
                "kubeblockscloud",
                "-t",
                "-A",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                "select kubeconfig from admin_environment where name = 'kb10' and deleted_at = 0;",
            ],
            9,
        )
    ]


def test_cloud_kubeconfig_resolver_discovers_postgres_pod_by_selector(tmp_path):
    calls = []
    kubeconfig = base64.b64encode(b"apiVersion: v1\nclusters: []\n").decode("ascii")

    def runner(cmd, *, timeout):
        calls.append(cmd)
        if cmd[:3] == ["kubectl", "get", "pods"]:
            return FakeResult(stdout="apecloud-pg-0")
        return FakeResult(stdout=kubeconfig)

    resolver = CloudKubeconfigResolver(
        CloudMetaConfig(
            pg_pod_name="",
            pg_pod_selector="app=apecloud-pg",
        ),
        runner=runner,
    )

    prepared = resolver.prepare_contexts(tmp_path / "home", ["kb10"])

    assert prepared[0]["kubeconfig_path"].endswith("/kubeconfigs/kb10.yaml")
    assert calls[0] == [
        "kubectl",
        "get",
        "pods",
        "-n",
        "kb-cloud",
        "-l",
        "app=apecloud-pg",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ]
    assert calls[1][4] == "apecloud-pg-0"


def test_cloud_kubeconfig_resolver_rejects_unsafe_environment_names(tmp_path):
    resolver = CloudKubeconfigResolver(runner=lambda cmd, timeout: FakeResult(stdout=""))

    with pytest.raises(ValueError, match="invalid environment name"):
        resolver.prepare_contexts(tmp_path / "home", ["kb10'; drop table admin_environment; --"])


def test_cloud_kubeconfig_resolver_rejects_invalid_base64_kubeconfig(tmp_path):
    resolver = CloudKubeconfigResolver(runner=lambda cmd, timeout: FakeResult(stdout="not-base64!"))

    with pytest.raises(RuntimeError, match="invalid base64 kubeconfig"):
        resolver.prepare_contexts(tmp_path / "home", ["kb10"])


def test_cloud_kubeconfig_resolver_rejects_decoded_non_kubeconfig(tmp_path):
    encoded = base64.b64encode(b"hello").decode("ascii")
    resolver = CloudKubeconfigResolver(runner=lambda cmd, timeout: FakeResult(stdout=encoded))

    with pytest.raises(RuntimeError, match="does not look like a Kubernetes config"):
        resolver.prepare_contexts(tmp_path / "home", ["kb10"])


def test_run_command_converts_timeout_to_failed_result(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["kubectl"], timeout=3)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _run_command(["kubectl"], timeout=3)

    assert result.returncode == 124
    assert "timed out" in result.stderr


def test_cluster_context_prompt_contains_only_paths_not_kubeconfig_content():
    prompt = build_cluster_context_prompt(
        [
            {
                "display_name": "orders",
                "environment_name": "kb10",
                "namespace": "default",
                "kubeconfig_path": "/data/users/u1/kubeconfigs/org/kb10.yaml",
            }
        ]
    )

    assert "KUBECONFIG=/data/users/u1/kubeconfigs/org/kb10.yaml" in prompt
    assert "Do not print" in prompt
    assert "apiVersion" not in prompt


@pytest.mark.asyncio
async def test_runtime_manager_prefetches_context_kubeconfigs_and_injects_prompt(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "req = json.loads(sys.stdin.readline())",
                "run_id = req['run_id']",
                "print(json.dumps({'event': 'run.completed', 'run_id': run_id, 'timestamp': time.time(), 'output': json.dumps({'system_prompt': req.get('system_prompt'), 'cluster_contexts': req.get('cluster_contexts')})}), flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    from runtime_manager.manager import RuntimeManager
    import sys

    class FakeResolver:
        def prepare_contexts(self, user_home, raw_contexts):
            path = user_home / "kubeconfigs" / "kb10.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("apiVersion: v1\n", encoding="utf-8")
            return [
                {
                    "environment_name": "kb10",
                    "display_name": "orders",
                    "cluster_name": "orders",
                    "namespace": "default",
                    "org_id": "org-1",
                    "kubeconfig_path": str(path),
                    "source": "cloud_meta_pg",
                }
            ]

        def build_prompt(self, contexts):
            return build_cluster_context_prompt(contexts)

    manager = RuntimeManager(
        users_root=tmp_path / "users",
        python_executable=sys.executable,
        worker_script=worker,
        cloud_kubeconfig_resolver=FakeResolver(),
    )
    handle = await manager.start_run(
        {
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "message": "hello",
            "model": "openai/test",
            "contexts": [
                {
                    "orgId": "org-1",
                    "environmentName": "kb10",
                    "clusterName": "orders",
                    "namespace": "default",
                }
            ],
            "system_prompt": "Use selected clusters only.",
        }
    )

    for _ in range(100):
        if handle.status == "completed":
            break
        await asyncio.sleep(0.02)

    assert handle.status == "completed"
    output = json.loads(handle.output)
    assert output["cluster_contexts"][0]["environment_name"] == "kb10"
    assert "KUBECONFIG=" in output["system_prompt"]
    assert "Use selected clusters only." in output["system_prompt"]
    assert "apiVersion" not in output["system_prompt"]
