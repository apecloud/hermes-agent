from __future__ import annotations

import base64
import binascii
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}$")
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")
_PATH_COMPONENT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.@:-]+")


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Any


@dataclass(frozen=True)
class CloudMetaConfig:
    namespace: str = "kb-cloud"
    pg_pod_name: str = "apecloud-pg-0"
    pg_pod_selector: str = ""
    pg_container: str = ""
    database: str = "kubeblockscloud"
    environment_table: str = "admin_environment"
    kubectl: str = "kubectl"
    query_timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "CloudMetaConfig":
        return cls(
            namespace=os.getenv("RUNTIME_MANAGER_CLOUD_META_NAMESPACE", "kb-cloud").strip() or "kb-cloud",
            pg_pod_name=os.getenv("RUNTIME_MANAGER_CLOUD_META_PG_POD_NAME", "apecloud-pg-0").strip(),
            pg_pod_selector=os.getenv("RUNTIME_MANAGER_CLOUD_META_PG_POD_SELECTOR", "").strip(),
            pg_container=os.getenv("RUNTIME_MANAGER_CLOUD_META_PG_CONTAINER", "").strip(),
            database=os.getenv("RUNTIME_MANAGER_CLOUD_META_PG_DATABASE", "kubeblockscloud").strip()
            or "kubeblockscloud",
            environment_table=os.getenv(
                "RUNTIME_MANAGER_CLOUD_META_ENVIRONMENT_TABLE",
                "admin_environment",
            ).strip()
            or "admin_environment",
            kubectl=os.getenv("RUNTIME_MANAGER_KUBECTL", "kubectl").strip() or "kubectl",
            query_timeout_seconds=float(
                os.getenv("RUNTIME_MANAGER_CLOUD_META_QUERY_TIMEOUT_SECONDS", "15")
            ),
        )


@dataclass(frozen=True)
class ClusterContextRequest:
    environment_name: str
    display_name: str
    org_id: str = ""
    namespace: str = ""
    cluster_name: str = ""


def normalize_cluster_contexts(raw_contexts: Any) -> list[ClusterContextRequest]:
    if raw_contexts is None:
        return []
    if isinstance(raw_contexts, dict):
        for key in ("contexts", "clusters", "cluster_contexts", "clusterContexts"):
            nested = raw_contexts.get(key)
            if isinstance(nested, list):
                raw_contexts = nested
                break
        else:
            raw_contexts = [raw_contexts]
    if isinstance(raw_contexts, str):
        raw_contexts = [raw_contexts]
    if not isinstance(raw_contexts, list):
        return []

    normalized: list[ClusterContextRequest] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_contexts:
        context = _normalize_one_context(item)
        if context is None:
            continue
        key = (context.org_id, context.environment_name, context.namespace)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(context)
    return normalized


def build_cluster_context_prompt(contexts: list[dict[str, str]]) -> str | None:
    if not contexts:
        return None
    lines = [
        "Kubernetes contexts prepared for this message:",
        (
            "Use the exact KUBECONFIG path below when running kubectl or kbcli. "
            "Do not print, cat, copy, summarize, or expose kubeconfig file contents."
        ),
    ]
    for context in contexts:
        label = context.get("display_name") or context.get("environment_name") or "cluster"
        environment_name = context.get("environment_name") or ""
        kubeconfig_path = context.get("kubeconfig_path") or ""
        namespace = context.get("namespace") or ""
        namespace_suffix = f", namespace={namespace}" if namespace else ""
        lines.append(
            f"- {label}: environment={environment_name}{namespace_suffix}, "
            f"KUBECONFIG={kubeconfig_path}"
        )
    if len(contexts) > 1:
        lines.append("If the user intent does not identify a specific context, ask a clarifying question first.")
    return "\n".join(lines)


class CloudKubeconfigResolver:
    def __init__(
        self,
        config: CloudMetaConfig | None = None,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config or CloudMetaConfig.from_env()
        self.runner = runner or _run_command
        _validate_sql_identifier(self.config.environment_table, "environment table")

    def prepare_contexts(self, user_home: str | Path, raw_contexts: Any) -> list[dict[str, str]]:
        contexts = normalize_cluster_contexts(raw_contexts)
        if not contexts:
            return []
        home = Path(user_home)
        prepared: list[dict[str, str]] = []
        path_by_environment: dict[tuple[str, str], Path] = {}
        for context in contexts:
            cache_key = (context.org_id, context.environment_name)
            path = path_by_environment.get(cache_key)
            if path is None:
                kubeconfig = self._query_kubeconfig(context.environment_name)
                path = self._write_kubeconfig(home, context, kubeconfig)
                path_by_environment[cache_key] = path
            prepared.append(
                {
                    "environment_name": context.environment_name,
                    "display_name": context.display_name,
                    "cluster_name": context.cluster_name,
                    "namespace": context.namespace,
                    "org_id": context.org_id,
                    "kubeconfig_path": str(path),
                    "source": "cloud_meta_pg",
                }
            )
        return prepared

    def build_prompt(self, contexts: list[dict[str, str]]) -> str | None:
        return build_cluster_context_prompt(contexts)

    def _query_kubeconfig(self, environment_name: str) -> str:
        _validate_environment_name(environment_name)
        pod_name = self._resolve_postgres_pod()
        cmd = [
            self.config.kubectl,
            "exec",
            "-n",
            self.config.namespace,
            pod_name,
        ]
        if self.config.pg_container:
            cmd.extend(["-c", self.config.pg_container])
        cmd.extend(
            [
                "--",
                "psql",
                "-d",
                self.config.database,
                "-t",
                "-A",
                "-v",
                "ON_ERROR_STOP=1",
                "-v",
                f"env_name={environment_name}",
                "-c",
                (
                    f"select kubeconfig from {self.config.environment_table} "
                    "where name = :'env_name' and deleted_at = 0;"
                ),
            ]
        )
        result = self.runner(cmd, timeout=self.config.query_timeout_seconds)
        if result.returncode != 0:
            raise RuntimeError(_command_failure_message("query Cloud metadata kubeconfig", result))
        encoded_kubeconfig = str(result.stdout or "").strip()
        if not encoded_kubeconfig:
            raise RuntimeError(f"no kubeconfig found for environment {environment_name!r}")
        return _decode_kubeconfig(encoded_kubeconfig, environment_name)

    def _resolve_postgres_pod(self) -> str:
        if self.config.pg_pod_name:
            return self.config.pg_pod_name
        if not self.config.pg_pod_selector:
            raise RuntimeError("Cloud metadata postgres podName or podSelector is required")
        cmd = [
            self.config.kubectl,
            "get",
            "pods",
            "-n",
            self.config.namespace,
            "-l",
            self.config.pg_pod_selector,
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
        result = self.runner(cmd, timeout=self.config.query_timeout_seconds)
        if result.returncode != 0:
            raise RuntimeError(_command_failure_message("discover Cloud metadata postgres pod", result))
        pod_name = str(result.stdout or "").strip()
        if not pod_name:
            raise RuntimeError("Cloud metadata postgres pod selector did not match any pod")
        return pod_name

    def _write_kubeconfig(
        self,
        user_home: Path,
        context: ClusterContextRequest,
        kubeconfig: str,
    ) -> Path:
        base = user_home / "kubeconfigs"
        if context.org_id:
            base = base / _safe_path_component(context.org_id)
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = base / f"{_safe_path_component(context.environment_name)}.yaml"
        temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temp.write_text(kubeconfig.rstrip() + "\n", encoding="utf-8")
        temp.chmod(0o600)
        temp.replace(target)
        target.chmod(0o600)
        return target


def _normalize_one_context(item: Any) -> ClusterContextRequest | None:
    if isinstance(item, str):
        environment_name = item.strip()
        if not environment_name:
            return None
        _validate_environment_name(environment_name)
        return ClusterContextRequest(
            environment_name=environment_name,
            display_name=environment_name,
            cluster_name=environment_name,
        )
    if not isinstance(item, dict):
        return None
    environment_name = _first_string(
        item,
        "environment_name",
        "environmentName",
        "env_name",
        "envName",
        "name",
        "cluster_name",
        "clusterName",
    )
    if not environment_name:
        return None
    _validate_environment_name(environment_name)
    cluster_name = _first_string(item, "cluster_name", "clusterName", "name") or environment_name
    display_name = _first_string(item, "display_name", "displayName", "label") or cluster_name
    return ClusterContextRequest(
        environment_name=environment_name,
        display_name=display_name,
        org_id=_first_string(item, "org_id", "orgId", "organization", "organizationName") or "",
        namespace=_first_string(item, "namespace", "namespaceName") or "",
        cluster_name=cluster_name,
    )


def _first_string(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _validate_environment_name(value: str) -> None:
    if not _ENVIRONMENT_NAME_RE.fullmatch(value):
        raise ValueError(
            "invalid environment name; expected 1-128 chars of letters, digits, '.', '_', ':', '@', or '-'"
        )


def _validate_sql_identifier(value: str, label: str) -> None:
    if not _SQL_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")


def _decode_kubeconfig(encoded_value: str, environment_name: str) -> str:
    compact_value = "".join(encoded_value.split())
    try:
        decoded = base64.b64decode(compact_value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"invalid base64 kubeconfig for environment {environment_name!r}") from exc
    try:
        kubeconfig = decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"decoded kubeconfig is not UTF-8 for environment {environment_name!r}") from exc
    if "apiVersion:" not in kubeconfig or "clusters:" not in kubeconfig:
        raise RuntimeError(f"decoded kubeconfig does not look like a Kubernetes config for {environment_name!r}")
    return kubeconfig


def _safe_path_component(value: str) -> str:
    sanitized = _PATH_COMPONENT_SAFE_RE.sub("_", value).strip("._-")
    return sanitized or "context"


def _command_failure_message(action: str, result: CommandResult) -> str:
    stderr = str(getattr(result, "stderr", "") or "").strip()
    if not stderr:
        return f"failed to {action}"
    return f"failed to {action}: {stderr[:500]}"


def _run_command(cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"command timed out after {timeout:g}s",
        )
