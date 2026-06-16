from __future__ import annotations

import os
from typing import Any


APISERVER_OPS_ENV_NAMES = (
    "APISERVER_USER_TYPE",
    "APISERVER_OPS_ENDPOINT",
    "APISERVER_USER_TOKEN",
)
APISERVER_OPS_SECRET_ENV_NAMES = frozenset({"APISERVER_USER_TOKEN"})

_OPS_ENV_CONTAINER_KEYS = (
    "ops_env",
    "opsEnv",
    "apiserver_ops_env",
    "apiserverOpsEnv",
)
_OPS_ENV_NESTED_KEYS = (
    "ops_context",
    "opsContext",
    "cluster_ops",
    "clusterOps",
    "apiserver_ops",
    "apiserverOps",
)


def extract_apiserver_ops_env(payload: dict[str, Any]) -> dict[str, str]:
    """Return the whitelisted APISERVER_* environment values for a run.

    Cloud owns the source of these values.  Runtime Manager only accepts a
    narrow allowlist and never forwards caller-supplied arbitrary environment
    variables into the worker.
    """

    if not isinstance(payload, dict):
        return {}

    merged: dict[str, Any] = {}
    for mapping in _iter_ops_env_mappings(payload):
        for name in APISERVER_OPS_ENV_NAMES:
            if name in mapping and (value := _normalize_env_value(mapping[name])):
                merged[name] = value
            for alternate in _alternate_payload_keys(name):
                if alternate in mapping and (value := _normalize_env_value(mapping[alternate])):
                    merged[name] = value

    return {
        name: value
        for name in APISERVER_OPS_ENV_NAMES
        if (value := _normalize_env_value(merged.get(name)))
    }


def apply_apiserver_ops_env(values: dict[str, Any]) -> dict[str, str]:
    """Apply whitelisted ops env values to os.environ and return what was set."""

    normalized = extract_apiserver_ops_env({"ops_env": values})
    for name, value in normalized.items():
        os.environ[name] = value
    return normalized


def runtime_secret_values_from_env() -> list[str]:
    values: list[str] = []
    for name in APISERVER_OPS_SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if isinstance(value, str) and value:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def _iter_ops_env_mappings(payload: dict[str, Any]):
    for key in _OPS_ENV_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            yield value

    for key in _OPS_ENV_NESTED_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            yield value

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in _OPS_ENV_CONTAINER_KEYS + _OPS_ENV_NESTED_KEYS:
            value = metadata.get(key)
            if isinstance(value, dict):
                yield value

    yield payload


def _normalize_env_value(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    text = str(value).strip()
    if not text or "\x00" in text:
        return ""
    return text


def _alternate_payload_keys(env_name: str) -> tuple[str, ...]:
    lowered = env_name.lower()
    if lowered == "apiserver_user_type":
        return ("apiserver_user_type", "apiserverUserType")
    if lowered == "apiserver_ops_endpoint":
        return ("apiserver_ops_endpoint", "apiserverOpsEndpoint")
    if lowered == "apiserver_user_token":
        return ("apiserver_user_token", "apiserverUserToken")
    return (lowered,)
