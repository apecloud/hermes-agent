from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _set_env_value(name: str, value: Any, *, override: bool = True) -> None:
    if not override and name in os.environ:
        return
    if isinstance(value, (list, dict)):
        os.environ[name] = json.dumps(value)
    else:
        os.environ[name] = str(value)


def _workspace_dir(home: Path) -> Path:
    return home / "workspace"


def _safe_terminal_cwd(home: Path, value: Any = None) -> Path:
    workspace = _workspace_dir(home)
    raw = str(value or "").strip()
    if not raw or raw in {".", "auto", "cwd"}:
        cwd = workspace
    else:
        expanded = Path(os.path.expanduser(os.path.expandvars(raw)))
        cwd = expanded if expanded.is_absolute() else home / expanded
    try:
        resolved = cwd.resolve()
        resolved.relative_to(home)
    except Exception:
        print(
            f"  Warning: Runtime Manager ignored terminal.cwd outside HERMES_HOME: {raw!r}",
            file=sys.stderr,
        )
        resolved = workspace.resolve()
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    return resolved


def load_profile_environment(hermes_home: str | os.PathLike[str]) -> None:
    """Load a Runtime Manager worker profile before importing agent modules.

    Hermes has several tools that read settings directly from environment
    variables. Gateway startup bridges documented config.yaml keys into those
    variables before creating an agent; Runtime Manager workers must do the
    same because each worker runs with a per-user HERMES_HOME.
    """

    home = Path(hermes_home).expanduser().resolve()
    home_home = home / "home"
    workspace = _workspace_dir(home)
    home_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HOME"] = str(home_home)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["TERMINAL_CWD"] = str(workspace.resolve())

    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(
            hermes_home=home,
            project_env=Path(__file__).resolve().parents[1] / ".env",
        )
    except Exception as exc:
        print(f"  Warning: Runtime Manager .env load failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    os.environ["HERMES_HOME"] = str(home)
    os.environ["HOME"] = str(home_home)

    config_path = home / "config.yaml"
    if not config_path.exists():
        os.environ["TERMINAL_CWD"] = str(_safe_terminal_cwd(home, os.environ.get("TERMINAL_CWD")))
        return

    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        from hermes_cli.config import _expand_env_vars

        cfg = _expand_env_vars(cfg)
    except Exception as exc:
        print(
            f"  Warning: Runtime Manager config.yaml -> env bridge failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return

    if not isinstance(cfg, dict):
        return

    for key, value in cfg.items():
        if isinstance(value, (str, int, float, bool)):
            _set_env_value(str(key), value, override=False)

    terminal_cfg = cfg.get("terminal", {})
    if isinstance(terminal_cfg, dict):
        terminal_env_map = {
            "backend": "TERMINAL_ENV",
            "cwd": "TERMINAL_CWD",
            "timeout": "TERMINAL_TIMEOUT",
            "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
            "docker_image": "TERMINAL_DOCKER_IMAGE",
            "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
            "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
            "modal_image": "TERMINAL_MODAL_IMAGE",
            "daytona_image": "TERMINAL_DAYTONA_IMAGE",
            "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
            "ssh_host": "TERMINAL_SSH_HOST",
            "ssh_user": "TERMINAL_SSH_USER",
            "ssh_port": "TERMINAL_SSH_PORT",
            "ssh_key": "TERMINAL_SSH_KEY",
            "container_cpu": "TERMINAL_CONTAINER_CPU",
            "container_memory": "TERMINAL_CONTAINER_MEMORY",
            "container_disk": "TERMINAL_CONTAINER_DISK",
            "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
            "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
            "docker_env": "TERMINAL_DOCKER_ENV",
            "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
            "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
            "sandbox_dir": "TERMINAL_SANDBOX_DIR",
            "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
        }
        for cfg_key, env_var in terminal_env_map.items():
            if cfg_key not in terminal_cfg:
                continue
            value = terminal_cfg[cfg_key]
            if cfg_key == "cwd":
                value = str(_safe_terminal_cwd(home, value))
            _set_env_value(env_var, value)

    os.environ["TERMINAL_CWD"] = str(_safe_terminal_cwd(home, os.environ.get("TERMINAL_CWD")))

    auxiliary_cfg = cfg.get("auxiliary", {})
    if isinstance(auxiliary_cfg, dict):
        aux_keys = {"vision", "web_extract", "approval"}
        try:
            from hermes_cli.plugins import get_plugin_auxiliary_tasks

            for entry in get_plugin_auxiliary_tasks():
                aux_keys.add(entry["key"])
        except Exception:
            pass

        for task_key in aux_keys:
            task_cfg = auxiliary_cfg.get(task_key, {})
            if not isinstance(task_cfg, dict):
                continue
            upper = task_key.upper()
            mapping = {
                "provider": f"AUXILIARY_{upper}_PROVIDER",
                "model": f"AUXILIARY_{upper}_MODEL",
                "base_url": f"AUXILIARY_{upper}_BASE_URL",
                "api_key": f"AUXILIARY_{upper}_API_KEY",
            }
            for cfg_key, env_var in mapping.items():
                value = str(task_cfg.get(cfg_key, "")).strip()
                if value and not (cfg_key == "provider" and value == "auto"):
                    os.environ[env_var] = value

    agent_cfg = cfg.get("agent", {})
    if isinstance(agent_cfg, dict):
        agent_env_map = {
            "max_turns": "HERMES_MAX_ITERATIONS",
            "gateway_timeout": "HERMES_AGENT_TIMEOUT",
            "gateway_timeout_warning": "HERMES_AGENT_TIMEOUT_WARNING",
            "gateway_notify_interval": "HERMES_AGENT_NOTIFY_INTERVAL",
            "restart_drain_timeout": "HERMES_RESTART_DRAIN_TIMEOUT",
            "gateway_auto_continue_freshness": "HERMES_AUTO_CONTINUE_FRESHNESS",
        }
        for cfg_key, env_var in agent_env_map.items():
            if cfg_key in agent_cfg:
                _set_env_value(env_var, agent_cfg[cfg_key])

    display_cfg = cfg.get("display", {})
    if isinstance(display_cfg, dict):
        display_env_map = {
            "busy_input_mode": "HERMES_GATEWAY_BUSY_INPUT_MODE",
            "busy_text_mode": "HERMES_GATEWAY_BUSY_TEXT_MODE",
            "busy_ack_enabled": "HERMES_GATEWAY_BUSY_ACK_ENABLED",
        }
        for cfg_key, env_var in display_env_map.items():
            if cfg_key in display_cfg:
                _set_env_value(env_var, display_cfg[cfg_key])

    timezone = cfg.get("timezone")
    if isinstance(timezone, str) and timezone.strip():
        os.environ["HERMES_TIMEZONE"] = timezone.strip()

    security_cfg = cfg.get("security", {})
    if isinstance(security_cfg, dict) and "redact_secrets" in security_cfg:
        os.environ["HERMES_REDACT_SECRETS"] = str(security_cfg["redact_secrets"]).lower()

    network_cfg = cfg.get("network", {})
    if isinstance(network_cfg, dict) and network_cfg.get("force_ipv4"):
        try:
            from hermes_constants import apply_ipv4_preference

            apply_ipv4_preference(force=True)
        except Exception as exc:
            print(f"  Warning: Runtime Manager IPv4 preference failed: {exc}", file=sys.stderr)
