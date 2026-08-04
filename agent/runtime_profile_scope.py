"""Request-scoped runtime profile constraints.

API-server conversations can bind a business-level agent profile without
switching the user's HERMES_HOME.  The profile affects only the current turn's
runtime scope: visible skill roots and subprocess environment filtering.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterable


_PROFILE_SKILL_DIRS: ContextVar[tuple[str, ...]] = ContextVar(
    "HERMES_AGENT_PROFILE_SKILL_DIRS",
    default=(),
)
_PROFILE_ENV_BLOCKLIST: ContextVar[frozenset[str]] = ContextVar(
    "HERMES_AGENT_PROFILE_ENV_BLOCKLIST",
    default=frozenset(),
)


def set_runtime_profile_scope(
    *,
    skill_dirs: Iterable[str | Path] | None = None,
    env_blocklist: Iterable[str] | None = None,
) -> list[Token]:
    """Install per-turn profile scope and return reset tokens."""

    tokens: list[Token] = []
    if skill_dirs is not None:
        normalized_dirs = tuple(
            str(Path(str(path)).expanduser())
            for path in skill_dirs
            if str(path).strip()
        )
        tokens.append(_PROFILE_SKILL_DIRS.set(normalized_dirs))

    if env_blocklist is not None:
        normalized_env = frozenset(
            str(name).strip()
            for name in env_blocklist
            if str(name).strip()
        )
        tokens.append(_PROFILE_ENV_BLOCKLIST.set(normalized_env))

    return tokens


def clear_runtime_profile_scope(tokens: Iterable[Token]) -> None:
    """Reset tokens returned by :func:`set_runtime_profile_scope`."""

    for token in reversed(list(tokens)):
        token.var.reset(token)


def get_runtime_profile_skill_dirs() -> tuple[Path, ...]:
    """Return profile-scoped skill roots for the current context, if any."""

    return tuple(Path(path) for path in _PROFILE_SKILL_DIRS.get(()))


def get_runtime_profile_env_blocklist() -> frozenset[str]:
    """Return environment variable names hidden from profile subprocesses."""

    return _PROFILE_ENV_BLOCKLIST.get(frozenset())
