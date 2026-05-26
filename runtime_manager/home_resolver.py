from __future__ import annotations

import re
from pathlib import Path

_USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")
_BOOTSTRAP_DIRS = (
    "sessions",
    "memories",
    "skills",
    "logs",
)


class UserHomeResolver:
    """Resolve a user id to an isolated Hermes home directory."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).expanduser().resolve()

    @staticmethod
    def validate_user_id(user_id: str) -> str:
        value = str(user_id or "").strip()
        if not value:
            raise ValueError("user_id is required")
        if not _USER_ID_RE.fullmatch(value):
            raise ValueError(f"invalid user_id: {user_id!r}")
        return value

    def resolve(self, user_id: str, *, create: bool = True) -> Path:
        value = self.validate_user_id(user_id)
        home = (self.base_dir / value).resolve()
        home.relative_to(self.base_dir)
        if create:
            home.mkdir(parents=True, exist_ok=True)
            for name in _BOOTSTRAP_DIRS:
                (home / name).mkdir(exist_ok=True)
        return home
