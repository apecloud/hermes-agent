from __future__ import annotations

import os

import uvicorn

from .app import create_app


def main() -> None:
    app = create_app()
    uvicorn.run(
        app,
        host=os.getenv("RUNTIME_MANAGER_HOST", "0.0.0.0"),
        port=int(os.getenv("RUNTIME_MANAGER_PORT", "8765")),
        log_level=os.getenv("RUNTIME_MANAGER_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
