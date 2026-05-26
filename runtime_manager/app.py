from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "runtime_manager requires fastapi and pydantic. Install the web extra or run inside the published container."
    ) from exc

from .manager import RuntimeManager


class RunRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    enabled_toolsets: list[str] | None = None
    disabled_toolsets: list[str] | None = None
    skip_memory: bool = False
    skip_context_files: bool = True
    max_iterations: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None


class ApprovalRequest(BaseModel):
    choice: str
    resolve_all: bool = False


class RuntimeManagerState:
    def __init__(self, manager: RuntimeManager):
        self.manager = manager


async def _authorize(state: RuntimeManagerState, authorization: str | None) -> None:
    if not state.manager.authorize(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _format_sse(event: dict[str, Any]) -> str:
    event_id = event.get("event_id")
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _sse_stream(handle, *, last_event_id: int | None = None):
    queue = handle.subscribe()
    try:
        replay = list(handle.events)
        replayed_event_id = last_event_id or 0
        for event in replay:
            event_id = int(event.get("event_id") or 0)
            if last_event_id is not None and event_id <= last_event_id:
                continue
            replayed_event_id = max(replayed_event_id, event_id)
            yield _format_sse(event)
        latest_event_id = int(handle.events[-1].get("event_id") or 0) if handle.events else 0
        if handle.status in {"completed", "failed", "cancelled"} and latest_event_id <= replayed_event_id:
            return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            event_id = int(event.get("event_id") or 0)
            if event_id <= replayed_event_id:
                continue
            replayed_event_id = max(replayed_event_id, event_id)
            yield _format_sse(event)
            if event.get("event") in {"run.completed", "run.failed", "run.cancelled"}:
                break
    finally:
        handle.unsubscribe(queue)


def create_app(
    *,
    users_root: str | Path | None = None,
    api_key: str | None = None,
    python_executable: str | None = None,
    worker_script: str | Path | None = None,
) -> FastAPI:
    manager = RuntimeManager(
        users_root=users_root or os.getenv("RUNTIME_MANAGER_USERS_ROOT", "/opt/data/users"),
        api_key=api_key if api_key is not None else os.getenv("RUNTIME_MANAGER_API_KEY", ""),
        python_executable=python_executable
        or os.getenv("RUNTIME_MANAGER_PYTHON", os.getenv("PYTHON", "python3")),
        worker_script=worker_script,
    )

    app = FastAPI(title="Hermes Runtime Manager")
    app.state.runtime_manager = RuntimeManagerState(manager)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/agent/runs")
    async def start_run(body: RunRequest, authorization: str | None = Header(default=None)):
        state = app.state.runtime_manager
        await _authorize(state, authorization)
        try:
            handle = await state.manager.start_run(body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"run_id": handle.run_id, "status": "started"}, status_code=202)

    @app.get("/agent/runs/{run_id}")
    async def get_run(run_id: str, authorization: str | None = Header(default=None)):
        state = app.state.runtime_manager
        await _authorize(state, authorization)
        handle = state.manager.registry.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return handle.snapshot()

    @app.get("/agent/runs/{run_id}/events")
    async def events(
        run_id: str,
        authorization: str | None = Header(default=None),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        state = app.state.runtime_manager
        await _authorize(state, authorization)
        handle = state.manager.registry.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="Run not found")
        cursor: int | None = None
        if last_event_id:
            try:
                cursor = int(last_event_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer") from None
        return StreamingResponse(
            _sse_stream(handle, last_event_id=cursor),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/agent/runs/{run_id}/approval")
    async def approval(
        run_id: str,
        body: ApprovalRequest,
        authorization: str | None = Header(default=None),
    ):
        state = app.state.runtime_manager
        await _authorize(state, authorization)
        handle = state.manager.registry.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            await state.manager.approve_run(run_id, choice=body.choice, resolve_all=body.resolve_all)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run_id": run_id, "status": handle.status}

    @app.post("/agent/runs/{run_id}/stop")
    async def stop(run_id: str, authorization: str | None = Header(default=None)):
        state = app.state.runtime_manager
        await _authorize(state, authorization)
        handle = state.manager.registry.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            await state.manager.stop_run(run_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run_id": run_id, "status": handle.status}

    return app


app = create_app()
