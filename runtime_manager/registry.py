from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunHandle:
    run_id: str
    user_id: str
    conversation_id: str
    session_id: str
    model: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_event: str | None = None
    output: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    process: Any = None
    stdin: Any = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    max_events: int = 1000
    _next_event_id: int = 1

    def snapshot(self) -> dict[str, Any]:
        data = {
            "object": "runtime_manager.run",
            "run_id": self.run_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_event": self.last_event,
        }
        if self.output is not None:
            data["output"] = self.output
        if self.error is not None:
            data["error"] = self.error
        if self.usage is not None:
            data["usage"] = self.usage
        return data

    def publish(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("event_id", self._next_event_id)
        self._next_event_id += 1
        self.events.append(event)
        if self.max_events > 0 and len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]
        self.last_event = event.get("event")
        self.updated_at = float(event.get("timestamp", time.time()))
        kind = self.last_event or ""
        if kind == "run.running":
            self.status = "running"
        elif kind == "approval.request":
            self.status = "waiting_for_approval"
        elif kind == "approval.responded":
            self.status = "running"
        elif kind == "run.cancelling":
            self.status = "cancelling"
        elif kind == "run.completed":
            self.status = "completed"
            self.output = event.get("output")
            self.usage = event.get("usage")
        elif kind == "run.failed":
            self.status = "failed"
            self.error = event.get("error")
        elif kind == "run.cancelled":
            self.status = "cancelled"
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def subscribe(self, maxsize: int = 256) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)


class RunRegistry:
    _TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

    def __init__(self, *, max_events_per_run: int = 1000, completed_run_ttl_seconds: float = 3600.0) -> None:
        self._runs: dict[str, RunHandle] = {}
        self.max_events_per_run = max_events_per_run
        self.completed_run_ttl_seconds = completed_run_ttl_seconds

    def create(self, *, run_id: str, user_id: str, conversation_id: str, session_id: str, model: str) -> RunHandle:
        self.prune()
        handle = RunHandle(
            run_id=run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            model=model,
            max_events=self.max_events_per_run,
        )
        self._runs[run_id] = handle
        return handle

    def get(self, run_id: str) -> RunHandle | None:
        self.prune()
        return self._runs.get(run_id)

    def require(self, run_id: str) -> RunHandle:
        handle = self.get(run_id)
        if handle is None:
            raise KeyError(run_id)
        return handle

    def prune(self, *, now: float | None = None) -> int:
        if self.completed_run_ttl_seconds <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - self.completed_run_ttl_seconds
        expired = [
            run_id
            for run_id, handle in self._runs.items()
            if handle.status in self._TERMINAL_STATUSES and handle.updated_at < cutoff
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)
        return len(expired)
