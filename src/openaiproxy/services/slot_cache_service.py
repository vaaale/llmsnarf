"""Persist llama-server KV cache slots per conversation.

llama-server started with --slot-save-path exposes
POST /slots/{id}?action=save|restore, taking a filename relative to that
directory. Restoring a saved slot before forwarding a request lets the server
skip re-processing the conversation prefix; saving after the response keeps
the snapshot current. Conversations are keyed by the X-Correlation-Id header,
falling back to a shared "default" session when the header is absent.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio
import httpx

from openaiproxy.models.models import EndpointConfig

DEFAULT_SESSION = "default"
SLOT_ID = 0


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")


def slot_filename(correlation_id: str | None, model: str | None = None) -> str:
    """Slot file for a conversation.

    The model is part of the name because a slot snapshot only makes sense for
    the instance that produced it — in router mode one session may hit several
    models, and restoring model A's KV into model B would be garbage.
    """
    session = _sanitize(correlation_id or "") or DEFAULT_SESSION
    name = f"slot_{session[:120]}"
    if model:
        name += f"__{_sanitize(model)[:120]}"
    return f"{name}.bin"


def server_root(base_url: str) -> str:
    """The /slots API lives at the server root, not under /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


async def probe_backend(
    base_url: str, api_key: str = "", model: str | None = None, autoload: bool = False
) -> dict:
    """Detect whether an upstream is a llama.cpp server, and how many slots it has.

    GET /props (at the server root) answers with total_slots on a single-model
    server, and on a router when ?model= names an instance — slots belong to the
    instance, not the router. A router queried without a model answers 200 with
    a stub carrying role="router" instead, which still identifies llama.cpp but
    yields no slot count.

    autoload=False keeps the probe from booting an unloaded model just to answer;
    the router then reports "model is not loaded" for idle models, which is fine
    for detection but means the slot count is only readable once loaded.
    """
    url = f"{server_root(base_url)}/props"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {}
    if model:
        params = {"model": model, "autoload": "true" if autoload else "false"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers, params=params)
    except Exception:
        return {"reachable": False, "backend": "generic", "total_slots": None, "router": False}
    try:
        data = response.json()
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {"reachable": True, "backend": "generic", "total_slots": None, "router": False}
    is_router = data.get("role") == "router"
    if response.status_code != 200:
        # a router rejects an unknown/unloaded model with a JSON error — the reply
        # still proves llama.cpp is there when we asked about a specific model
        message = str(data.get("error", {}).get("message", "")) if isinstance(data.get("error"), dict) else ""
        if model and ("not loaded" in message or "not found" in message or "model name is missing" in message):
            return {"reachable": True, "backend": "llamacpp", "total_slots": None, "router": True}
        return {"reachable": True, "backend": "generic", "total_slots": None, "router": False}
    total_slots = data.get("total_slots")
    if isinstance(total_slots, int):
        return {"reachable": True, "backend": "llamacpp", "total_slots": total_slots, "router": is_router}
    if is_router or "build_info" in data:
        return {"reachable": True, "backend": "llamacpp", "total_slots": None, "router": is_router}
    return {"reachable": True, "backend": "generic", "total_slots": None, "router": False}


def _slot_cache_applicable(endpoint: EndpointConfig) -> bool:
    return (
        endpoint.slot_cache
        and endpoint.backend == "llamacpp"
        and endpoint.mode != "local"
        and bool(endpoint.base_url)
    )


class SlotCacheService:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    async def restore(
        self, endpoint: EndpointConfig, correlation_id: str | None, slot: int = SLOT_ID, model: str | None = None
    ) -> None:
        await self._slot_action(endpoint, correlation_id, "restore", slot, model)

    async def save(
        self, endpoint: EndpointConfig, correlation_id: str | None, slot: int = SLOT_ID, model: str | None = None
    ) -> None:
        await self._slot_action(endpoint, correlation_id, "save", slot, model)

    async def _slot_action(
        self,
        endpoint: EndpointConfig,
        correlation_id: str | None,
        action: str,
        slot: int = SLOT_ID,
        model: str | None = None,
    ) -> None:
        if not _slot_cache_applicable(endpoint):
            return
        url = f"{server_root(endpoint.base_url)}/slots/{slot}"
        filename = slot_filename(correlation_id, model)
        headers = {}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        # /slots is a POST route: a router picks the target instance from the
        # body's "model" field, and a single-model server ignores the extra key
        body: dict = {"filename": filename}
        if model:
            body["model"] = model
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, params={"action": action}, headers=headers, json=body)
        except Exception as exc:
            self._logger.warning(
                "slot_cache_%s_error endpoint=%s filename=%s: %s", action, endpoint.name, filename, exc
            )
            return
        if response.status_code == 200:
            self._logger.debug("slot_cache_%s endpoint=%s filename=%s", action, endpoint.name, filename)
        else:
            # a failed restore is expected for the first message of a session
            log = self._logger.debug if action == "restore" else self._logger.warning
            log(
                "slot_cache_%s_failed endpoint=%s filename=%s status=%s body=%s",
                action,
                endpoint.name,
                filename,
                response.status_code,
                response.text[:200],
            )


@dataclass
class SlotLease:
    slot: int
    resident: bool  # this session was the slot's last occupant — its KV is still in VRAM


class _EndpointSlotState:
    def __init__(self, slot_count: int, base_url: str, override: int, probed_at: float):
        self.slot_count = slot_count
        self.base_url = base_url
        self.override = override
        self.probed_at = probed_at
        self.locks = [anyio.Lock() for _ in range(slot_count)]
        self.sessions: OrderedDict[str, int] = OrderedDict()  # session -> slot, LRU order
        self.last_session: list[str | None] = [None] * slot_count


class SlotAllocator:
    """Map conversation sessions to llama-server slots and serialize slot access.

    Assignment is sticky LRU: a session keeps its slot while active; when all
    slots are taken the least recently used session is evicted (its state is
    already on disk from its last save, so eviction needs no server call).
    The per-slot lock is held across restore -> inference -> save, which is
    what makes the sequence safe: restoring into a slot that is mid-generation
    for another session would destroy that generation's KV state.

    The slot count comes from the endpoint's slot_count override, or is probed
    from GET /props (total_slots) and re-checked every PROBE_TTL seconds so a
    llama-server restarted with a different -np is picked up.
    """

    PROBE_TTL = 300.0

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._lock = anyio.Lock()
        self._states: dict[str, _EndpointSlotState] = {}

    @asynccontextmanager
    async def lease(
        self,
        endpoint: EndpointConfig | None,
        correlation_id: str | None,
        enabled: bool = True,
        model: str | None = None,
    ):
        if not enabled or endpoint is None or not _slot_cache_applicable(endpoint):
            yield None
            return
        session = correlation_id or DEFAULT_SESSION
        # slots belong to a model instance, so each model gets its own slot space
        state = await self._state_for(endpoint, model)
        async with self._lock:
            slot = self._assign(state, session)
        async with state.locks[slot]:
            resident = state.last_session[slot] == session
            try:
                yield SlotLease(slot=slot, resident=resident)
            finally:
                # whatever happened, this session's tokens are what the slot holds now
                state.last_session[slot] = session

    async def _state_for(self, endpoint: EndpointConfig, model: str | None = None) -> _EndpointSlotState:
        key = f"{endpoint.name}\x00{model or ''}"
        override = endpoint.slot_count if endpoint.slot_count > 0 else 0
        now = time.monotonic()
        async with self._lock:
            state = self._states.get(key)
            if state is not None and state.base_url == endpoint.base_url and state.override == override:
                if override or now - state.probed_at < self.PROBE_TTL:
                    return state

        if override:
            count = override
        else:
            # autoload: the model is about to serve this request anyway
            probe = await probe_backend(endpoint.base_url, endpoint.api_key, model, autoload=True)
            total_slots = probe.get("total_slots")
            if isinstance(total_slots, int) and total_slots > 0:
                count = total_slots
            else:
                count = 1
                self._logger.warning(
                    "slot_probe_failed endpoint=%s base_url=%s model=%s reachable=%s — assuming 1 slot; "
                    "set slot_count on the endpoint to override",
                    endpoint.name,
                    endpoint.base_url,
                    model,
                    probe.get("reachable"),
                )

        async with self._lock:
            state = self._states.get(key)
            if (
                state is not None
                and state.base_url == endpoint.base_url
                and state.override == override
                and state.slot_count == count
            ):
                state.probed_at = now  # unchanged — keep locks and session map
                return state
            if state is not None:
                self._logger.info(
                    "slot_count_changed endpoint=%s model=%s slots=%d->%d — resetting session map",
                    endpoint.name,
                    model,
                    state.slot_count,
                    count,
                )
            state = _EndpointSlotState(count, endpoint.base_url, override, now)
            self._states[key] = state
            return state

    def _assign(self, state: _EndpointSlotState, session: str) -> int:
        if session in state.sessions:
            state.sessions.move_to_end(session)
            return state.sessions[session]
        if len(state.sessions) < state.slot_count:
            used = set(state.sessions.values())
            slot = next(i for i in range(state.slot_count) if i not in used)
        else:
            _, slot = state.sessions.popitem(last=False)  # evict least recently used
        state.sessions[session] = slot
        return slot
