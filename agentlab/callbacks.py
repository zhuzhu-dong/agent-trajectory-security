"""Framework-level instrumentation.

Captures what the runtime stream cannot: token usage, exact model timings,
the full prompt as the model received it, and errors the graph swallowed.

Events are stored raw and correlated by ``run_id`` later; this class makes no
attempt to order or interpret them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .schema import serialize_messages


@dataclass
class LLMEvent:
    run_id: str
    parent_run_id: str | None
    node: str | None
    started: float
    finished: float | None = None
    messages_in: list[dict[str, Any]] = field(default_factory=list)
    completion: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    streaming_chunks: int = 0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        end = self.finished if self.finished is not None else time.perf_counter()
        return (end - self.started) * 1000.0


@dataclass
class ToolEvent:
    run_id: str
    parent_run_id: str | None
    node: str | None
    name: str
    call_id: str | None
    args: dict[str, Any]
    started: float
    finished: float | None = None
    content: str = ""
    artifact: dict[str, Any] | None = None
    message_status: str | None = None
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        end = self.finished if self.finished is not None else time.perf_counter()
        return (end - self.started) * 1000.0


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token usage out of an LLMResult across the shapes providers use."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    llm_output = getattr(response, "llm_output", None) or {}
    for key in ("token_usage", "usage"):
        if isinstance(llm_output.get(key), dict):
            raw = llm_output[key]
            usage["prompt_tokens"] = raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0
            usage["completion_tokens"] = raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0
            usage["total_tokens"] = raw.get("total_tokens", 0) or 0
            if not usage["total_tokens"]:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            return usage

    generations = getattr(response, "generations", None) or []
    for batch in generations:
        for generation in batch:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if isinstance(meta, dict):
                usage["prompt_tokens"] = meta.get("input_tokens", 0) or 0
                usage["completion_tokens"] = meta.get("output_tokens", 0) or 0
                usage["total_tokens"] = meta.get("total_tokens", 0) or 0
                if not usage["total_tokens"]:
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                return usage
    return usage


class TrajectoryCallbackHandler(BaseCallbackHandler):
    """Records raw framework events. Not thread-safe; one instance per run."""

    # LangChain requires this to accept arbitrary kwargs on every hook.
    raise_error = False

    def __init__(self) -> None:
        super().__init__()
        self.llm_events: list[LLMEvent] = []
        self.tool_events: list[ToolEvent] = []
        self._llm_by_run: dict[str, LLMEvent] = {}
        self._tool_by_run: dict[str, ToolEvent] = {}
        self.notes: list[str] = []

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _node_from(metadata: Any, tags: Any) -> str | None:
        """LangGraph stamps the emitting node into callback metadata."""
        if isinstance(metadata, dict):
            for key in ("langgraph_node", "langgraph_node_name", "checkpoint_ns"):
                value = metadata.get(key)
                if value:
                    # checkpoint_ns looks like "agent:uuid"; take the leading name.
                    return str(value).split(":")[0]
        if isinstance(tags, (list, tuple)):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("graph:step:"):
                    return None
        return None

    # --- LLM hooks ---------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        event = LLMEvent(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            node=self._node_from(metadata, tags),
            started=time.perf_counter(),
            messages_in=serialize_messages(messages),
            model=str((serialized or {}).get("name") or ""),
        )
        self.llm_events.append(event)
        self._llm_by_run[event.run_id] = event

    def on_llm_new_token(self, token: str, *, run_id: Any = None, **kwargs: Any) -> None:
        event = self._llm_by_run.get(str(run_id))
        if event is not None:
            event.streaming_chunks += 1

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        event = self._llm_by_run.get(str(run_id))
        if event is None:
            return
        event.finished = time.perf_counter()
        event.usage = _extract_usage(response)

        generations = getattr(response, "generations", None) or []
        if generations and generations[0]:
            generation = generations[0][0]
            message = getattr(generation, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                event.completion = content if isinstance(content, str) else str(content)
                raw_calls = getattr(message, "tool_calls", None) or []
                event.tool_calls = [
                    {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                    for tc in raw_calls
                ]
                response_metadata = getattr(message, "response_metadata", None) or {}
                event.finish_reason = response_metadata.get("finish_reason")
                if not event.model:
                    event.model = str(response_metadata.get("model_name") or "")
                if not event.finish_reason:
                    info = getattr(generation, "generation_info", None) or {}
                    event.finish_reason = info.get("finish_reason")

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        event = self._llm_by_run.get(str(run_id))
        if event is not None:
            event.finished = time.perf_counter()
            event.error = f"{type(error).__name__}: {error}"
        else:
            self.notes.append(f"llm error without start: {type(error).__name__}: {error}")

    # --- tool hooks --------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        args = inputs if isinstance(inputs, dict) else {"input": input_str}
        event = ToolEvent(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            node=self._node_from(metadata, tags),
            name=str((serialized or {}).get("name") or kwargs.get("name") or "unknown"),
            call_id=kwargs.get("tool_call_id"),
            args=args,
            started=time.perf_counter(),
        )
        self.tool_events.append(event)
        self._tool_by_run[event.run_id] = event

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        event = self._tool_by_run.get(str(run_id))
        if event is None:
            return
        event.finished = time.perf_counter()
        # ToolNode passes the ToolMessage through, which is where the
        # structured artifact lives.
        content = getattr(output, "content", output)
        event.content = content if isinstance(content, str) else str(content)
        artifact = getattr(output, "artifact", None)
        if isinstance(artifact, dict):
            event.artifact = artifact
        # ToolNode marks tool-level failures here even when the tool function
        # returned normally, so this is the only signal for a swallowed error.
        event.message_status = getattr(output, "status", None)
        if getattr(output, "tool_call_id", None):
            event.call_id = output.tool_call_id

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        event = self._tool_by_run.get(str(run_id))
        if event is not None:
            event.finished = time.perf_counter()
            event.error = f"{type(error).__name__}: {error}"
        else:
            self.notes.append(f"tool error without start: {type(error).__name__}: {error}")
