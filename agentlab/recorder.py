"""Merge framework-level and runtime-level instrumentation into a trajectory.

Neither source is sufficient alone:

* callbacks give token usage, exact model timings and the prompt as sent, but
  no view of the graph -- which node ran, which edge was taken;
* the stream gives node transitions and route decisions, but not usage or
  per-call latency.

Steps are assembled by ordering both event streams on their start timestamps
and linking tool calls to the model turn that requested them.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from .callbacks import TrajectoryCallbackHandler
from .config import Settings
from .graph import build_graph, run_config
from .schema import (
    AgentConfig,
    Integrity,
    LLMCall,
    Outcome,
    Provenance,
    SandboxDeclaration,
    Step,
    ToolCallRecord,
    Trajectory,
    Usage,
    build_chain,
    new_span_id,
    new_trace_id,
    serialize_message,
    utc_now_iso,
)
from .scenarios import Scenario
from .tools import build_tools

# A synthetic, format-valid key that lives in agent_config on purpose, so the
# "known-prefix detection works inside metadata" claim stays testable without
# a live credential entering the artifact.
SYNTHETIC_KEY_FIXTURE = "sk-0000deadbeef0000deadbeef0000dead"

SANDBOX_NOTE = (
    "Every credential-bearing value inside the agent workspace is synthetic and "
    "non-functional. The live model API key that drives the agent is deliberately "
    "NOT recorded in this artifact: only a sha256 fingerprint of it appears, in "
    "agent_config.api_key_fingerprint. The value in "
    "agent_config.synthetic_api_key_fixture is a fake planted so that "
    "known-prefix detection remains testable inside metadata."
)


@dataclass
class RunResult:
    trajectory: Trajectory
    messages: list[Any] = field(default_factory=list)
    custom_events: list[dict[str, Any]] = field(default_factory=list)


def _agent_config(settings: Settings, tool_names: list[str]) -> AgentConfig:
    return AgentConfig(
        model=settings.model,
        base_url_host=settings.base_url.split("://", 1)[-1].split("/", 1)[0],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        tool_names=list(tool_names),
        framework_versions=settings.framework_versions(),
        api_key_fingerprint=settings.api_key_fingerprint,
        api_key_recorded=False,
        synthetic_api_key_fixture=SYNTHETIC_KEY_FIXTURE,
    )


def _tool_step_from_event(event: Any, index: int, trace_id: str) -> Step:
    artifact = event.artifact or {}
    status = artifact.get("status")
    if status is None:
        # No artifact means the tool failed before returning one; ToolNode
        # marks that on the message itself.
        status = "error" if event.error or event.message_status == "error" else "ok"
    return Step(
        step_index=index,
        node=event.node or "tools",
        type="tool_call",
        span_id=new_span_id(),
        trace_id=trace_id,
        ts_start=event.started,
        ts_end=event.finished or event.started,
        latency_ms=event.latency_ms,
        provenance=Provenance(
            origin="tool_result",
            source_ref=str(artifact.get("source_ref") or event.name),
            # Tool output is never authored by the model or the user. In this
            # sandbox a file can contain text addressed to automated tools.
            untrusted=True,
        ),
        tool=ToolCallRecord(
            name=event.name,
            call_id=event.call_id or event.run_id,
            args=event.args,
            result=event.content,
            status=status,
            rule_id=artifact.get("rule_id"),
            truncated=bool(artifact.get("truncated", False)),
            original_bytes=artifact.get("original_bytes"),
            touched_paths=list(artifact.get("touched_paths") or []),
            error=event.error,
        ),
    )


def _llm_step_from_event(event: Any, index: int, trace_id: str) -> Step:
    return Step(
        step_index=index,
        node=event.node or "agent",
        type="llm_call",
        span_id=new_span_id(),
        trace_id=trace_id,
        ts_start=event.started,
        ts_end=event.finished or event.started,
        latency_ms=event.latency_ms,
        provenance=Provenance(origin="model_output", source_ref=event.model or None, untrusted=False),
        llm=LLMCall(
            model=event.model or "unknown",
            messages_in=event.messages_in,
            completion=event.completion,
            tool_calls=event.tool_calls,
            finish_reason=event.finish_reason,
            usage=Usage(
                prompt_tokens=event.usage.get("prompt_tokens", 0),
                completion_tokens=event.usage.get("completion_tokens", 0),
                total_tokens=event.usage.get("total_tokens", 0),
            ),
            streaming_chunks=event.streaming_chunks,
        ),
        error=None if not event.error else event.error,
    )


def _route_step(payload: dict[str, Any], index: int, trace_id: str, at: float) -> Step:
    return Step(
        step_index=index,
        node=str(payload.get("from") or "agent"),
        type="route",
        span_id=new_span_id(),
        trace_id=trace_id,
        ts_start=at,
        ts_end=at,
        latency_ms=0.0,
        provenance=Provenance(origin="model_output", untrusted=False),
        state_delta={
            "decision": payload.get("decision"),
            "pending_tool_calls": payload.get("pending_tool_calls", []),
        },
    )


def _tool_step_from_observation(
    payload: dict[str, Any], index: int, trace_id: str, at: float
) -> Step:
    """Materialise a tool step from a node that reported its own observation.

    Replayed tool nodes are plain functions and so never fire LangChain's tool
    callbacks. Rather than pretend they did, they emit a ``tool_observation``
    custom event carrying the same fields, and it becomes a normal step here.
    """
    record = payload.get("record") or {}
    return Step(
        step_index=index,
        node="tools",
        type="tool_call",
        span_id=new_span_id(),
        trace_id=trace_id,
        ts_start=at,
        ts_end=at,
        latency_ms=0.0,
        provenance=Provenance(
            origin="tool_result",
            source_ref=str(record.get("source_ref") or payload.get("name") or ""),
            untrusted=True,
        ),
        tool=ToolCallRecord(
            name=str(payload.get("name") or record.get("name") or "unknown"),
            call_id=str(payload.get("call_id") or "unknown"),
            args=payload.get("args") or {},
            result=record.get("result"),
            status=record.get("status") or "unknown",
            rule_id=record.get("rule_id"),
            truncated=bool(record.get("truncated", False)),
            original_bytes=record.get("original_bytes"),
            touched_paths=list(record.get("touched_paths") or []),
        ),
    )


def _link_parents(steps: list[Step]) -> None:
    """Link each tool call to the model turn that requested it, and each model
    turn to the observation it was reacting to."""
    call_id_to_step: dict[str, str] = {}
    for step in steps:
        if step.type == "llm_call" and step.llm:
            for call in step.llm.tool_calls:
                if call.get("id"):
                    call_id_to_step[str(call["id"])] = step.span_id

    previous_span: str | None = None
    for step in steps:
        if step.type == "tool_call" and step.tool:
            step.parent_span_id = call_id_to_step.get(step.tool.call_id)
        elif step.type == "llm_call":
            step.parent_span_id = previous_span
        if step.type in ("llm_call", "tool_call"):
            previous_span = step.span_id


def run_scenario(
    settings: Settings,
    scenario: Scenario,
    *,
    repeat_index: int = 0,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    graph: Any = None,
    tools: list[Any] | None = None,
    resume_thread_id: str | None = None,
    journal: Any = None,
    model: Any = None,
) -> RunResult:
    """Execute one scenario and return its trajectory.

    ``graph``/``tools`` may be injected (used by the replay path, which
    substitutes a recording tool node for the live one). ``journal`` is called
    with each step as soon as it completes, so an interrupted run still leaves
    a usable append-only record on disk.
    """
    tools = tools if tools is not None else build_tools(settings.sandbox_root)
    graph = graph if graph is not None else build_graph(
        settings,
        tools,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        model=model,
    )

    thread_id = resume_thread_id or f"{scenario.scenario_id}-{repeat_index}-{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4().hex
    trace_id = new_trace_id()
    handler = TrajectoryCallbackHandler()
    config = {**run_config(thread_id), "callbacks": [handler]}

    state_messages: list[Any] = []
    custom_events: list[dict[str, Any]] = []
    extra_steps: list[Step] = []
    stop_reason = "completed"

    # Live journal state: steps are emitted the moment their event completes.
    journalled: list[Step] = []
    emitted_run_ids: set[str] = set()
    sent_count = 0

    def flush_completed() -> None:
        nonlocal sent_count
        if journal is None:
            return
        for event in handler.llm_events:
            if event.finished is not None and event.run_id not in emitted_run_ids:
                emitted_run_ids.add(event.run_id)
                journalled.append(_llm_step_from_event(event, -1, trace_id))
        for event in handler.tool_events:
            if event.finished is not None and event.run_id not in emitted_run_ids:
                emitted_run_ids.add(event.run_id)
                journalled.append(_tool_step_from_event(event, -1, trace_id))
        for index, step in enumerate(journalled):
            step.step_index = index
        _link_parents(journalled)
        while sent_count < len(journalled):
            journal(journalled[sent_count])
            sent_count += 1

    started_at = utc_now_iso()
    wall_start = time.perf_counter()

    inputs: Any
    if resume_thread_id is None:
        inputs = {"messages": [HumanMessage(content=scenario.prompt)]}
        # Seed the message list with the input. Node updates only carry what
        # each node *produced*, so without this the opening turn would be
        # missing from the reconstruction check.
        state_messages.append(inputs["messages"][0])
    else:
        # Resuming: the checkpoint already holds the conversation.
        inputs = None

    try:
        stream = graph.stream(
            inputs,
            config=config,
            stream_mode=["updates", "messages", "custom"],
            subgraphs=False,
        )
        for chunk in stream:
            mode, payload = chunk
            if mode == "updates":
                for _node, update in (payload or {}).items():
                    if isinstance(update, dict) and update.get("messages"):
                        state_messages.extend(update["messages"])
            elif mode == "custom":
                if isinstance(payload, dict):
                    custom_events.append(payload)
                    event_type = payload.get("type")
                    if event_type == "route":
                        extra = _route_step(payload, -1, trace_id, time.perf_counter())
                    elif event_type == "tool_observation":
                        extra = _tool_step_from_observation(
                            payload, -1, trace_id, time.perf_counter()
                        )
                    else:
                        extra = None
                    if extra is not None:
                        extra_steps.append(extra)
                        # Recorded in the journal too, so an interrupted run
                        # keeps which edge was taken and which observations
                        # were served, not only the model turns.
                        journalled.append(extra)
            elif mode == "messages":
                pass  # deltas are already summarised by the callback handler
            flush_completed()
    except GraphRecursionError:
        stop_reason = "recursion_limit"
    finally:
        flush_completed()

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    ended_at = utc_now_iso()

    if not state_messages and checkpointer is not None:
        snapshot = graph.get_state(config)
        state_messages = list(snapshot.values.get("messages", []))

    # --- assemble steps in start-time order -------------------------------
    raw_steps: list[Step] = []
    for event in handler.llm_events:
        raw_steps.append(_llm_step_from_event(event, -1, trace_id))
    for event in handler.tool_events:
        raw_steps.append(_tool_step_from_event(event, -1, trace_id))
    raw_steps.extend(extra_steps)
    raw_steps.sort(key=lambda s: s.ts_start)

    steps: list[Step] = []
    for index, step in enumerate(raw_steps):
        step.step_index = index
        steps.append(step)

    _link_parents(steps)

    # --- outcome ----------------------------------------------------------
    final_answer = ""
    for message in reversed(state_messages):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            content = getattr(message, "content", "")
            final_answer = content if isinstance(content, str) else str(content)
            break

    if final_answer:
        steps.append(
            Step(
                step_index=len(steps),
                node="agent",
                type="final",
                span_id=new_span_id(),
                trace_id=trace_id,
                ts_start=wall_start + wall_ms / 1000.0,
                ts_end=wall_start + wall_ms / 1000.0,
                latency_ms=0.0,
                provenance=Provenance(origin="model_output", untrusted=False),
                state_delta={"final_answer": final_answer},
            )
        )
    elif stop_reason == "completed":
        stop_reason = "no_final_answer"

    total_tokens = sum(
        (s.llm.usage.total_tokens if s.llm else 0) for s in steps
    )
    tool_steps = [s for s in steps if s.type == "tool_call"]

    trajectory = Trajectory(
        run_id=run_id,
        thread_id=thread_id,
        trace_id=trace_id,
        scenario_id=scenario.scenario_id,
        scenario_prompt=scenario.prompt,
        repeat_index=repeat_index,
        agent_config=_agent_config(settings, [t.name for t in tools]),
        sandbox_declaration=SandboxDeclaration(
            all_secrets_synthetic=True,
            real_credentials_recorded=False,
            exceptions=[],
            consent_recorded=True,
            note=SANDBOX_NOTE,
        ),
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_ms=round(wall_ms, 3),
        steps=steps,
        outcome=Outcome(
            final_answer=final_answer,
            stop_reason=stop_reason,
            total_steps=len(steps),
            total_tokens=total_tokens,
            total_tool_calls=len(tool_steps),
            tool_errors=sum(1 for s in tool_steps if s.tool and s.tool.status != "ok"),
        ),
        integrity=Integrity(),
    )
    trajectory.integrity.chain_hash = build_chain(trajectory.steps, None)

    return RunResult(trajectory=trajectory, messages=state_messages, custom_events=custom_events)


def reconstruct_messages(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Rebuild the conversation from the trajectory alone.

    This is the third piece of reuse evidence: if the trajectory carries the
    whole interaction, the message list must be recoverable without the graph,
    the model, or the tools.
    """
    messages: list[dict[str, Any]] = []
    if trajectory.scenario_prompt:
        messages.append({"role": "user", "content": trajectory.scenario_prompt})

    for step in trajectory.steps:
        if step.type == "llm_call" and step.llm:
            entry: dict[str, Any] = {"role": "assistant", "content": step.llm.completion}
            if step.llm.tool_calls:
                entry["tool_calls"] = step.llm.tool_calls
            messages.append(entry)
        elif step.type == "tool_call" and step.tool:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": step.tool.call_id,
                    "name": step.tool.name,
                    "content": step.tool.result or "",
                }
            )
    return messages


def live_message_digest(messages: list[Any]) -> str:
    """Canonical digest of a live message list, for the reconstruct check."""
    from canonical import sha256_hex, canonical_json

    return sha256_hex(canonical_json([serialize_message(m) for m in messages]))
