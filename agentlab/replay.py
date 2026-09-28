"""The three pieces of evidence that a trajectory is *reusable*.

Storing JSON proves nothing. Each function here tests a different claim:

``replay_trajectory``
    The trajectory alone is sufficient to re-execute the run: the model turns
    are replayed from the recording and the tool observations are served from
    the recording, with no API call and no filesystem access.
``verify_reconstruction``
    The conversation is recoverable from the artifact without the graph.
``resume_demo``
    A checkpointed run that was interrupted mid-tool-call can be continued,
    and the continued portion links back to the original span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from canonical import canonical_json, sha256_hex

from .config import Settings
from .fakemodel import ScriptedChatModel
from .graph import build_graph
from .recorder import RunResult, live_message_digest, reconstruct_messages, run_scenario
from .schema import Trajectory, serialize_message
from .scenarios import Scenario


# --- replay ----------------------------------------------------------------


def build_replay_model(trajectory: Trajectory) -> ScriptedChatModel:
    """Rebuild the model turns from the recorded trajectory."""
    script: list[AIMessage] = []
    for step in trajectory.steps:
        if step.type != "llm_call" or step.llm is None:
            continue
        script.append(
            AIMessage(
                content=step.llm.completion,
                tool_calls=[
                    {
                        "name": call.get("name"),
                        "args": call.get("args") or {},
                        "id": call.get("id"),
                        "type": "tool_call",
                    }
                    for call in step.llm.tool_calls
                ],
            )
        )
    if not script:
        raise ValueError(f"trajectory {trajectory.run_id} contains no model turns to replay")
    return ScriptedChatModel(script=script, model_name="replay-from-trajectory")


def build_replay_tool_node(trajectory: Trajectory) -> Any:
    """A tool node that serves recorded observations instead of executing.

    Keyed by tool call id rather than by call order, so a replayed run that
    diverges produces a visible miss rather than silently misaligned results.
    """
    recorded: dict[str, dict[str, Any]] = {}
    for step in trajectory.steps:
        if step.type == "tool_call" and step.tool is not None:
            recorded[step.tool.call_id] = {
                "name": step.tool.name,
                "result": step.tool.result,
                "status": step.tool.status,
                "rule_id": step.tool.rule_id,
                "truncated": step.tool.truncated,
                "original_bytes": step.tool.original_bytes,
                "touched_paths": step.tool.touched_paths,
                "source_ref": step.provenance.source_ref,
            }

    def replay_node(state: dict[str, Any]) -> dict[str, Any]:
        last = state["messages"][-1]
        produced: list[ToolMessage] = []

        # A plain function node does not go through LangChain's tool
        # callbacks, so the recorder cannot see these observations the way it
        # sees live ones. Report them explicitly instead of letting the
        # replayed trajectory come out empty.
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            writer = None

        for call in getattr(last, "tool_calls", None) or []:
            call_id = call.get("id") or "unknown"
            record = recorded.get(call_id)
            if record is None:
                produced.append(
                    ToolMessage(
                        content=f"REPLAY MISS: no recorded observation for {call.get('name')}",
                        tool_call_id=call_id,
                        name=call.get("name") or "unknown",
                        status="error",
                        artifact={"status": "replay_miss", "tool": call.get("name")},
                    )
                )
                if writer is not None:
                    writer(
                        {
                            "type": "tool_observation",
                            "call_id": call_id,
                            "name": call.get("name"),
                            "args": call.get("args") or {},
                            "record": {"status": "replay_miss", "result": None},
                        }
                    )
                continue

            produced.append(
                ToolMessage(
                    content=record["result"] or "",
                    tool_call_id=call_id,
                    name=record["name"],
                    status="success" if record["status"] == "ok" else "error",
                    artifact={
                        "tool": record["name"],
                        "source_ref": record["source_ref"],
                        "status": record["status"],
                        "rule_id": record["rule_id"],
                        "truncated": record["truncated"],
                        "original_bytes": record["original_bytes"],
                        "touched_paths": record["touched_paths"],
                        "replayed": True,
                    },
                )
            )
            if writer is not None:
                writer(
                    {
                        "type": "tool_observation",
                        "call_id": call_id,
                        "name": record["name"],
                        "args": call.get("args") or {},
                        "record": record,
                    }
                )
        return {"messages": produced}

    return replay_node


@dataclass
class ReplayReport:
    run_id: str
    original_tool_calls: int
    replayed_tool_calls: int
    identical_sequence: bool
    mismatches: list[str] = field(default_factory=list)
    replay_misses: int = 0

    @property
    def ok(self) -> bool:
        return self.identical_sequence and self.replay_misses == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_tool_calls": self.original_tool_calls,
            "replayed_tool_calls": self.replayed_tool_calls,
            "identical_sequence": self.identical_sequence,
            "replay_misses": self.replay_misses,
            "mismatches": self.mismatches,
            "ok": self.ok,
        }


def replay_trajectory(settings: Settings, trajectory: Trajectory) -> tuple[RunResult, ReplayReport]:
    """Re-execute a recorded run without the model or the tools."""
    model = build_replay_model(trajectory)
    tool_node = build_replay_tool_node(trajectory)
    graph = build_graph(settings, [], model=model, tool_node=tool_node)

    scenario = Scenario(
        scenario_id=f"replay-{trajectory.scenario_id}",
        prompt=trajectory.scenario_prompt,
        kind="task",
        expects_tools=True,
        notes="synthetic scenario used only to drive a replay",
    )
    result = run_scenario(settings, scenario, graph=graph, tools=[], model=model)

    original = [
        (s.tool.name, canonical_json(s.tool.args), s.tool.result, s.tool.status)
        for s in trajectory.steps
        if s.type == "tool_call" and s.tool
    ]
    replayed = [
        (s.tool.name, canonical_json(s.tool.args), s.tool.result, s.tool.status)
        for s in result.trajectory.steps
        if s.type == "tool_call" and s.tool
    ]

    mismatches: list[str] = []
    for index, (want, got) in enumerate(zip(original, replayed)):
        if want != got:
            detail = []
            for label, a, b in zip(("name", "args", "result", "status"), want, got):
                if a != b:
                    detail.append(f"{label}: {str(a)[:60]!r} != {str(b)[:60]!r}")
            mismatches.append(f"tool call {index}: " + "; ".join(detail))
    if len(original) != len(replayed):
        mismatches.append(f"tool call count {len(original)} != {len(replayed)}")

    misses = sum(
        1
        for s in result.trajectory.steps
        if s.type == "tool_call"
        and s.tool
        and s.tool.result is not None
        and s.tool.result.startswith("REPLAY MISS")
    )

    return result, ReplayReport(
        run_id=trajectory.run_id,
        original_tool_calls=len(original),
        replayed_tool_calls=len(replayed),
        identical_sequence=not mismatches,
        mismatches=mismatches,
        replay_misses=misses,
    )


# --- reconstruction --------------------------------------------------------


def verify_reconstruction(trajectory: Trajectory, live_messages: list[Any]) -> dict[str, Any]:
    """Compare the message list rebuilt from the artifact with the live one.

    Compared on the canonical serialization, so provider extras and object
    identity cannot make a correct reconstruction look wrong.
    """
    rebuilt = reconstruct_messages(trajectory)
    live = [serialize_message(m) for m in live_messages]

    rebuilt_digest = sha256_hex(canonical_json(rebuilt))
    live_digest = sha256_hex(canonical_json(live))

    differences: list[str] = []
    if len(rebuilt) != len(live):
        differences.append(f"length: rebuilt {len(rebuilt)} vs live {len(live)}")
    for index, (a, b) in enumerate(zip(rebuilt, live)):
        if a != b:
            keys = sorted(set(a) | set(b))
            changed = [k for k in keys if a.get(k) != b.get(k)]
            differences.append(f"message {index} differs in {changed}")

    return {
        "run_id": trajectory.run_id,
        "rebuilt_messages": len(rebuilt),
        "live_messages": len(live),
        "rebuilt_digest": rebuilt_digest,
        "live_digest": live_digest,
        "identical": rebuilt_digest == live_digest,
        "differences": differences,
    }


def live_messages_from_artifact(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Rebuild the conversation the model actually saw, from the recorded prompts.

    Every model turn logs the exact prompt it received, so the last turn's
    prompt plus its own reply is the conversation as it stood at the end. That
    is an *independently recorded* view of the run -- the callback's log of
    what was sent -- and the assembled step list has to agree with it. A
    message the recorder dropped from ``steps`` but that the model demonstrably
    saw would show up as a mismatch here.

    This is what makes the reconstruction check verifiable offline, from a
    stored artifact, by anyone -- rather than only in the process that ran it.
    """
    llm_steps = [s for s in trajectory.steps if s.type == "llm_call" and s.llm]
    if not llm_steps:
        return []
    last = llm_steps[-1].llm
    messages: list[dict[str, Any]] = [dict(m) for m in last.messages_in]
    entry: dict[str, Any] = {"role": "assistant", "content": last.completion}
    if last.tool_calls:
        entry["tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
            for c in last.tool_calls
        ]
    messages.append(entry)
    return messages


def verify_reconstruction_offline(trajectory: Trajectory) -> dict[str, Any]:
    """Reconstruction check that needs nothing but the artifact."""
    live = live_messages_from_artifact(trajectory)
    rebuilt = reconstruct_messages(trajectory)
    rebuilt_digest = sha256_hex(canonical_json(rebuilt))
    live_digest = sha256_hex(canonical_json(live))

    differences: list[str] = []
    if len(rebuilt) != len(live):
        differences.append(f"length: rebuilt {len(rebuilt)} vs live {len(live)}")
    for index, (a, b) in enumerate(zip(rebuilt, live)):
        if a != b:
            keys = sorted(set(a) | set(b))
            changed = [k for k in keys if a.get(k) != b.get(k)]
            differences.append(f"message {index} differs in {changed}")

    return {
        "run_id": trajectory.run_id,
        "rebuilt_messages": len(rebuilt),
        "live_messages": len(live),
        "rebuilt_digest": rebuilt_digest,
        "live_digest": live_digest,
        "identical": rebuilt_digest == live_digest,
        "differences": differences,
    }


def reconfigure_reuse_report(
    settings: Settings, trajectories: list[Trajectory]
) -> dict[str, Any]:
    """Run all three reuse checks over every stored trajectory."""
    reconstruction = [verify_reconstruction_offline(t) for t in trajectories]
    replay_reports = []
    for trajectory in trajectories:
        if trajectory.outcome.total_tool_calls == 0:
            continue
        _, report = replay_trajectory(settings, trajectory)
        replay_reports.append(report)

    return {
        "runs_checked": len(trajectories),
        "reconstruction_identical": sum(1 for r in reconstruction if r["identical"]),
        "reconstruction": reconstruction,
        "replay": [r.as_dict() for r in replay_reports],
        "replay_identical": sum(1 for r in replay_reports if r.ok),
        "replay_total": len(replay_reports),
    }


# --- resume ----------------------------------------------------------------


def resume_demo(
    settings: Settings,
    scenario: Scenario,
    *,
    checkpointer: Any,
    model: Any,
    tool_node: Any = None,
) -> dict[str, Any]:
    """Interrupt a run before its tool call, then continue it.

    Returns a report describing both halves and whether the resumed portion
    links back to the span that requested the tool.
    """
    from .tools import build_tools

    tools = build_tools(settings.sandbox_root)

    interrupted_graph = build_graph(
        settings, tools, model=model, checkpointer=checkpointer,
        interrupt_before=["tools"], tool_node=tool_node,
    )
    first = run_scenario(settings, scenario, graph=interrupted_graph, tools=tools, model=model)

    resumed_graph = build_graph(
        settings, tools, model=model, checkpointer=checkpointer, tool_node=tool_node
    )
    second = run_scenario(
        settings,
        scenario,
        graph=resumed_graph,
        tools=tools,
        model=model,
        resume_thread_id=first.trajectory.thread_id,
    )

    resumed_tool_steps = [s for s in second.trajectory.steps if s.type == "tool_call"]
    requested_spans = {
        str(call.get("id")): s.span_id
        for s in first.trajectory.steps
        if s.type == "llm_call" and s.llm
        for call in s.llm.tool_calls
        if call.get("id")
    }
    linked = [
        s.tool.call_id for s in resumed_tool_steps if s.tool and s.tool.call_id in requested_spans
    ]

    return {
        "thread_id": first.trajectory.thread_id,
        "first_half_steps": len(first.trajectory.steps),
        "first_half_stop_reason": first.trajectory.outcome.stop_reason,
        "first_half_tool_calls": first.trajectory.outcome.total_tool_calls,
        "second_half_steps": len(second.trajectory.steps),
        "second_half_stop_reason": second.trajectory.outcome.stop_reason,
        "second_half_tool_calls": second.trajectory.outcome.total_tool_calls,
        "second_half_final_answer": second.trajectory.outcome.final_answer[:120],
        "resumed_tool_calls_linked_to_original_request": len(linked),
        "interrupted_before_tools": first.trajectory.outcome.total_tool_calls == 0,
        "resumed_successfully": bool(second.trajectory.outcome.total_tool_calls) and bool(linked),
    }


def reconstruction_digest_of_live(messages: list[Any]) -> str:
    return live_message_digest(messages)
