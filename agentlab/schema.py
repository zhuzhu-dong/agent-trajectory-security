"""Trajectory schema.

The output contract for Experiment 1 and the input contract for Experiment 2,
so the fields here are shaped by what the security work needs to measure:

* every span is a decoded-string offset under a JSON Pointer (see ``textindex``)
* every step carries provenance, so "the agent read this because a file told
  it to" is distinguishable from "the agent read this because the user asked"
* truncation is visible, so a missed secret can be attributed to the detector
  rather than to the recorder
* the sanitized artifact records the original chain hash, because tokenizing
  a trajectory necessarily invalidates the chain over its bytes
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from canonical import canonical_json, chain_hash, sha256_hex

SCHEMA_VERSION = "1.0"
CHAIN_ALGORITHM = "sha256(prev_chain_hash || canonical_json(step_without_chain_hash))"


class StrictModel(BaseModel):
    """Reject unknown fields instead of dropping them.

    Pydantic's default is to *ignore* extras, which turns a typo in a
    constructor into silent data loss -- a missing field in the trajectory is
    indistinguishable from a field that was never produced. Forbidding extras
    makes that a loud error. Evolution is handled through ``schema_version``.
    """

    model_config = ConfigDict(extra="forbid")

Origin = Literal["user_input", "system_prompt", "tool_result", "model_output", "config"]
StepType = Literal["llm_call", "tool_call", "route", "final", "error"]
ToolStatus = Literal["ok", "denied", "error", "unknown"]


# --- message serialization -------------------------------------------------


def serialize_message(message: Any) -> dict[str, Any]:
    """Flatten a LangChain message into a stable, hashable dict.

    Only the fields the trajectory and the detector care about; provider
    extras are dropped so the hash does not move when a vendor adds a field.
    """
    msg_type = getattr(message, "type", None) or type(message).__name__
    role = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "SystemMessage": "system",
        "ToolMessage": "tool",
    }.get(msg_type, str(msg_type))

    content = getattr(message, "content", "")
    if not isinstance(content, str):
        # Some providers return a list of content blocks.
        content = canonical_json(content) if content else ""

    out: dict[str, Any] = {"role": role, "content": content}

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    name = getattr(message, "name", None)
    if name:
        out["name"] = name
    return out


def serialize_messages(messages: Any) -> list[dict[str, Any]]:
    """Handle both ``[msg, ...]`` and the batched ``[[msg, ...], ...]`` shape."""
    if not messages:
        return []
    if isinstance(messages, (list, tuple)) and messages and isinstance(messages[0], (list, tuple)):
        flat: list[Any] = []
        for batch in messages:
            flat.extend(batch)
        return [serialize_message(m) for m in flat]
    return [serialize_message(m) for m in messages]


# --- models ----------------------------------------------------------------


class Provenance(StrictModel):
    """Where a step's content came from, and whether it is attacker-influenced.

    ``untrusted`` is the taint bit: anything the model did not author and the
    user did not supply is untrusted, because in this sandbox a file can
    contain text that tries to steer the agent.
    """

    origin: Origin
    source_ref: str | None = None
    untrusted: bool = False


class Usage(StrictModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMCall(StrictModel):
    model: str
    messages_in: list[dict[str, Any]] = Field(default_factory=list)
    completion: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    streaming_chunks: int = 0


class ToolCallRecord(StrictModel):
    name: str
    call_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    status: ToolStatus = "unknown"
    rule_id: str | None = None
    truncated: bool = False
    original_bytes: int | None = None
    touched_paths: list[str] = Field(default_factory=list)
    error: str | None = None


class Step(StrictModel):
    step_index: int
    node: str
    type: StepType
    span_id: str
    parent_span_id: str | None = None
    trace_id: str
    ts_start: float
    ts_end: float
    latency_ms: float
    provenance: Provenance
    llm: LLMCall | None = None
    tool: ToolCallRecord | None = None
    state_delta: dict[str, Any] | None = None
    # Framework-level failures (LLM errors, swallowed exceptions) belong in
    # the trajectory: a run that failed partway is still evidence.
    error: str | None = None
    chain_hash: str = ""


class Outcome(StrictModel):
    final_answer: str = ""
    stop_reason: str = "unknown"
    total_steps: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    tool_errors: int = 0


class Integrity(StrictModel):
    chain_hash: str = ""
    prev_run_hash: str | None = None
    algorithm: str = CHAIN_ALGORITHM
    # Populated only when the trajectory has been through a sanitizer. The
    # original hash is the immutable evidence of the pre-sanitization bytes.
    original_chain_hash: str | None = None
    sanitized_chain_hash: str | None = None
    transform_id: str | None = None
    policy_version: str | None = None


class AgentConfig(StrictModel):
    model: str
    base_url_host: str
    temperature: float
    max_tokens: int
    tool_names: list[str] = Field(default_factory=list)
    framework_versions: dict[str, str] = Field(default_factory=dict)
    # The live key is never written. Only a fingerprint, so a run can be tied
    # to a credential without the credential being recoverable.
    api_key_fingerprint: str = ""
    api_key_recorded: bool = False
    # A synthetic, format-valid key placed here on purpose, so the
    # "known-prefix detection inside metadata" claim stays testable without
    # a live credential entering the artifact.
    synthetic_api_key_fixture: str | None = None


class SandboxDeclaration(StrictModel):
    """Which secrets in this artifact are real. Stated up front, not implied."""

    all_secrets_synthetic: bool = True
    real_credentials_recorded: bool = False
    exceptions: list[str] = Field(default_factory=list)
    consent_recorded: bool = True
    note: str = ""


class Trajectory(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    thread_id: str
    trace_id: str
    scenario_id: str
    scenario_prompt: str = ""
    repeat_index: int = 0
    agent_config: AgentConfig
    sandbox_declaration: SandboxDeclaration
    started_at: str
    ended_at: str
    wall_clock_ms: float
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    integrity: Integrity = Field(default_factory=Integrity)


# --- identifiers -----------------------------------------------------------


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


# --- hash chain ------------------------------------------------------------


def _step_payload(step: Step) -> dict[str, Any]:
    data = step.model_dump(mode="json")
    data.pop("chain_hash", None)
    return data


def build_chain(steps: list[Step], prev_run_hash: str | None = None) -> str:
    """Assign ``chain_hash`` to each step in order; return the final hash.

    Mutates ``steps`` in place. The genesis link mixes in ``prev_run_hash``
    so runs also chain to one another when a store supplies one.
    """
    previous = prev_run_hash or ""
    for step in steps:
        step.chain_hash = chain_hash(previous, _step_payload(step))
        previous = step.chain_hash
    return previous


def verify_chain(steps: list[Step], prev_run_hash: str | None = None) -> tuple[bool, int | None]:
    """Recompute the chain. Returns ``(ok, first_bad_index)``."""
    previous = prev_run_hash or ""
    for index, step in enumerate(steps):
        expected = chain_hash(previous, _step_payload(step))
        if expected != step.chain_hash:
            return False, index
        previous = step.chain_hash
    return True, None


def trajectory_content_hash(trajectory: Trajectory) -> str:
    """Hash of the whole artifact, independent of the chain fields."""
    data = trajectory.model_dump(mode="json")
    integrity = data.get("integrity", {})
    for key in ("chain_hash", "sanitized_chain_hash", "original_chain_hash"):
        integrity.pop(key, None)
    return sha256_hex(canonical_json(data))
