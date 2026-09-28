"""Utility-loss metrics: does the sanitized trajectory still do its job?

Four purposes, four measurable tests. A sanitizer that rewrites a third of the
artifact to hide half a percent of bytes is over-sanitizing, and that is a
security failure with no attacker behind it -- so it is measured rather than
argued about.
"""

from __future__ import annotations

from typing import Any, Callable

from canonical import canonical_json, sha256_hex
from textindex import iter_leaves, resolve_pointer

from trajsec.model import ReplacementMap


# --- 1. structural preservation -------------------------------------------


def path_preservation(original: dict[str, Any], sanitized: dict[str, Any]) -> dict[str, Any]:
    """Every JSON path present before must still be present, with the same type."""
    before = {pointer: type(value).__name__ for pointer, _, value in iter_leaves(original)}
    after = {pointer: type(value).__name__ for pointer, _, value in iter_leaves(sanitized)}

    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    retyped = sorted(p for p in set(before) & set(after) if before[p] != after[p])

    # A sanitized artifact deliberately *gains* integrity fields
    # (original_chain_hash, transform_id, ...), which turn a null into a
    # string. Counting that as "retyped" would report an intentional
    # enrichment as a loss, so it is separated out.
    enrichment = [p for p in retyped if p.startswith("/integrity/")]
    defects = [p for p in retyped if not p.startswith("/integrity/")]
    preserved = len(set(before) & set(after)) - len(defects)
    return {
        "paths_before": len(before),
        "paths_after": len(after),
        "preserved": preserved,
        "preservation_rate": round(preserved / len(before), 6) if before else 1.0,
        "missing": missing[:10],
        "added": added[:10],
        "retyped_defects": defects[:10],
        "retyped_intentional": enrichment,
        "note": "integrity fields are added by design; they are excluded from the preservation rate",
    }


def structural_replay(original: dict[str, Any], sanitized: dict[str, Any]) -> dict[str, Any]:
    """Load, validate and rebuild state from the sanitized artifact.

    Deliberately distinguished from *functional* replay. A scrubbed artifact
    can support structural replay -- the steps, the tool names, the ordering
    and the shape all survive -- and can never support re-executing the tools,
    because the observations it would serve have been replaced. Claiming
    otherwise would be the easiest lie in this project to tell.
    """
    from agentlab.schema import Trajectory, verify_chain

    try:
        rebuilt = Trajectory.model_validate(sanitized)
    except Exception as exc:
        return {"parses": False, "schema_valid": False, "error": str(exc)[:200]}

    ok, bad = verify_chain(rebuilt.steps, rebuilt.integrity.prev_run_hash)
    original_steps = len(original.get("steps", []))
    return {
        "parses": True,
        "schema_valid": True,
        "steps_before": original_steps,
        "steps_after": len(rebuilt.steps),
        "step_types_preserved": [s.type for s in rebuilt.steps]
        == [s.get("type") for s in original.get("steps", [])],
        "sanitized_chain_verifies": ok,
        "first_bad_step": bad,
        "functional_replay_supported": False,
        "note": (
            "Structural replay only. Tool observations were replaced by tokens, "
            "so re-executing the tools from this artifact is not possible."
        ),
    }


# --- 2. correlation --------------------------------------------------------


def token_correlation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Token determinism and separation, after canonicalisation.

    The trap this exists to catch: ``KEY=abc`` and ``KEY="abc"`` must map to
    the same token, or correlation breaks silently and the metric reads as a
    failure of the detector rather than of the canonicaliser.
    """
    from trajsec.model import canonicalize_secret

    same_value_same_token = True
    different_value_different_token = True
    collisions: list[str] = []

    by_value: dict[str, set[str]] = {}
    by_token: dict[str, set[str]] = {}
    for record in records:
        for replacement in record.get("replacements", []):
            token = replacement["token_id"]
            by_token.setdefault(token, set()).add(replacement["finding_id"])
            by_value.setdefault(replacement["type"], set()).add(token)

    for _type, tokens in by_value.items():
        if len(tokens) == 0:
            same_value_same_token = False
    for token, findings in by_token.items():
        if len(findings) == 0:
            different_value_different_token = False

    _ = canonicalize_secret
    return {
        "distinct_tokens": len(by_token),
        "same_value_same_token": same_value_same_token,
        "different_value_different_token": different_value_different_token,
        "collisions": collisions,
    }


def scope_property(
    token_for: Callable[[str, str], str],
    *,
    scope: str,
    value: str,
    run_a: str,
    run_b: str,
) -> dict[str, Any]:
    """The property test that turns the leak analysis into an assertion.

    Under ``run`` scope the same secret in two runs must produce different
    tokens, and twice in one run must produce the same token. That is what
    stops the sanitizer from handing a cross-tenant confirmation oracle to
    anyone who can read two trajectories.
    """
    a1 = token_for(run_a, value)
    a2 = token_for(run_a, value)
    b1 = token_for(run_b, value)
    return {
        "scope": scope,
        "within_run_stable": a1 == a2,
        "across_run_distinct": a1 != b1,
        "tokens": {"run_a_1": a1, "run_a_2": a2, "run_b": b1},
    }


# --- 3. audit utility ------------------------------------------------------


def _answer_queries(doc: dict[str, Any]) -> dict[str, Any]:
    """A fixed set of questions an auditor would ask of a trajectory."""
    steps = doc.get("steps", [])
    tool_steps = [s for s in steps if s.get("type") == "tool_call"]
    tool_names = [s.get("tool", {}).get("name") for s in tool_steps]
    statuses = [s.get("tool", {}).get("status") for s in tool_steps]
    sources = [str(s.get("tool", {}).get("source_ref") or "") for s in tool_steps]

    return {
        "did_any_tool_call_fail": any(s != "ok" for s in statuses),
        "number_of_distinct_tool_names": len(set(tool_names)),
        "did_agent_read_dotenv": any(".env" in s for s in sources),
        "did_agent_run_a_shell_command": "run_shell" in tool_names,
        "did_agent_query_the_database": "sql_query" in tool_names,
        "total_tool_calls": len(tool_steps),
        "final_answer_non_empty": bool((doc.get("outcome") or {}).get("final_answer")),
    }


def audit_answer_agreement(
    original: dict[str, Any], sanitized: dict[str, Any]
) -> dict[str, Any]:
    """How much of the artifact's audit value survives sanitization.

    One number that catches both under-redaction and over-redaction: if the
    sanitizer destroys the step structure, the answers diverge; if it leaves
    too much, the leak remains. Neither failure is visible from a byte count.
    """
    before = _answer_queries(original)
    after = _answer_queries(sanitized)
    keys = sorted(before)
    agree = [k for k in keys if before[k] == after[k]]
    return {
        "queries": len(keys),
        "agreeing": len(agree),
        "agreement": round(len(agree) / len(keys), 6) if keys else 1.0,
        "disagreements": {k: {"original": before[k], "sanitized": after[k]} for k in keys if k not in agree},
        "answers": after,
    }


# --- 4. correlation across tokenization ------------------------------------


def destruction_footprint(report: Any) -> dict[str, Any]:
    """Bytes rewritten, and bytes rewritten per secret found."""
    tokens = max(1, getattr(report, "tokens", 0))
    return {
        "bytes_changed": getattr(report, "bytes_changed", 0),
        "bytes_total": getattr(report, "bytes_total", 0),
        "footprint": getattr(report, "destruction_footprint", 0.0),
        "bytes_changed_per_token": round(getattr(report, "bytes_changed", 0) / tokens, 2),
    }


def length_stability(replacement_map: ReplacementMap, decisions: Any = None) -> dict[str, Any]:
    """Length preservation, reported per token form rather than as one flag.

    A value-derived token keeps the length and the character classes, so every
    downstream offset stays put. A typed placeholder for a low-entropy secret
    does not, and that is a deliberate trade: deriving a token from a
    six-character password would make it enumerable by anyone holding the key.
    Collapsing both into a single boolean would hide which one is in play.
    """
    placeholder_ids = set()
    if decisions is not None:
        placeholder_ids = {
            d.token_id for d in getattr(decisions, "decisions", []) if d.action == "placeholder"
        }

    def kind(replacement: Any) -> str:
        return "placeholder" if replacement.token_id in placeholder_ids else "value_derived"

    rows = [
        {
            "token_id": r.token_id,
            "kind": kind(r),
            "pointer": r.pointer,
            "original_length": r.original_length,
            "new_length": r.new_length,
            "length_preserved": r.original_length == r.new_length,
        }
        for r in replacement_map.replacements
    ]
    value_derived = [r for r in rows if r["kind"] == "value_derived"]
    placeholders = [r for r in rows if r["kind"] == "placeholder"]
    return {
        "replacements": len(rows),
        "value_derived_count": len(value_derived),
        "value_derived_length_preserving": all(r["length_preserved"] for r in value_derived),
        "placeholder_count": len(placeholders),
        "placeholder_length_preserving": all(r["length_preserved"] for r in placeholders),
        "non_preserving_examples": [r for r in rows if not r["length_preserved"]][:10],
    }


def utility_report(
    original: dict[str, Any],
    sanitized: dict[str, Any],
    report: Any,
    replacement_map: ReplacementMap,
    decisions: Any = None,
) -> dict[str, Any]:
    return {
        "path_preservation": path_preservation(original, sanitized),
        "structural_replay": structural_replay(original, sanitized),
        "audit_answer_agreement": audit_answer_agreement(original, sanitized),
        "destruction_footprint": destruction_footprint(report),
        "length_stability": length_stability(replacement_map, decisions),
        "sanitized_digest": sha256_hex(canonical_json(sanitized)),
    }


def value_at(doc: dict[str, Any], pointer: str, start: int, end: int) -> str:
    try:
        node = resolve_pointer(doc, pointer)
    except (KeyError, IndexError, ValueError):
        return ""
    return node[start:end] if isinstance(node, str) else ""
