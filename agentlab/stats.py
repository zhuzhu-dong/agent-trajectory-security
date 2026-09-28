"""Experiment 1 aggregate metrics.

Everything here is computed from stored trajectories. Nothing is asserted
that is not summed from the artifacts.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from .schema import Trajectory, verify_chain


@dataclass
class RateCard:
    """Per-million-token prices, in USD. Zero means "not priced here"."""

    input_cache_hit: float = 0.0
    input_cache_miss: float = 0.0
    output: float = 0.0
    currency: str = "USD"

    @property
    def priced(self) -> bool:
        return self.output > 0.0


@dataclass
class RunSummary:
    run_id: str
    scenario_id: str
    repeat_index: int
    kind: str
    steps: int
    llm_calls: int
    tool_calls: int
    route_steps: int
    tool_errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_ms: float
    tool_ms: float
    wall_ms: float
    max_depth: int
    stop_reason: str
    chain_ok: bool
    chain_bad_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ExperimentOneStats:
    summaries: list[RunSummary] = field(default_factory=list)
    rate_card: RateCard = field(default_factory=RateCard)

    # --- aggregation -------------------------------------------------------

    def _values(self, attr: str) -> list[float]:
        return [float(getattr(s, attr)) for s in self.summaries]

    def totals(self) -> dict[str, Any]:
        if not self.summaries:
            return {}
        llm_ms = sum(s.llm_ms for s in self.summaries)
        tool_ms = sum(s.tool_ms for s in self.summaries)
        measured = llm_ms + tool_ms
        return {
            "runs": len(self.summaries),
            "scenarios": len({s.scenario_id for s in self.summaries}),
            "steps_total": sum(s.steps for s in self.summaries),
            "steps_per_run_mean": round(statistics.fmean(self._values("steps")), 2),
            "llm_calls_total": sum(s.llm_calls for s in self.summaries),
            "tool_calls_total": sum(s.tool_calls for s in self.summaries),
            "tool_errors_total": sum(s.tool_errors for s in self.summaries),
            "tokens_prompt": sum(s.prompt_tokens for s in self.summaries),
            "tokens_completion": sum(s.completion_tokens for s in self.summaries),
            "tokens_total": sum(s.total_tokens for s in self.summaries),
            "llm_ms_total": round(llm_ms, 1),
            "tool_ms_total": round(tool_ms, 1),
            "llm_time_share": round(llm_ms / measured, 4) if measured else None,
            "tool_time_share": round(tool_ms / measured, 4) if measured else None,
            "max_depth": max((s.max_depth for s in self.summaries), default=0),
            "chain_pass_rate": round(
                sum(1 for s in self.summaries if s.chain_ok) / len(self.summaries), 4
            ),
            "stop_reasons": _tally(s.stop_reason for s in self.summaries),
        }

    def cost_estimate(self) -> dict[str, Any]:
        """Cost from the rate card. Returns tokens-only if no rates are set."""
        if not self.rate_card.priced:
            return {
                "priced": False,
                "note": "no rate card configured; token counts reported instead",
                "tokens_total": sum(s.total_tokens for s in self.summaries),
            }
        prompt_tokens = sum(s.prompt_tokens for s in self.summaries)
        completion_tokens = sum(s.completion_tokens for s in self.summaries)
        # Without cache telemetry every prompt token is billed at the miss rate.
        cost = (
            prompt_tokens * self.rate_card.input_cache_miss
            + completion_tokens * self.rate_card.output
        ) / 1_000_000
        return {
            "priced": True,
            "currency": self.rate_card.currency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "assumed_input_rate_per_mtok": self.rate_card.input_cache_miss,
            "assumed_output_rate_per_mtok": self.rate_card.output,
            "estimated_total": round(cost, 6),
            "note": "upper bound: all prompt tokens billed at the cache-miss rate",
        }

    def per_scenario(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[RunSummary]] = {}
        for summary in self.summaries:
            grouped.setdefault(summary.scenario_id, []).append(summary)
        rows: list[dict[str, Any]] = []
        for scenario_id, group in grouped.items():
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "kind": group[0].kind,
                    "runs": len(group),
                    "steps_mean": round(statistics.fmean(g.steps for g in group), 2),
                    "tool_calls_mean": round(statistics.fmean(g.tool_calls for g in group), 2),
                    "tokens_mean": round(statistics.fmean(g.total_tokens for g in group), 1),
                    "tool_errors": sum(g.tool_errors for g in group),
                    "chain_ok": all(g.chain_ok for g in group),
                }
            )
        return sorted(rows, key=lambda r: r["scenario_id"])

    # --- rendering ---------------------------------------------------------

    def to_markdown(self) -> str:
        totals = self.totals()
        if not totals:
            return "_No runs recorded._\n"

        lines = ["### Experiment 1 — run statistics", ""]
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in (
            "runs",
            "scenarios",
            "steps_total",
            "steps_per_run_mean",
            "llm_calls_total",
            "tool_calls_total",
            "tool_errors_total",
        ):
            lines.append(f"| {key} | {totals[key]} |")
        lines.append(f"| tokens_prompt | {totals['tokens_prompt']:,} |")
        lines.append(f"| tokens_completion | {totals['tokens_completion']:,} |")
        lines.append(f"| tokens_total | {totals['tokens_total']:,} |")
        lines.append(f"| llm_ms_total | {totals['llm_ms_total']:,} |")
        lines.append(f"| tool_ms_total | {totals['tool_ms_total']:,} |")
        share = totals["llm_time_share"]
        lines.append(f"| llm_time_share | {share if share is not None else 'n/a'} |")
        lines.append(f"| tool_time_share | {totals['tool_time_share']} |")
        lines.append(f"| max_depth | {totals['max_depth']} |")
        lines.append(f"| chain_pass_rate | {totals['chain_pass_rate']} |")
        lines.append("")

        cost = self.cost_estimate()
        lines.append("### Cost")
        lines.append("")
        if cost["priced"]:
            lines.append(
                f"- Estimated total: **{cost['estimated_total']} {cost['currency']}** "
                f"for {cost['prompt_tokens']:,} prompt + {cost['completion_tokens']:,} completion tokens"
            )
            lines.append(
                f"- Rates assumed (per million tokens): "
                f"input {cost['assumed_input_rate_per_mtok']}, output {cost['assumed_output_rate_per_mtok']}"
            )
            lines.append(f"- {cost['note']}")
        else:
            lines.append(f"- {cost['note']} ({cost['tokens_total']:,} tokens total)")
        lines.append("")

        lines.append("### Per scenario")
        lines.append("")
        lines.append("| scenario | kind | runs | steps (mean) | tool calls (mean) | tokens (mean) | tool errors | chain |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in self.per_scenario():
            lines.append(
                f"| {row['scenario_id']} | {row['kind']} | {row['runs']} | {row['steps_mean']} | "
                f"{row['tool_calls_mean']} | {row['tokens_mean']} | {row['tool_errors']} | "
                f"{'ok' if row['chain_ok'] else 'FAIL'} |"
            )
        lines.append("")

        reasons = totals["stop_reasons"]
        lines.append("### Stop reasons")
        lines.append("")
        for reason, count in sorted(reasons.items()):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
        return "\n".join(lines)


def _tally(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _max_depth(trajectory: Trajectory) -> int:
    """Longest chain of parent links, i.e. the deepest nesting reached."""
    by_span = {step.span_id: step for step in trajectory.steps}
    depth = 0
    for step in trajectory.steps:
        current, hops = step, 0
        seen = set()
        while current.parent_span_id and current.parent_span_id in by_span:
            if current.span_id in seen:
                break
            seen.add(current.span_id)
            current = by_span[current.parent_span_id]
            hops += 1
        depth = max(depth, hops)
    return depth


def summarise(trajectory: Trajectory, kind: str = "task") -> RunSummary:
    chain_ok, chain_bad = verify_chain(trajectory.steps, trajectory.integrity.prev_run_hash)
    llm_steps = [s for s in trajectory.steps if s.type == "llm_call"]
    tool_steps = [s for s in trajectory.steps if s.type == "tool_call"]
    return RunSummary(
        run_id=trajectory.run_id,
        scenario_id=trajectory.scenario_id,
        repeat_index=trajectory.repeat_index,
        kind=kind,
        steps=len(trajectory.steps),
        llm_calls=len(llm_steps),
        tool_calls=len(tool_steps),
        route_steps=sum(1 for s in trajectory.steps if s.type == "route"),
        tool_errors=sum(1 for s in tool_steps if s.tool and s.tool.status != "ok"),
        prompt_tokens=sum(s.llm.usage.prompt_tokens for s in llm_steps if s.llm),
        completion_tokens=sum(s.llm.usage.completion_tokens for s in llm_steps if s.llm),
        total_tokens=sum(s.llm.usage.total_tokens for s in llm_steps if s.llm),
        llm_ms=round(sum(s.latency_ms for s in llm_steps), 3),
        tool_ms=round(sum(s.latency_ms for s in tool_steps), 3),
        wall_ms=trajectory.wall_clock_ms,
        max_depth=_max_depth(trajectory),
        stop_reason=trajectory.outcome.stop_reason,
        chain_ok=chain_ok,
        chain_bad_index=chain_bad,
    )


def collect(
    trajectories: Iterable[Trajectory],
    *,
    rate_card: RateCard | None = None,
    kinds: dict[str, str] | None = None,
) -> ExperimentOneStats:
    kinds = kinds or {}
    stats = ExperimentOneStats(rate_card=rate_card or RateCard())
    for trajectory in trajectories:
        stats.summaries.append(summarise(trajectory, kinds.get(trajectory.scenario_id, "task")))
    return stats
