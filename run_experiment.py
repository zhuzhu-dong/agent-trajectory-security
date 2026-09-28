"""Single entry point for both experiments.

    python run_experiment.py --e1              # live agent runs -> trajectories
    python run_experiment.py --e1 --mock       # scripted model, no API calls
    python run_experiment.py --e1 --verify-reuse
    python run_experiment.py --e2              # detection, sanitization, access control
    python run_experiment.py --all             # both, then the report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentlab.config import RUNS_DIR, load_settings, rate_card_for  # noqa: E402
from agentlab.fakemodel import build_scripted_model  # noqa: E402
from agentlab.llm import describe_target  # noqa: E402
from agentlab.recorder import run_scenario  # noqa: E402
from agentlab.replay import (  # noqa: E402
    reconfigure_reuse_report,
    replay_trajectory,
    resume_demo,
    verify_reconstruction,
)
from agentlab.scenarios import SCENARIOS, by_id, selected  # noqa: E402
from agentlab.stats import collect  # noqa: E402
from agentlab.store import RunWriter, TrajectoryStore  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

KINDS = {s.scenario_id: s.kind for s in SCENARIOS}


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# --- experiment 1 ----------------------------------------------------------


def run_experiment_one(args: argparse.Namespace) -> int:
    _banner("Experiment 1 - real agent trajectory logging")

    settings = load_settings(require_key=not args.mock)
    scenarios = selected(args.scenarios.split(",") if args.scenarios else None)
    repeats = args.repeats

    # The sandbox's synthetic credentials are generated rather than committed,
    # because credential-shaped fixtures trip secret scanners. This keeps a
    # fresh clone runnable without shipping anything that looks like a key.
    from agentlab.workspace_fixture import ensure_workspace

    seeded = ensure_workspace(settings.sandbox_root)
    if seeded:
        print(f"seeded workspace fixtures: {', '.join(seeded)}")

    if args.mock:
        print("mode        : SCRIPTED TEST DOUBLE (no API calls; not reportable)")
    else:
        print(f"mode        : LIVE - {describe_target(settings)}")
    print(f"sandbox     : {settings.sandbox_root}")
    print(f"runs dir    : {settings.runs_dir}")
    print(f"scenarios   : {len(scenarios)} x {repeats} repeats = {len(scenarios) * repeats} runs")

    written: list[Path] = []
    reconstructions: list[tuple[str, dict]] = []
    for scenario in scenarios:
        for repeat in range(repeats):
            # Flat layout: runs/<scenario>__r<n>/trajectory.json, so a re-run
            # replaces its own run rather than accumulating duplicates.
            run_dir = settings.runs_dir / f"{scenario.scenario_id}__r{repeat}"
            writer = RunWriter(run_dir)
            # Fresh journal per run: append-only means append within a run.
            writer.journal_path.unlink(missing_ok=True)

            model = build_scripted_model(scenario.scenario_id) if args.mock else None
            try:
                result = run_scenario(
                    settings,
                    scenario,
                    repeat_index=repeat,
                    model=model,
                    journal=writer.journal_step,
                )
            except Exception as exc:
                print(f"  [FAIL] {scenario.scenario_id} r{repeat}: {type(exc).__name__}: {exc}")
                continue

            # The reconstruction check must run here, against the message list
            # this very run produced. Comparing a stored artifact with a later
            # run would be meaningless: a live model is not deterministic.
            reconstructions.append(
                (scenario.scenario_id, verify_reconstruction(result.trajectory, result.messages))
            )

            path = writer.finalize(result.trajectory)
            written.append(path)
            outcome = result.trajectory.outcome
            chain_ok, bad = __import__(
                "agentlab.schema", fromlist=["verify_chain"]
            ).verify_chain(result.trajectory.steps, result.trajectory.integrity.prev_run_hash)
            print(
                f"  {scenario.scenario_id:22s} r{repeat}  steps={outcome.total_steps:2d} "
                f"tools={outcome.total_tool_calls} tokens={outcome.total_tokens:6d} "
                f"chain={'ok' if chain_ok else f'BAD@{bad}'} stop={outcome.stop_reason}"
            )

    if not written:
        print("\nno runs completed")
        return 1

    store = TrajectoryStore(settings.runs_dir)
    trajectories = [
        t for t in store.iter_trajectories() if t.scenario_id in {s.scenario_id for s in scenarios}
    ]

    _banner("Reuse evidence")
    _verify_reuse(settings, store, mock=args.mock, reconstructions=reconstructions)

    _banner("Statistics")
    stats = collect(trajectories, rate_card=rate_card_for(settings.model), kinds=KINDS)
    markdown = stats.to_markdown()
    print(markdown)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = METRICS_DIR / ("e1_stats_mock.md" if args.mock else "e1_stats.md")
    stats_path.write_text(markdown, encoding="utf-8")
    (METRICS_DIR / ("e1_totals_mock.json" if args.mock else "e1_totals.json")).write_text(
        json.dumps(
            {
                "totals": stats.totals(),
                "cost": stats.cost_estimate(),
                "per_scenario": stats.per_scenario(),
                "mode": "mock" if args.mock else "live",
                "model": settings.model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {stats_path}")

    checks = store.verify_all()
    bad = [c for c in checks if not c[1]]
    print(f"\nchain verification: {len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


def run_stats_only(args: argparse.Namespace) -> int:
    """Recompute statistics and reuse evidence over everything already stored.

    Useful after adding runs to a subset of scenarios: the agent is not
    re-invoked, and the reuse checks run offline against the stored artifacts
    so the evidence is reproducible without repeating the calls.
    """
    _banner("Recomputing over stored trajectories")
    settings = load_settings(require_key=False)
    store = TrajectoryStore(settings.runs_dir)
    trajectories = list(store.iter_trajectories())
    if not trajectories:
        print("no stored trajectories")
        return 1
    print(f"trajectories: {len(trajectories)}")

    _banner("Reuse evidence (offline, from the stored artifacts)")
    offline = reconfigure_reuse_report(settings, trajectories)
    print(
        f"[1] reconstruct  {offline['reconstruction_identical']}/{offline['runs_checked']} "
        "runs digest-identical"
    )
    for recon in offline["reconstruction"]:
        if recon["identical"]:
            continue
        print(f"      MISMATCH {recon['run_id'][:12]}: {recon['differences'][:3]}")
    print(
        f"[2] replay       {offline['replay_identical']}/{offline['replay_total']} runs "
        "replayed identically"
    )
    for report in offline["replay"]:
        if report["ok"]:
            continue
        print(f"      MISMATCH {report['run_id'][:12]}: {report['mismatches'][:2]}")

    from langgraph.checkpoint.memory import InMemorySaver

    resume_scenario = by_id("config-summary")
    resume_report = resume_demo(
        settings, resume_scenario, checkpointer=InMemorySaver(), model=None
    )
    print(
        f"[3] resume       interrupted_before_tools={resume_report['interrupted_before_tools']} "
        f"resumed_tool_calls={resume_report['second_half_tool_calls']} "
        f"linked={resume_report['resumed_tool_calls_linked_to_original_request']} "
        f"ok={resume_report['resumed_successfully']}"
    )

    _banner("Statistics")
    stats = collect(trajectories, rate_card=rate_card_for(settings.model), kinds=KINDS)
    markdown = stats.to_markdown()
    print(markdown)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "e1_stats.md").write_text(markdown, encoding="utf-8")
    (METRICS_DIR / "e1_totals.json").write_text(
        json.dumps(
            {
                "totals": stats.totals(),
                "cost": stats.cost_estimate(),
                "per_scenario": stats.per_scenario(),
                "mode": "live",
                "model": settings.model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (METRICS_DIR / "e1_reuse.json").write_text(
        json.dumps(
            {
                "reconstruction_identical": offline["reconstruction_identical"],
                "runs_checked": offline["runs_checked"],
                "replay_identical": offline["replay_identical"],
                "replay_total": offline["replay_total"],
                "resume": resume_report,
                "reconstruction_detail": offline["reconstruction"],
                "replay_detail": offline["replay"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checks = store.verify_all()
    bad = [c for c in checks if not c[1]]
    print(f"chain verification: {len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


def _verify_reuse(
    settings, store: TrajectoryStore, *, mock: bool, reconstructions: list[tuple[str, dict]]
) -> None:
    """Run the three reuse checks.

    Reconstruction is measured *inside* the run that produced the trajectory.
    An earlier version compared a stored artifact against a fresh run, which is
    meaningless: a live model is not deterministic, so the two message lists
    differ for reasons that have nothing to do with the recorder. Replay and
    resume are deterministic given an artifact and can be checked afterwards.
    """
    candidates = [t for t in store.iter_trajectories() if t.outcome.total_tool_calls >= 1]
    if not candidates:
        print("no run with tool calls available; skipping reuse checks")
        return
    target = max(candidates, key=lambda t: t.outcome.total_tool_calls)

    if reconstructions:
        identical = sum(1 for _sid, recon in reconstructions if recon["identical"])
        print(f"[1] reconstruct  {identical}/{len(reconstructions)} runs digest-identical")
        for scenario_id, recon in reconstructions:
            if recon["identical"]:
                continue
            print(
                f"      MISMATCH {scenario_id}: rebuilt {recon['rebuilt_messages']} vs "
                f"live {recon['live_messages']}"
            )
            for difference in recon["differences"][:5]:
                print(f"        {difference}")
    else:
        print("[1] reconstruct  no runs recorded in this invocation")

    _, replay_report = replay_trajectory(settings, target)
    print(
        f"[2] replay       tool_calls {replay_report.replayed_tool_calls}/"
        f"{replay_report.original_tool_calls} identical={replay_report.identical_sequence} "
        f"misses={replay_report.replay_misses}"
    )
    for mismatch in replay_report.mismatches[:5]:
        print(f"      {mismatch}")

    from langgraph.checkpoint.memory import InMemorySaver

    resume_scenario = by_id("config-summary")
    resume_report = resume_demo(
        settings,
        resume_scenario,
        checkpointer=InMemorySaver(),
        model=build_scripted_model(resume_scenario.scenario_id) if mock else None,
    )
    print(
        f"[3] resume       interrupted_before_tools={resume_report['interrupted_before_tools']} "
        f"resumed_tool_calls={resume_report['second_half_tool_calls']} "
        f"linked={resume_report['resumed_tool_calls_linked_to_original_request']} "
        f"ok={resume_report['resumed_successfully']}"
    )


# --- experiment 2 ----------------------------------------------------------


def run_experiment_two(args: argparse.Namespace) -> int:
    _banner("Experiment 2 - trajectory security")
    from trajsec.cli import main as trajsec_main

    argv: list[str] = ["reproduce"]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.skip_redteam:
        argv += ["--skip-redteam"]
    code = trajsec_main(argv)
    if code != 0:
        return code

    from trajsec.report import generate

    path = generate()
    print(f"\nwrote report: {path}")
    return 0


# --- entry -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--e1", action="store_true", help="run Experiment 1")
    group.add_argument("--e2", action="store_true", help="run Experiment 2 over stored trajectories")
    group.add_argument("--all", action="store_true", help="run both, then write the report")
    group.add_argument(
        "--stats-only",
        action="store_true",
        help="recompute statistics and offline reuse evidence over stored runs; no API calls",
    )

    parser.add_argument("--mock", action="store_true", help="use the scripted test double (no API calls)")
    parser.add_argument("--repeats", type=int, default=4, help="repeats per scenario (default 4)")
    parser.add_argument("--scenarios", default=None, help="comma-separated scenario ids")
    parser.add_argument("--verify-reuse", action="store_true", help="run the reuse checks")
    parser.add_argument("--seed", type=int, default=None, help="Experiment 2 corpus seed")
    parser.add_argument("--skip-redteam", action="store_true", help="skip the adversarial suite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stats_only:
        return run_stats_only(args)
    if args.e1:
        return run_experiment_one(args)
    if args.e2:
        return run_experiment_two(args)
    code = run_experiment_one(args)
    if code != 0:
        print("\nExperiment 1 did not complete cleanly; stopping before Experiment 2.")
        return code
    return run_experiment_two(args)


if __name__ == "__main__":
    raise SystemExit(main())
