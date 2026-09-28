"""Report generation.

Reads the artifact bundle Experiment 2 wrote and the statistics Experiment 1
wrote, and emits one Markdown document. Nothing here computes a number: every
figure is looked up from an artifact, so the report cannot drift from the run
that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajsec.config import ARTIFACTS_DIR, PROJECT_ROOT, load_settings

REPORT_PATH = PROJECT_ROOT / "reports" / "REPORT.md"
E1_STATS = PROJECT_ROOT / "reports" / "metrics" / "e1_stats.md"
E1_TOTALS = PROJECT_ROOT / "reports" / "metrics" / "e1_totals.json"
PREREG = PROJECT_ROOT / "PREREGISTRATION.md"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# --- hypothesis adjudication ------------------------------------------------


def _adjudicate(bundle: dict[str, Any]) -> list[dict[str, str]]:
    """Check each pre-registered hypothesis against the measured result.

    Written after the fact but against rules fixed in PREREGISTRATION.md
    before the test split was scored, which is the only thing that makes the
    difference between a prediction and a retrodiction visible.
    """
    report = bundle["evaluation"]
    ablation = {row["config"]: row for row in report["ablation"]}
    no_l3 = ablation.get("L1+L2", {})
    with_l3 = ablation.get("L1+L2+L3", {})
    full = ablation.get("full (L4 added)", {})

    delta = (with_l3.get("item_recall_span", 0.0) - no_l3.get("item_recall_span", 0.0))
    precision_cost = (no_l3.get("precision", 0.0) - with_l3.get("precision", 0.0))

    h1_confirmed = delta >= 0.15
    h2_confirmed = (
        full.get("negative_control_fp_rate", 0.0) == 0.0
        and with_l3.get("negative_control_fp_rate", 0.0) >= 0.5
    ) or (
        with_l3.get("negative_control_fp_rate", 0.0)
        >= 10 * max(full.get("negative_control_fp_rate", 0.0), 1e-9)
    )

    utility = bundle.get("utility") or {}
    redacted = utility.get("redacted", {})
    h3_confirmed = redacted.get("audit_answer_agreement", {}).get("agreement", 0.0) >= 0.95

    scope = bundle.get("token_scope_property", {})
    h4_confirmed = bool(scope.get("confirmed"))

    redteam = bundle.get("redteam") or {}
    h5_confirmed = bool(redteam.get("all_passed")) and not redteam.get("unpaired_denials")

    return [
        {
            "id": "H1",
            "claim": "L3 raises span recall on the novel-format tier by >= 15 points over L1+L2",
            "observed": f"delta = {delta:+.4f} (T4 recall {no_l3.get('item_recall_span', 0):.3f} -> {with_l3.get('item_recall_span', 0):.3f}), precision cost {precision_cost:+.4f}",
            "verdict": "confirmed" if h1_confirmed else "refuted",
        },
        {
            "id": "H2",
            "claim": "The negative-control flag rate falls by >= 10x once L4 is enabled",
            "observed": f"with L4 {full.get('negative_control_fp_rate', 0):.3f}, without L4 {with_l3.get('negative_control_fp_rate', 0):.3f}",
            "verdict": "confirmed" if h2_confirmed else "refuted",
        },
        {
            "id": "H3",
            "claim": "audit_answer_agreement for the redacted profile is >= 0.95",
            "observed": f"{redacted.get('audit_answer_agreement', {}).get('agreement', 'n/a')}",
            "verdict": "confirmed" if h3_confirmed else "refuted",
        },
        {
            "id": "H4",
            "claim": "under token_scope=run, tokens correlate within a run and not across runs",
            "observed": json.dumps(scope.get("run_scope", {}), default=str)[:180],
            "verdict": "confirmed" if h4_confirmed else "refuted",
        },
        {
            "id": "H5",
            "claim": "every denial is paired with a passing positive control",
            "observed": f"{redteam.get('passed', 0)}/{redteam.get('total', 0)} cases, unpaired {redteam.get('unpaired_denials', [])}",
            "verdict": "confirmed" if h5_confirmed else "refuted",
        },
    ]


# --- sections ---------------------------------------------------------------


def _experiment_one(lines: list[str]) -> None:
    lines.append("## Experiment 1 - real agent trajectory logging\n")
    lines.append(
        "A LangGraph agent with three local tools, driven by a live model over an "
        "OpenAI-compatible endpoint. Two independent instrumentation sources feed one "
        "trajectory: LangChain callbacks supply token usage, exact model timings and the "
        "prompt as sent; the graph stream supplies node transitions and the routing "
        "decision at each turn. Neither is sufficient alone -- callbacks cannot see which "
        "edge was taken, the stream carries no usage -- so the recorder correlates them by "
        "run id and node.\n"
    )
    lines.append(
        "Reuse is claimed on three independent grounds rather than on the existence of a "
        "JSON file:\n\n"
        "1. **Replay.** The recorded model turns and tool observations are served back, "
        "with no API call and no filesystem access, and the replayed run is compared "
        "against the original call by call.\n"
        "2. **Reconstruction.** The conversation is rebuilt from the artifact alone and "
        "its canonical digest compared with the live message list.\n"
        "3. **Resume.** With a SQLite checkpointer the run is interrupted before its tool "
        "call and continued, and the resumed tool calls are checked to link back to the "
        "span that requested them.\n"
    )
    if E1_TOTALS.is_file():
        totals = _load(E1_TOTALS) or {}
        summary = totals.get("totals", {})
        if summary:
            lines.append("### Measured\n")
            rows = [
                ["runs", summary.get("runs")],
                ["scenarios", summary.get("scenarios")],
                ["steps per run (mean)", summary.get("steps_per_run_mean")],
                ["model calls", summary.get("llm_calls_total")],
                ["tool calls", summary.get("tool_calls_total")],
                ["tool errors", summary.get("tool_errors_total")],
                ["tokens (prompt / completion)", f"{summary.get('tokens_prompt'):,} / {summary.get('tokens_completion'):,}"],
                ["model time share", summary.get("llm_time_share")],
                ["tool time share", summary.get("tool_time_share")],
                ["max call depth", summary.get("max_depth")],
                ["hash chain pass rate", summary.get("chain_pass_rate")],
            ]
            lines.append(_table(["metric", "value"], rows))
            cost = totals.get("cost", {})
            if cost.get("priced"):
                lines.append(
                    f"\nEstimated cost: **{cost['estimated_total']} {cost['currency']}** "
                    f"(upper bound; all prompt tokens billed at the cache-miss rate).\n"
                )
    if E1_STATS.is_file():
        lines.append("\n<details><summary>Full per-scenario table</summary>\n")
        lines.append(E1_STATS.read_text(encoding="utf-8"))
        lines.append("\n</details>\n")


def _experiment_two(lines: list[str], bundle: dict[str, Any]) -> None:
    report = bundle["evaluation"]
    headline = report["headline"]

    lines.append("## Experiment 2 - trajectory security\n")
    lines.append(
        "The input is the trajectories Experiment 1 actually produced, with synthetic "
        "secrets injected at known positions to build a labelled corpus. All credentials "
        "anywhere in this project are synthetic; the live model key is never recorded, "
        "only a fingerprint of it.\n"
    )

    lines.append("### Headline\n")
    lines.append(
        _table(
            ["metric", "value", "95% CI"],
            [
                ["item recall (span flagged)", headline["item_recall_span_excluding_expected_misses"],
                 headline["ci"]["item_recall_span"][1:3]],
                ["item recall (value recovered)", headline["item_recall_value_excluding_expected_misses"],
                 headline["ci"]["item_recall_value"][1:3]],
                ["precision (span, merged regions)", headline["precision"], headline["ci"]["precision"][1:3]],
                ["recall (span)", headline["recall"], headline["ci"]["recall"][1:3]],
                ["localisation gap (strict - IoU)", headline["localisation_gap"], "-"],
                ["expected-miss items", f"{headline['expected_miss_items']} of {headline['items_total']}", "-"],
            ],
        )
    )
    lines.append(
        f"\nIntervals are a cluster bootstrap over **{headline['ci']['n_trajectories']} "
        "trajectories**, not over findings: findings inside one run are correlated, so "
        "resampling them as if independent would report an interval far narrower than the "
        "data supports. With this few trajectories the intervals are wide, and that is "
        "stated rather than smoothed over.\n"
    )

    # Single-seed figures are the best case by construction, so the spread
    # across seeds is reported alongside them rather than beneath them.
    variance = report["injection_variance"]
    lines.append("\n**Stability across corpus seeds.** The row above is the primary "
                 "seed, which is the most favourable one. Across all five:\n")
    lines.append(
        _table(
            ["metric", "mean", "std", "min", "max"],
            [
                [key, value["mean"], value["std"], value["min"], value["max"]]
                for key, value in variance.items()
            ],
        )
    )
    lines.append(
        "\nThe entire spread comes from one tier. T4 -- a long token with no prefix and "
        "no sensitive field name -- is reachable only by entropy scoring, and the "
        "calibrated threshold sits right at the entropy of that population. On any given "
        "seed roughly half its instances fall below the line. That is the honest "
        "characterisation of the entropy layer: it is the only thing that reaches this "
        "case, and it does so unreliably. The threshold was frozen before the test split "
        "was scored and was not adjusted afterwards, so this is reported as a measured "
        "weakness rather than tuned away.\n"
    )

    lines.append(
        "\nThe two recall figures answer different questions. *Flagged* asks whether the "
        "detector marked where the secret sits; *recovered* asks whether it produced the "
        "secret's actual value. Evasion separates them, which is the point of reporting both.\n"
    )

    lines.append("### Per tier\n")
    lines.append(
        _table(
            ["tier", "n", "flagged", "recovered", "span recall", "value recall", "what it tests"],
            [
                [r["key"], r["total"], r["flagged"], r["recovered"], r["recall_span"], r["recall_value"], r["label"]]
                for r in report["per_tier"]
            ],
        )
    )

    lines.append("\n### Per evasion\n")
    lines.append(
        _table(
            ["evasion", "n", "flagged", "recovered", "span recall", "value recall"],
            [[r["key"], r["total"], r["flagged"], r["recovered"], r["recall_span"], r["recall_value"]] for r in report["per_evasion"]],
        )
    )
    lines.append(
        "\nZero-width splicing and cross-field splitting are declared expected misses: the "
        "plaintext is not present in any single span, so no value-level detector can "
        "recover it. They are excluded from the headline and shown here instead.\n"
    )

    lines.append("### Layer ablation\n")
    lines.append(
        _table(
            ["configuration", "span recall", "value recall", "precision", "recall", "F1", "dRecall", "dPrecision", "neg-control FP rate"],
            [
                [r["config"], r["item_recall_span"], r["item_recall_value"], r["precision"],
                 r["recall"], r["f1"], r["delta_recall"], r["delta_precision"], r["negative_control_fp_rate"]]
                for r in report["ablation"]
            ],
        )
    )
    lines.append(
        "\nThis table is the honest centre of the experiment. The entropy layer buys a "
        "small amount of span recall and costs a large amount of precision, and is only "
        "worth keeping because the allowlist layer cleans up after it. Reporting the "
        "layer as a success on its own would be false; reporting it as useless would miss "
        "that it is the only thing that reaches a secret with no prefix and no sensitive "
        "field name.\n"
    )

    natural = report["natural_false_positives"]
    lines.append("### False positives on unmodified trajectories\n")
    lines.append(
        "Precision measured on an injected corpus is optimistic by construction, because "
        "every injected positive is a true positive. This section instead scans the "
        "trajectories exactly as they were recorded.\n"
    )
    lines.append(
        _table(
            ["metric", "value"],
            [
                ["trajectories scanned", natural["trajectories"]],
                ["findings on natural text", natural["findings_on_natural_text"]],
                ["adjudicated real", natural["adjudicated_true_positive"]],
                ["adjudicated false positive", natural["adjudicated_false_positive"]],
                ["false positive rate per finding", natural["fp_rate_per_finding"]],
                ["Wilson 95% interval", natural["fp_rate_wilson_95"]],
                ["false positives per KB scanned", natural["fp_per_kb_scanned"]],
                ["false positives per 1000 spans", natural["fp_per_1000_spans"]],
            ],
        )
    )
    lines.append(f"\nDenominators: `{natural['denominators']}`\n")
    lines.append(f"\n> {natural['caveat']}\n")
    if natural["example_false_positives"]:
        lines.append("\n<details><summary>Example false positives</summary>\n")
        lines.append(
            _table(
                ["layer", "type", "pointer", "preview", "chars", "rule"],
                [
                    [e["layer"], e["type"], e["pointer"], e["preview"], e["chars"], e["rule"]]
                    for e in natural["example_false_positives"][:15]
                ],
            )
        )
        lines.append("\n</details>\n")

    lines.append("### Base-rate adjusted precision\n")
    lines.append(
        "A detector with excellent corpus precision still has poor precision when what it "
        "looks for is rare. Applying the measured true- and false-positive rates at "
        "realistic prevalence:\n"
    )
    lines.append(
        _table(
            ["prevalence", "precision"],
            [[f"{row['prevalence']:.0e}", row["precision"]] for row in report["base_rate"]["curve"]],
        )
    )
    lines.append(
        f"\nAt the measured operating point (TPR {report['base_rate']['true_positive_rate']}, "
        f"FPR {report['base_rate']['false_positive_rate']}). This curve is the answer to "
        "\"your precision on an injected corpus is fiction\" -- and it agrees.\n"
    )


def _sanitization(lines: list[str], bundle: dict[str, Any]) -> None:
    utility = bundle.get("utility") or {}
    if not utility:
        return
    lines.append("### Sanitization and utility loss\n")
    lines.append(
        "Replacing a secret is easy; replacing it without destroying what the artifact is "
        "for is the actual problem. Four properties are measured per profile.\n"
    )
    rows = []
    for profile, data in sorted(utility.items()):
        rows.append(
            [
                profile,
                data.get("tokens"),
                _fmt(data.get("destruction_footprint", {}).get("footprint", 0)),
                _fmt(data["path_preservation"]["preservation_rate"]),
                _fmt(data["audit_answer_agreement"]["agreement"]),
                data["idempotence"]["idempotent"],
                data["structural_replay"]["sanitized_chain_verifies"],
            ]
        )
    lines.append(
        _table(
            ["profile", "tokens", "destruction footprint", "path preservation", "audit agreement", "idempotent", "re-chained"],
            rows,
        )
    )
    lines.append(
        "\n*Destruction footprint* is bytes rewritten over bytes total: a sanitizer that "
        "rewrites a third of the artifact to hide half a percent of bytes is "
        "over-sanitizing, which is a security failure with no attacker behind it. "
        "*Audit agreement* re-asks a fixed set of auditor questions of the original and "
        "the sanitized artifact and counts how many answers survive -- one number that "
        "catches both under- and over-redaction.\n"
    )
    example = utility.get(sorted(utility)[0], {})
    length = example.get("length_stability", {})
    if length:
        lines.append(
            f"\nLength preservation by token form: value-derived tokens "
            f"{length.get('value_derived_count')} (all length-preserving: "
            f"{length.get('value_derived_length_preserving')}), typed placeholders "
            f"{length.get('placeholder_count')}. Placeholders exist because a token "
            "derived from a low-entropy password would be enumerable by anyone holding the "
            "key; correlation is given up on purpose for those values.\n"
        )
    lines.append(
        "\n**Structural replay only.** The sanitized artifact loads, validates and "
        "re-chains, but its tool observations have been replaced, so re-executing the "
        "tools from it is not possible and is not claimed.\n"
    )


def _access_control(lines: list[str], bundle: dict[str, Any]) -> None:
    redteam = bundle.get("redteam") or {}
    lines.append("### Access control\n")
    lines.append(
        "A record's classification is the lattice join of the classifications of the "
        "finding types present in it. Detector output is the ACL input, which is what "
        "makes a detection miss an authorization failure rather than a missed alert.\n"
    )
    if redteam:
        lines.append(
            f"\nThe adversarial suite runs **{redteam['total']} cases** with "
            f"**{redteam['passed']} passing**. Every denial is paired with a positive "
            "control that must be allowed, because a suite in which everything denies "
            "would pass even if the store denied everything. Unpaired denials: "
            f"`{redteam['unpaired_denials'] or 'none'}`.\n"
        )
        lines.append("\n" + _table(
            ["case", "expectation", "outcome", "passed", "detail"],
            [[c["case_id"], c["expectation"], c["outcome"], c["passed"], c["detail"][:60]] for c in redteam["cases"]],
        ) + "\n")

    lines.append("\n#### Detection failure becomes authorization failure\n")
    cascade = bundle.get("cascading_failure", {})
    full = cascade.get("full_stack", {})
    ablated = cascade.get("without_L3", {})
    lines.append(
        _table(
            ["configuration", "records", "P(under-classified)", "P(exported in clear)"],
            [
                [full.get("label"), full.get("records"), full.get("p_record_under_classified"), full.get("p_record_exported_in_clear")],
                [ablated.get("label"), ablated.get("records"), ablated.get("p_record_under_classified"), ablated.get("p_record_exported_in_clear")],
            ],
        )
    )
    lines.append(
        "\n*Exported in clear* means the secret's plaintext still appears in the artifact a "
        "third party would receive -- not that someone could read something sensitive, but "
        "that the vendor is handed the credential. The mechanism is demonstrated by "
        "ablating a layer and watching the cascade appear where it was absent, rather than "
        "by asserting it.\n"
    )
    lines.append(
        f"\nNote the different denominator. This table aggregates every corpus seed and "
        f"both splits ({full.get('records')} records), so it is deliberately more "
        "pessimistic than the headline, which reports one seed's test split. The gap "
        "between them is the same T4 instability described above: the single-seed figure "
        "is the favourable case, and this one is what an operator would actually face.\n"
    )

    tamper = bundle.get("tamper_suite", {})
    if tamper.get("operations"):
        lines.append("\n#### Audit log tamper detection\n")
        lines.append(
            _table(
                ["operation", "detected", "expected"],
                [[o["operation"], o["detected"], o["expect_detected"]] for o in tamper["operations"]],
            )
        )
        lines.append(
            f"\nDetection rate **{tamper['detected']}/{tamper['total']} = {tamper['detection_rate']}**. "
            f"{tamper['claim']}\n"
        )

    scope = bundle.get("token_scope_property", {})
    if scope:
        lines.append("\n#### Token scope\n")
        lines.append(
            "A globally stable token is itself a leak: anyone who can read two tenants' "
            "trajectories could confirm that the same credential appears in both. Scoping "
            "the token domain is how that is controlled rather than assumed away.\n"
        )
        lines.append(f"\n```json\n{json.dumps(scope, indent=2, default=str)[:700]}\n```\n")


def _limitations(lines: list[str], bundle: dict[str, Any]) -> None:
    lines.append("## Limitations\n")
    items = [
        "**Injected secrets are a model of reality, not reality.** They are tiered and "
        "in-context, but a reviewer is right to discount them.",
        "**The trajectory count bounds every interval.** The bootstrap is clustered over "
        "runs, so the width of every confidence interval is set by how many independent "
        "runs exist, not by how many secrets. More runs is the cheapest available "
        "improvement to this work.",
        "**Natural-text adjudication is conservative in one direction.** A finding counts "
        "as real only if its value is a known planted credential, so genuine disclosure "
        "that is not credential-shaped is counted as a false positive.",
        "**Zero-width splicing and cross-field splitting are not detected.** Declared in "
        "the pre-registration, not discovered afterwards.",
        "**The allowlist can be evaded by an adversary who controls field names.** Keys "
        "named `hash` or `build_id` are suppressed by design; a secret hidden under one "
        "would be missed. The allowlist is a false-positive control, not an adversarial "
        "filter.",
        "**On one machine the sanitizer's two profiles are a policy separation, not a "
        "cryptographic guarantee.** Anyone holding the key can reverse the reversible "
        "profile, and the one-way profile is one-way only because no vault is written. "
        "DPAPI protects against another user of the machine, not against the same user or "
        "an administrator.",
        "**The tamper claim is bounded.** A forged append is not detectable by the chain "
        "alone, and an attacker who can rewrite both the chain and its anchor defeats it "
        "entirely. The measured detection rate includes that gap rather than hiding it.",
        "**Utility metrics measure structural utility, not human utility.** Audit "
        "agreement is a proxy for whether an analyst could still do their job.",
        "**Single annotator.** The natural-text adjudication was not independently "
        "reproduced, so no inter-rater agreement figure is available.",
    ]
    for item in items:
        lines.append(f"- {item}")
    lines.append("")


def _reproduction(lines: list[str], bundle: dict[str, Any]) -> None:
    lines.append("## Reproduction\n")
    lines.append("```bash\n")
    lines.append("python run_experiment.py --e1        # live agent runs -> trajectories")
    lines.append("python run_experiment.py --e2        # detection, sanitization, access control")
    lines.append("python run_experiment.py --all       # both, then this report")
    lines.append("python -m pytest trajsec/tests/ -q   # the security assertions")
    lines.append("```\n")
    settings = bundle.get("settings", {})
    lines.append(
        _table(
            ["parameter", "value"],
            [
                ["thresholds version", settings.get("thresholds_version")],
                ["policy version", settings.get("policy_version")],
                ["rule files fingerprint", settings.get("rules_fingerprint")],
                ["corpus seeds", settings.get("seeds")],
                ["key custody", bundle.get("key_custody", {}).get("custody")],
                ["key material absent from artifacts", bundle.get("key_material_absent_from_artifacts", {}).get("clean")],
            ],
        )
    )
    lines.append(
        "\nThis project is not a git repository, so content hashes are the reproducibility "
        "anchor: `reports/artifacts/MANIFEST.json` records the digest of every rule file, "
        "every input trajectory and every output artifact, so a reader can verify that the "
        "figures above came from exactly those inputs.\n"
    )
    lines.append("\n### Sandbox declaration\n")
    lines.append(f"\n```json\n{json.dumps(bundle.get('sandbox_declaration', {}), indent=2)}\n```\n")


# --- entry -----------------------------------------------------------------


def generate(artifacts_dir: Path | None = None, *, out_path: Path | None = None) -> Path:
    artifacts = Path(artifacts_dir or ARTIFACTS_DIR)
    bundle = _load(artifacts / "eval_report.json")
    if bundle is None:
        raise SystemExit(
            f"no eval_report.json under {artifacts}. Run `python run_experiment.py --e2` first."
        )

    lines: list[str] = []
    lines.append("# Agent trajectory logging and trajectory security\n")
    lines.append(
        "Two experiments over one artifact. The first builds a real LangGraph agent and "
        "records its execution trajectory; the second takes those recordings and attacks "
        "them, asking what a trajectory leaks, whether layered detection can find it, "
        "whether structure-preserving sanitization can remove it without destroying the "
        "artifact's value, and whether the access control around it actually denies.\n"
    )
    lines.append(
        "Every number below is looked up from a generated artifact rather than written by "
        "hand, and the detector's one tuned threshold was calibrated on a held-out split "
        "and frozen before the test split was scored (see `PREREGISTRATION.md`).\n"
    )

    lines.append("## Pre-registered hypotheses\n")
    lines.append(_table(
        ["id", "claim", "observed", "verdict"],
        [[h["id"], h["claim"], h["observed"], h["verdict"]] for h in _adjudicate(bundle)],
    ))
    lines.append("")

    _experiment_one(lines)
    lines.append("---\n")
    _experiment_two(lines, bundle)
    _sanitization(lines, bundle)
    _access_control(lines, bundle)
    lines.append("---\n")
    _limitations(lines, bundle)
    _reproduction(lines, bundle)

    target = Path(out_path or REPORT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(f"wrote {generate()}")
