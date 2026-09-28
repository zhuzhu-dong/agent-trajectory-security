"""Evaluation.

Every number in the report comes from here. Three things this module is
careful about, because they are where evaluations usually cheat by accident:

* **The matching unit.** Precision is computed over *merged prediction
  regions*, not individual findings. Otherwise two layers flagging the same
  span would score as one true positive and one false positive.
* **The resampling unit.** Confidence intervals come from a cluster bootstrap
  over *trajectories*. Findings inside one run are correlated -- one
  over-firing rule inflates false positives across every finding in that run --
  so treating findings as independent would understate the interval badly.
* **The base rate.** Precision measured on an injected corpus says nothing
  about precision at realistic prevalence, so the base-rate curve is reported
  alongside it rather than instead of it.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from canonical import canonical_json, sha256_hex
from textindex import resolve_pointer

from trajsec.config import RULES_DIR, Settings, load_settings
from trajsec.corpus import EXPECTED_MISSES, TIER_DESCRIPTION
from trajsec.detect import scan_document
from trajsec.model import CorpusEntry, GroundTruthItem, Manifest, canonicalize_secret

# --- layer configurations for the ablation ---------------------------------

LAYER_CONFIGS: dict[str, tuple[str, ...]] = {
    "L1_only": ("L1",),
    "L1+L2": ("L2", "L1"),
    "L1+L2+L3": ("L2", "L1", "L3"),
    "full (L4 added)": ("L4_deny", "L2", "L1", "L3", "L4_allow"),
}


# --- ground truth helpers --------------------------------------------------


def load_fixtures() -> dict[str, Any]:
    return json.loads((RULES_DIR / "workspace_fixtures.json").read_text(encoding="utf-8"))


def fixture_hashes(fixtures: dict[str, Any] | None = None) -> dict[str, str]:
    """``{value_sha256: type}`` for secrets genuinely present in the workspace."""
    fixtures = fixtures or load_fixtures()
    return {
        sha256_hex(canonicalize_secret(entry["value"])): entry["type"]
        for entry in fixtures["values"]
    }


def finding_value(doc: dict[str, Any], pointer: str, start: int, end: int) -> str:
    """Recover the matched text for evaluation.

    Findings deliberately carry no plaintext, so the evaluator reads it back
    out of the document it already holds. That keeps the store safe to hand to
    a third party while still allowing ground-truth comparison here.
    """
    try:
        node = resolve_pointer(doc, pointer)
    except (KeyError, IndexError, ValueError):
        return ""
    if not isinstance(node, str):
        return ""
    return node[max(0, start) : min(len(node), end)]


@dataclass(frozen=True)
class Span:
    pointer: str
    start: int
    end: int
    type: str
    kind: str  # "injected" | "natural"
    tier: str = ""
    evasion: str = ""


def natural_positives(doc: dict[str, Any], fixtures: dict[str, Any] | None = None) -> list[Span]:
    """Spans in the document that genuinely carry sensitive material.

    Two sources: the exact synthetic values planted in the workspace, and
    pattern matches for the classes that are disclosures without being
    credentials (account names in paths, internal hostnames).
    """
    fixtures = fixtures or load_fixtures()
    wanted = {entry["value"]: entry["type"] for entry in fixtures["values"]}
    patterns = [
        (re.compile(p), "INTERNAL_TOPOLOGY") for p in fixtures.get("hostname_patterns", [])
    ] + [(re.compile(p), "PII_PATH_USERNAME") for p in fixtures.get("path_patterns", [])]

    spans: list[Span] = []
    from textindex import iter_leaves

    for pointer, _path, value in iter_leaves(doc):
        if not isinstance(value, str) or not value:
            continue
        for needle, type_id in wanted.items():
            cursor = 0
            while True:
                found = value.find(needle, cursor)
                if found < 0:
                    break
                spans.append(Span(pointer, found, found + len(needle), type_id, "natural"))
                cursor = found + len(needle)
        for pattern, type_id in patterns:
            for match in pattern.finditer(value):
                spans.append(
                    Span(pointer, match.start(), match.end(), type_id, "natural")
                )
    return spans


# --- prediction regions ----------------------------------------------------


@dataclass
class Region:
    pointer: str
    start: int
    end: int
    types: set[str] = field(default_factory=set)
    layers: set[str] = field(default_factory=set)
    findings: int = 0


def merge_regions(findings: Iterable[Any]) -> list[Region]:
    """Union overlapping active findings per pointer into prediction regions."""
    by_pointer: dict[str, list[Any]] = {}
    for finding in findings:
        if finding.suppressed:
            continue
        by_pointer.setdefault(finding.locator.pointer, []).append(finding)

    regions: list[Region] = []
    for pointer, group in by_pointer.items():
        group.sort(key=lambda f: (f.locator.char_start, f.locator.char_end))
        current: Region | None = None
        for finding in group:
            start, end = finding.locator.char_start, finding.locator.char_end
            if current is None or start > current.end:
                if current is not None:
                    regions.append(current)
                current = Region(pointer, start, end)
            else:
                current.end = max(current.end, end)
            current.types.add(finding.type)
            current.layers.add(finding.layer)
            current.findings += 1
        if current is not None:
            regions.append(current)
    return regions


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union else 0.0


# --- per-run scoring -------------------------------------------------------


@dataclass
class RunScore:
    run_id: str
    split: str
    # item-level recall, two definitions that answer different questions
    items_total: int = 0
    items_span_flagged: int = 0
    items_value_recovered: int = 0
    items_expected_miss: int = 0
    items_expected_miss_span_flagged: int = 0
    # span-level
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_iou: int = 0
    regions: int = 0
    # negatives
    negative_suppressible: int = 0
    negative_suppressible_flagged: int = 0
    negative_unavoidable: int = 0
    negative_unavoidable_flagged: int = 0
    # breakdowns
    per_tier: dict[str, dict[str, int]] = field(default_factory=dict)
    per_evasion: dict[str, dict[str, int]] = field(default_factory=dict)
    per_type: dict[str, dict[str, int]] = field(default_factory=dict)


def score_run(
    entry: CorpusEntry,
    settings: Settings,
    *,
    layers: Sequence[str] | None = None,
    iou_threshold: float = 0.5,
    fixtures: dict[str, Any] | None = None,
) -> tuple[RunScore, Any]:
    """Scan one corpus entry and score it against its manifest."""
    doc = entry.document
    finding_set = scan_document(doc, settings, layers)
    regions = merge_regions(finding_set.active)
    positives = natural_positives(doc, fixtures)
    manifest = entry.manifest

    score = RunScore(run_id=entry.run_id, split=entry.split, regions=len(regions))

    # --- item recall, measured two ways ------------------------------------
    #
    # "flagged" asks whether the detector marked the place the secret sits.
    # "recovered" asks whether it produced the secret's actual value. They are
    # different questions and evasion separates them: a base64-wrapped key can
    # be flagged while its plaintext is never recovered. Reporting one number
    # for both would hide exactly the thing worth knowing.
    found_hashes = {f.value_sha256 for f in finding_set.active}
    for item in manifest.items:
        score.items_total += 1
        flagged = any(
            r.pointer == item.pointer and _overlaps(r.start, r.end, item.char_start, item.char_end)
            for r in regions
        )
        recovered = item.value_sha256 in found_hashes
        expected_miss = item.evasion in EXPECTED_MISSES
        if expected_miss:
            score.items_expected_miss += 1
            score.items_expected_miss_span_flagged += int(flagged)
        if flagged:
            score.items_span_flagged += 1
        if recovered:
            score.items_value_recovered += 1

        tier = score.per_tier.setdefault(item.tier, {"total": 0, "flagged": 0, "recovered": 0})
        tier["total"] += 1
        tier["flagged"] += int(flagged)
        tier["recovered"] += int(recovered)
        evasion = score.per_evasion.setdefault(
            item.evasion, {"total": 0, "flagged": 0, "recovered": 0}
        )
        evasion["total"] += 1
        evasion["flagged"] += int(flagged)
        evasion["recovered"] += int(recovered)
        by_type = score.per_type.setdefault(item.type, {"total": 0, "flagged": 0, "recovered": 0})
        by_type["total"] += 1
        by_type["flagged"] += int(flagged)
        by_type["recovered"] += int(recovered)

    # --- span-level precision and recall -----------------------------------
    all_positive_spans = positives + [
        Span(i.pointer, i.char_start, i.char_end, i.type, "injected", i.tier, i.evasion)
        for i in manifest.items
    ]
    matched_positive = [False] * len(all_positive_spans)

    for region in regions:
        is_tp = False
        for index, span in enumerate(all_positive_spans):
            if span.pointer != region.pointer:
                continue
            if _overlaps(region.start, region.end, span.start, span.end):
                is_tp = True
                matched_positive[index] = True
                if _iou(region.start, region.end, span.start, span.end) >= iou_threshold:
                    score.tp_iou += 1
                break
        if is_tp:
            score.tp += 1
        else:
            score.fp += 1

    score.fn = sum(1 for matched in matched_positive if not matched)

    # --- negative controls -------------------------------------------------
    for control in manifest.negative_controls:
        flagged = any(
            r.pointer == control["pointer"]
            and _overlaps(r.start, r.end, control["char_start"], control["char_end"])
            for r in regions
        )
        if control["expect_suppressed"]:
            score.negative_suppressible += 1
            score.negative_suppressible_flagged += int(flagged)
        else:
            score.negative_unavoidable += 1
            score.negative_unavoidable_flagged += int(flagged)

    return score, finding_set


# --- metrics ---------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def aggregate(scores: Sequence[RunScore]) -> dict[str, Any]:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    out = prf(tp, fp, fn)
    out["strict_precision"] = out["precision"]
    out["strict_recall"] = out["recall"]
    iou_precision = sum(s.tp_iou for s in scores) / max(1, sum(s.tp_iou + s.fp for s in scores))
    iou_recall = sum(s.tp_iou for s in scores) / max(1, sum(s.tp_iou + s.fn for s in scores))
    out["iou_precision"] = round(iou_precision, 4)
    out["iou_recall"] = round(iou_recall, 4)
    out["iou_f1"] = round(
        2 * iou_precision * iou_recall / (iou_precision + iou_recall)
        if (iou_precision + iou_recall)
        else 0.0,
        4,
    )
    out["localisation_gap"] = round(out["strict_recall"] - iou_recall, 4)

    items = sum(s.items_total for s in scores)
    expected = sum(s.items_expected_miss for s in scores)
    out["items_total"] = items
    out["items_span_flagged"] = sum(s.items_span_flagged for s in scores)
    out["items_value_recovered"] = sum(s.items_value_recovered for s in scores)
    out["item_recall_span"] = round(out["items_span_flagged"] / items, 4) if items else 0.0
    out["item_recall_value"] = (
        round(out["items_value_recovered"] / items, 4) if items else 0.0
    )

    # Expected misses (zero-width splicing, values split across fields) are
    # unrecoverable by construction, so they are excluded from the headline
    # and reported on their own row instead of silently depressing it.
    out["expected_miss_items"] = expected
    out["expected_miss_span_flagged"] = sum(
        s.items_expected_miss_span_flagged for s in scores
    )
    scorable = items - expected
    out["scorable_items"] = scorable
    if scorable:
        out["item_recall_span_excluding_expected_misses"] = round(
            (out["items_span_flagged"] - out["expected_miss_span_flagged"]) / scorable, 4
        )
        out["item_recall_value_excluding_expected_misses"] = round(
            out["items_value_recovered"] / scorable, 4
        )
    else:
        out["item_recall_span_excluding_expected_misses"] = 0.0
        out["item_recall_value_excluding_expected_misses"] = 0.0

    suppressible = sum(s.negative_suppressible for s in scores)
    flagged_suppressible = sum(s.negative_suppressible_flagged for s in scores)
    out["negative_controls_suppressible"] = suppressible
    out["negative_controls_flagged"] = flagged_suppressible
    out["negative_control_fp_rate"] = (
        round(flagged_suppressible / suppressible, 4) if suppressible else 0.0
    )
    unavoidable = sum(s.negative_unavoidable for s in scores)
    flagged_unavoidable = sum(s.negative_unavoidable_flagged for s in scores)
    out["negative_unavoidable_total"] = unavoidable
    out["negative_unavoidable_flagged"] = flagged_unavoidable
    return out


def _merge_breakdown(scores: Sequence[RunScore], attribute: str) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for score in scores:
        for key, counts in getattr(score, attribute).items():
            bucket = merged.setdefault(key, {"total": 0, "flagged": 0, "recovered": 0})
            for field_name in ("total", "flagged", "recovered"):
                bucket[field_name] += counts.get(field_name, 0)
    return merged


def breakdown_table(
    bucket: dict[str, dict[str, int]], labels: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    labels = labels or {}
    rows = []
    for key, counts in sorted(bucket.items()):
        total = counts["total"]
        rows.append(
            {
                "key": key,
                "label": labels.get(key, ""),
                "total": total,
                "flagged": counts["flagged"],
                "recovered": counts["recovered"],
                "recall_span": round(counts["flagged"] / total, 4) if total else 0.0,
                "recall_value": round(counts["recovered"] / total, 4) if total else 0.0,
            }
        )
    return rows


# --- intervals -------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Closed-form binomial interval; behaves at small n and at 0 or 1."""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    )
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def cluster_bootstrap(
    scores: Sequence[RunScore],
    metric: Callable[[Sequence[RunScore]], float],
    *,
    iterations: int = 2000,
    seed: int = 1337,
) -> tuple[float, float, float]:
    """Resample *trajectories* with replacement.

    The resampling unit is the run, not the finding: findings within a run are
    correlated, so an i.i.d. bootstrap over findings would report an interval
    far narrower than the data supports.
    """
    if not scores:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    values: list[float] = []
    n = len(scores)
    for _ in range(iterations):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        try:
            values.append(float(metric(sample)))
        except ZeroDivisionError:
            continue
    if not values:
        return (0.0, 0.0, 0.0)
    values.sort()
    lo = values[int(0.025 * len(values))]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    point = float(metric(scores))
    return (round(point, 4), round(lo, 4), round(hi, 4))


def base_rate_precision(true_positive_rate: float, false_positive_rate: float, prevalence: float) -> float:
    """Precision as a function of how common the target actually is.

    A detector with excellent corpus precision still has poor precision when
    the thing it looks for is rare. Reporting this curve is the honest answer
    to "your precision on an injected corpus is fiction".
    """
    numerator = true_positive_rate * prevalence
    denominator = numerator + false_positive_rate * (1 - prevalence)
    return round(numerator / denominator, 6) if denominator else 0.0


def base_rate_curve(
    true_positive_rate: float, false_positive_rate: float
) -> list[dict[str, float]]:
    return [
        {
            "prevalence": p,
            "precision": base_rate_precision(true_positive_rate, false_positive_rate, p),
        }
        for p in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    ]


# --- top-level driver ------------------------------------------------------


@dataclass
class EvaluationReport:
    corpus: dict[int, list[CorpusEntry]]
    settings: Settings
    original_documents: list[dict[str, Any]] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)

    def scores_for(
        self,
        seed: int,
        split: str,
        layers: Sequence[str] | None,
    ) -> list[RunScore]:
        fixtures = load_fixtures()
        out: list[RunScore] = []
        for entry in self.corpus[seed]:
            if entry.split != split:
                continue
            score, _ = score_run(entry, self.settings, layers=layers, fixtures=fixtures)
            out.append(score)
        return out


def evaluate(
    corpus: dict[int, list[CorpusEntry]],
    settings: Settings | None = None,
    *,
    original_documents: list[dict[str, Any]] | None = None,
    primary_seed: int | None = None,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Run the full evaluation: headline, per-tier, ablation, CI, base rate.

    ``original_documents`` are the trajectories as recorded, without
    injections. The natural false-positive section scans those rather than
    approximating by filtering injected spans out of the corpus.
    """
    settings = settings or load_settings()
    # Insertion order, not sorted order: the first configured seed is the
    # declared primary, and sorting would silently promote a different one.
    seeds = list(corpus)
    primary = primary_seed if primary_seed is not None else seeds[0]
    report: EvaluationReport = EvaluationReport(
        corpus=corpus,
        settings=settings,
        original_documents=original_documents or [],
    )

    test_runs = sorted({e.run_id for e in corpus[primary] if e.split == "test"})
    calibration_runs = sorted({e.run_id for e in corpus[primary] if e.split == "calibration"})
    report.splits = {"calibration": calibration_runs, "test": test_runs}

    # --- headline on the test split, full layer stack ----------------------
    test_scores = report.scores_for(primary, "test", None)
    headline = aggregate(test_scores)

    def item_recall_metric(sample: Sequence[RunScore]) -> float:
        total = sum(s.items_total - s.items_expected_miss for s in sample)
        flagged = sum(
            s.items_span_flagged - s.items_expected_miss_span_flagged for s in sample
        )
        return flagged / total if total else 0.0

    def item_value_metric(sample: Sequence[RunScore]) -> float:
        total = sum(s.items_total - s.items_expected_miss for s in sample)
        recovered = sum(s.items_value_recovered for s in sample)
        return recovered / total if total else 0.0

    def precision_metric(sample: Sequence[RunScore]) -> float:
        tp = sum(s.tp for s in sample)
        fp = sum(s.fp for s in sample)
        return tp / (tp + fp) if (tp + fp) else 0.0

    def recall_metric(sample: Sequence[RunScore]) -> float:
        tp = sum(s.tp for s in sample)
        fn = sum(s.fn for s in sample)
        return tp / (tp + fn) if (tp + fn) else 0.0

    headline["ci"] = {
        "item_recall_span": cluster_bootstrap(
            test_scores, item_recall_metric, iterations=bootstrap_iterations, seed=primary
        ),
        "item_recall_value": cluster_bootstrap(
            test_scores, item_value_metric, iterations=bootstrap_iterations, seed=primary + 3
        ),
        "precision": cluster_bootstrap(
            test_scores, precision_metric, iterations=bootstrap_iterations, seed=primary + 1
        ),
        "recall": cluster_bootstrap(
            test_scores, recall_metric, iterations=bootstrap_iterations, seed=primary + 2
        ),
        "unit": "trajectory (cluster bootstrap)",
        "iterations": bootstrap_iterations,
        "n_trajectories": len(test_scores),
    }

    # --- injection variance across seeds -----------------------------------
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        scores = report.scores_for(seed, "test", None)
        summary = aggregate(scores)
        per_seed.append(
            {
                "seed": seed,
                "item_recall_span": summary["item_recall_span_excluding_expected_misses"],
                "item_recall_value": summary["item_recall_value_excluding_expected_misses"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
            }
        )

    def spread(key: str) -> dict[str, float]:
        values = [row[key] for row in per_seed]
        mean = sum(values) / len(values) if values else 0.0
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(values), 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
        }

    injection_variance = {
        key: spread(key)
        for key in ("item_recall_span", "item_recall_value", "precision", "recall", "f1")
    }

    # --- breakdowns --------------------------------------------------------
    per_tier = breakdown_table(_merge_breakdown(test_scores, "per_tier"), TIER_DESCRIPTION)
    per_evasion = breakdown_table(_merge_breakdown(test_scores, "per_evasion"))
    per_type = breakdown_table(_merge_breakdown(test_scores, "per_type"))

    # --- ablation ----------------------------------------------------------
    ablation: list[dict[str, Any]] = []
    previous_recall = None
    previous_precision = None
    for name, layers in LAYER_CONFIGS.items():
        scores = report.scores_for(primary, "test", layers)
        summary = aggregate(scores)
        row = {
            "config": name,
            "layers": list(layers),
            "item_recall_span": summary["item_recall_span_excluding_expected_misses"],
            "item_recall_value": summary["item_recall_value_excluding_expected_misses"],
            "precision": summary["precision"],
            "recall": summary["recall"],
            "f1": summary["f1"],
            "tp": summary["tp"],
            "fp": summary["fp"],
            "fn": summary["fn"],
            "delta_recall": None
            if previous_recall is None
            else round(summary["recall"] - previous_recall, 4),
            "delta_precision": None
            if previous_precision is None
            else round(summary["precision"] - previous_precision, 4),
            "negative_control_fp_rate": summary["negative_control_fp_rate"],
        }
        previous_recall = summary["recall"]
        previous_precision = summary["precision"]
        ablation.append(row)

    # --- base rate ---------------------------------------------------------
    tp_rate = headline["recall"]
    fp_denominator = headline["tp"] + headline["fp"]
    fp_rate = (headline["fp"] / fp_denominator) if fp_denominator else 0.0
    base_rate = {
        "true_positive_rate": tp_rate,
        "false_positive_rate": round(fp_rate, 6),
        "curve": base_rate_curve(tp_rate, fp_rate),
        "note": (
            "FPR here is measured on the injected corpus, which is far denser in "
            "positives than reality. The curve shows what precision becomes at "
            "realistic prevalence."
        ),
    }

    # --- natural-text false positives --------------------------------------
    natural = natural_false_positive_report(report)

    return {
        "primary_seed": primary,
        "seeds": seeds,
        "splits": report.splits,
        "headline": headline,
        "injection_variance": injection_variance,
        "per_seed": per_seed,
        "per_tier": per_tier,
        "per_evasion": per_evasion,
        "per_type": per_type,
        "ablation": ablation,
        "base_rate": base_rate,
        "natural_false_positives": natural,
        "thresholds_version": settings.thresholds_version,
    }


def natural_false_positive_report(report: EvaluationReport) -> dict[str, Any]:
    """Findings on the *unmodified* trajectories: the base-rate honesty check.

    On an injected corpus nearly every detection is a true positive by
    construction, so precision there is optimistic. This section scans the
    trajectories exactly as they were recorded and asks how much of what the
    detector flags is genuinely sensitive. That number is the one that
    transfers to a real deployment.

    Adjudication is against the known synthetic values planted in the
    workspace rather than by hand-labelling every span. The method is
    therefore conservative in one direction: disclosure that is real but not
    credential-shaped (an internal hostname, a path) counts as a false
    positive, so the reported rate is an upper bound.
    """
    fixtures = load_fixtures()
    known = fixture_hashes(fixtures)
    # A finding counts as real if its value is a known planted value OR
    # contains one. Without the containment test, an entire connection string
    # is scored as a false positive even though the password inside it is
    # genuine -- which would inflate the rate with an adjudication artefact.
    fixture_values = [entry["value"] for entry in fixtures["values"]]

    flagged = 0
    adjudicated_positive = 0
    adjudicated_negative = 0
    by_layer: dict[str, dict[str, int]] = {}
    by_type: dict[str, dict[str, int]] = {}
    bytes_scanned = 0
    spans_scanned = 0
    examples: list[dict[str, Any]] = []

    from textindex import iter_leaves

    for doc in report.original_documents:
        for _pointer, _path, value in iter_leaves(doc):
            if isinstance(value, str):
                bytes_scanned += len(value.encode("utf-8"))
                spans_scanned += 1

        finding_set = scan_document(doc, report.settings, None)
        for finding in finding_set.active:
            flagged += 1
            value = finding_value(
                doc, finding.locator.pointer, finding.locator.char_start, finding.locator.char_end
            )
            digest = sha256_hex(canonicalize_secret(value)) if value else ""
            is_real = (
                digest in known
                or any(needle in value for needle in fixture_values)
                or finding.type == "ABUSE_INJECTION_MARKER"
            )

            layer = by_layer.setdefault(finding.layer, {"flagged": 0, "real": 0})
            layer["flagged"] += 1
            layer["real"] += int(is_real)
            type_bucket = by_type.setdefault(finding.type, {"flagged": 0, "real": 0})
            type_bucket["flagged"] += 1
            type_bucket["real"] += int(is_real)

            if is_real:
                adjudicated_positive += 1
            else:
                adjudicated_negative += 1
                if len(examples) < 25:
                    examples.append(
                        {
                            "layer": finding.layer,
                            "type": finding.type,
                            "pointer": finding.locator.pointer,
                            "preview": finding.preview,
                            "chars": finding.value_length,
                            "rule": finding.evidence.rule_id,
                        }
                    )

    false_positives = adjudicated_negative
    trajectories = len(report.original_documents)
    return {
        "trajectories": trajectories,
        "findings_on_natural_text": flagged,
        "adjudicated_true_positive": adjudicated_positive,
        "adjudicated_false_positive": false_positives,
        "fp_rate_per_finding": round(false_positives / flagged, 4) if flagged else 0.0,
        "fp_rate_wilson_95": wilson_interval(false_positives, flagged),
        "fp_per_kb_scanned": round(false_positives / (bytes_scanned / 1024), 4)
        if bytes_scanned
        else 0.0,
        "fp_per_1000_spans": round(false_positives / (spans_scanned / 1000), 4)
        if spans_scanned
        else 0.0,
        "denominators": {
            "findings": flagged,
            "spans": spans_scanned,
            "kilobytes": round(bytes_scanned / 1024, 1),
            "trajectories": trajectories,
        },
        "by_layer": by_layer,
        "by_type": by_type,
        "example_false_positives": examples,
        "caveat": (
            "Adjudicated against the known planted fixtures only. Real "
            "disclosure that is not credential-shaped is counted as a false "
            "positive, so this rate is an upper bound, not an estimate."
        ),
    }


def manifest_digest(manifest: Manifest) -> str:
    """Hash of the ground truth. Only this is published; the manifest itself
    names where every secret sits and is handled like raw data."""
    return sha256_hex(canonical_json(manifest))
