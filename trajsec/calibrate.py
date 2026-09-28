"""Threshold calibration.

The one tuned numeric parameter in the detector is L3's normalised-entropy
threshold. It is selected on the calibration split, frozen to
``rules/thresholds.json``, and only then applied to the test split. The full
sweep is published alongside the winner, because a single selected value with
no curve behind it is indistinguishable from a number picked by looking at the
answer.

The decision rule is fixed in PREREGISTRATION.md before this runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from trajsec.config import Settings, load_settings, save_thresholds
from trajsec.corpus import CorpusEntry
from trajsec.evaluate import aggregate, load_fixtures, score_run

SWEEP_START = 0.45
SWEEP_STOP = 0.90
SWEEP_STEP = 0.01


@dataclass
class SweepPoint:
    threshold: float
    f1: float
    precision: float
    recall: float
    item_recall_span: float
    fp: int
    fn: int
    negative_fp_rate: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def sweep(
    corpus: dict[int, list[CorpusEntry]],
    settings: Settings,
    *,
    split: str = "calibration",
) -> list[SweepPoint]:
    """Score every candidate threshold on one split."""
    fixtures = load_fixtures()
    points: list[SweepPoint] = []
    threshold = SWEEP_START
    while threshold <= SWEEP_STOP + 1e-9:
        tuned = Settings(
            runs_dir=settings.runs_dir,
            artifacts_dir=settings.artifacts_dir,
            protected_dir=settings.protected_dir,
            rules_dir=settings.rules_dir,
            seeds=settings.seeds,
            calibration_fraction=settings.calibration_fraction,
            policy_version=settings.policy_version,
            thresholds={
                **settings.thresholds,
                "entropy_normalised_threshold": round(threshold, 4),
                "version": f"calib-sweep-{threshold:.2f}",
            },
        )
        scores = []
        for seed in sorted(corpus):
            for entry in corpus[seed]:
                if entry.split != split:
                    continue
                score, _ = score_run(entry, tuned, layers=None, fixtures=fixtures)
                scores.append(score)
        summary = aggregate(scores)
        points.append(
            SweepPoint(
                threshold=round(threshold, 4),
                f1=summary["f1"],
                precision=summary["precision"],
                recall=summary["recall"],
                item_recall_span=summary["item_recall_span_excluding_expected_misses"],
                fp=summary["fp"],
                fn=summary["fn"],
                negative_fp_rate=summary["negative_control_fp_rate"],
            )
        )
        threshold = round(threshold + SWEEP_STEP, 4)
    return points


def select(points: list[SweepPoint]) -> SweepPoint:
    """Highest F1; ties break toward the higher (more precise) threshold."""
    best = points[0]
    for point in points:
        if point.f1 > best.f1 + 1e-9:
            best = point
        elif abs(point.f1 - best.f1) <= 1e-9 and point.threshold > best.threshold:
            best = point
    return best


def calibrate(
    corpus: dict[int, list[CorpusEntry]],
    settings: Settings | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Sweep, select, freeze. Returns the sweep and the chosen point."""
    settings = settings or load_settings()
    points = sweep(corpus, settings)
    best = select(points)

    thresholds = dict(settings.thresholds)
    thresholds.update(
        {
            "version": f"calibrated-{best.threshold:.2f}",
            "frozen": True,
            "frozen_at": "2026-09-28",
            "entropy_normalised_threshold": best.threshold,
            "calibration": {
                "split": "calibration",
                "decision_rule": "maximise F1 over merged prediction regions; ties break toward the higher threshold",
                "sweep_range": [SWEEP_START, SWEEP_STOP, SWEEP_STEP],
                "selected_f1": best.f1,
                "selected_precision": best.precision,
                "selected_recall": best.recall,
            },
        }
    )
    path = save_thresholds(thresholds) if write else None

    return {
        "selected_threshold": best.threshold,
        "selected": best.as_dict(),
        "sweep": [p.as_dict() for p in points],
        "thresholds_path": str(path) if path else None,
        "thresholds_version": thresholds["version"],
        "sweep_interpretation": _interpret(points, best),
    }


def _interpret(points: list[SweepPoint], best: SweepPoint) -> dict[str, Any]:
    """A short, honest read of the curve so the report does not have to guess."""
    lowest = min(points, key=lambda p: p.threshold)
    highest = max(points, key=lambda p: p.threshold)
    return {
        "f1_at_lowest_threshold": lowest.f1,
        "f1_at_highest_threshold": highest.f1,
        "f1_spread": round(max(p.f1 for p in points) - min(p.f1 for p in points), 4),
        "monotonic_recall": all(
            points[i].item_recall_span <= points[i + 1].item_recall_span + 1e-9
            for i in range(len(points) - 1)
        ),
        "note": (
            "The curve shows the recall the entropy layer buys and the precision "
            "it costs. If the selected point sits at the low end of the swing, "
            "the layer is being asked to do real work; if F1 is flat across the "
            "range, the threshold is not load-bearing and should be reported as such."
        ),
    }
