"""Trajectory persistence.

Two artifacts per run, with different jobs:

``steps.jsonl``
    Append-only journal, one line per step, written *while the run proceeds*.
    If the process dies mid-run the journal still holds every step that had
    completed, which is what makes it crash-safe rather than merely a second
    copy of the final file.

``trajectory.json``
    The assembled artifact: steps in order, parents linked, hash chain built.
    This is the contract Experiment 2 consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from canonical import canonical_json, sha256_hex

from .schema import Step, Trajectory, verify_chain


class RunWriter:
    """Append-only journal plus final artifact for a single run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.run_dir / "steps.jsonl"
        self.artifact_path = self.run_dir / "trajectory.json"

    def journal_step(self, step: Step) -> None:
        """Append one completed step. Flushed immediately, so a crash keeps it."""
        line = canonical_json(step)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def finalize(self, trajectory: Trajectory) -> Path:
        self.artifact_path.write_text(
            json.dumps(trajectory.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.artifact_path

    def journal_digest(self) -> str:
        if not self.journal_path.is_file():
            return ""
        return sha256_hex(self.journal_path.read_bytes())


class TrajectoryStore:
    """Read access to a directory of runs."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)

    def run_dirs(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(p for p in self.runs_dir.iterdir() if (p / "trajectory.json").is_file())

    def load(self, run_id: str) -> Trajectory:
        path = self.runs_dir / run_id / "trajectory.json"
        if not path.is_file():
            raise FileNotFoundError(f"no trajectory for run {run_id!r} under {self.runs_dir}")
        return Trajectory.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_trajectories(self) -> Iterator[Trajectory]:
        for run_dir in self.run_dirs():
            try:
                yield Trajectory.model_validate_json(
                    (run_dir / "trajectory.json").read_text(encoding="utf-8")
                )
            except Exception as exc:  # a corrupt artifact must not stop a scan
                print(f"[store] skipping {run_dir.name}: {type(exc).__name__}: {exc}")

    def verify_all(self) -> list[tuple[str, bool, int | None]]:
        """Re-verify every stored chain. Returns ``(run_id, ok, first_bad)``."""
        results: list[tuple[str, bool, int | None]] = []
        for trajectory in self.iter_trajectories():
            ok, bad = verify_chain(trajectory.steps, trajectory.integrity.prev_run_hash)
            results.append((trajectory.run_id, ok, bad))
        return results

    def count(self) -> int:
        return len(self.run_dirs())
