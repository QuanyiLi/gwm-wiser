"""A run's timing rules and bookkeeping — no simulator, no torch.

- A deterministic sweep is skipped until the exploring rollout clears
  ``min_explore_success`` (nothing to measure yet, and a sweep would flood the
  buffer with an untrained policy's transitions).
- The next evaluation deadline is measured from the *end* of the last one:
  sweeps are charged against the same wall clock as training.
- The best checkpoint is the best sweep, ties do not re-save, and a threshold
  counts as reached on the first sweep that clears it ``patience`` times in a
  row.
- ``evals.json`` carries the whole history plus a summary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


class RunSchedule:
    def __init__(self, *, eval_interval_s: float, wall_budget_s: float, stop_at: float,
                 patience: int = 1, min_explore_success: float = 0.0):
        self.eval_interval_s = float(eval_interval_s)
        self.wall_budget_s = float(wall_budget_s)
        self.stop_at = float(stop_at)
        self.patience = max(1, int(patience))
        self.min_explore_success = float(min_explore_success)
        self._next_eval = float(eval_interval_s)
        self._cleared = 0

    def eval_due(self, elapsed: float) -> bool:
        return elapsed >= self._next_eval

    def note_eval(self, elapsed: float) -> None:
        self._next_eval = elapsed + self.eval_interval_s

    def sweep_worthwhile(self, explore_success: float) -> bool:
        return _is_number(explore_success) and explore_success >= self.min_explore_success

    def budget_spent(self, elapsed: float) -> bool:
        return elapsed >= self.wall_budget_s

    def clears(self, value: float) -> bool:
        if _is_number(value) and value >= self.stop_at:
            self._cleared += 1
            return self._cleared >= self.patience
        self._cleared = 0
        return False


class RunLedger:
    def __init__(self, path: str | Path, *, metric: str, static: dict | None = None):
        self.path = Path(path)
        self.metric = metric
        self.static = dict(static or {})
        self.history: list[dict] = []
        self.best: dict[str, Any] = {"value": float("-inf"), "wall_time_s": None, "env_steps": None}
        self.reached: dict[str, Any] | None = None
        #: First sweep at or above each level of the stop metric, by level.
        self.milestones: dict[str, dict] = {}

    def record(self, entry: dict) -> None:
        self.history.append(entry)

    def note_best(self, summary: dict, **info) -> bool:
        value = summary.get(self.metric, float("nan"))
        if not _is_number(value) or value <= self.best["value"]:
            return False
        self.best.update(value=float(value), summary=summary, **info)
        return True

    def note_milestones(self, value: float, levels=(0.5, 0.9, 0.95, 0.99, 1.0), **info) -> None:
        if not _is_number(value):
            return
        for level in levels:
            key = f"{level:g}"
            if key not in self.milestones and value >= level - 1e-9:
                self.milestones[key] = dict(value=float(value), **info)

    def note_reached(self, **info) -> None:
        self.reached = dict(info)

    def snapshot(self, *, elapsed: float, counters: dict, extra: dict | None = None) -> dict:
        return {
            **self.static,
            "history": self.history,
            "summary": {
                "reached_threshold": self.reached,
                "milestones": self.milestones,
                "best": self.best,
                "wall_time_s": round(elapsed, 1),
                **counters,
                "env_steps_per_s": round(counters.get("env_steps", 0) / max(elapsed, 1e-9)),
                **(extra or {}),
            },
        }

    def write(self, *, elapsed: float, counters: dict, extra: dict | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(self.snapshot(elapsed=elapsed, counters=counters, extra=extra), handle, indent=1, default=str)
