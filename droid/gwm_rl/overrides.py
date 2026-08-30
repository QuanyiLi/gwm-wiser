"""Dotted-path overrides into a config object — the experiment seam.

A tuning campaign is a sequence of one-line variations on a config. Editing
the config file for each one loses the record of what was varied; passing
``a.b.c=<python literal>`` on the command line keeps it, and the run's saved
arguments then carry the exact recipe.

Dict members are addressed exactly like attributes, so an Isaac Lab
actuator group reads ``scene.robot.actuators.panda_shoulder.stiffness`` and
a task knob reads ``task_params.static_reward_weight``.
"""

from __future__ import annotations

import ast
from typing import Any


def apply_override(cfg: Any, assignment: str) -> None:
    """Apply one ``dotted.path=<python literal>`` assignment to ``cfg``."""
    key, sep, value = assignment.partition("=")
    if not sep:
        raise ValueError(f"override {assignment!r} is not of the form KEY=VALUE")
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    parsed = ast.literal_eval(value)
    if isinstance(node, dict):
        node[parts[-1]] = parsed
    else:
        if not hasattr(node, parts[-1]):
            raise AttributeError(f"override {key!r}: {type(node).__name__} has no {parts[-1]!r}")
        setattr(node, parts[-1], parsed)


def apply_overrides(cfg: Any, assignments: list[str]) -> None:
    """Apply every assignment in order."""
    for assignment in assignments:
        apply_override(cfg, assignment)
