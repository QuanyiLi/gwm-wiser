"""Seeding and recipe loading. Run directories go under ``gwm_rl/experiments``
unless ``GWM_RL_EXPERIMENT_ROOT`` says otherwise (a cluster job points it at
scratch)."""

from __future__ import annotations

import datetime
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
EXPERIMENT_ROOT = Path(os.environ.get("GWM_RL_EXPERIMENT_ROOT", HERE / "experiments"))


def set_seed(seed: int) -> int:
    if seed == -1:
        seed = int(np.random.randint(0, 10000))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_overrides(pairs: Iterable[str] | None) -> dict:
    """``["agent.gamma=0.9", "num_envs=4096"]`` -> nested dict, values parsed as YAML."""
    out: dict = {}
    for pair in pairs or []:
        path, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {pair!r}")
        node = out
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def current_time_str() -> str:
    return datetime.datetime.now().strftime("%m%d-%H%M%S")


def load_recipe(path: str | Path, *, exp_name_add_time: bool = True, overrides: dict[str, Any] | None = None) -> dict:
    with open(path) as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    cfg = deep_merge(cfg, overrides or {})
    if exp_name_add_time:
        cfg["experiment_name"] = f"{cfg['experiment_name']}-{current_time_str()}"
    cfg["exp_dir"] = str(EXPERIMENT_ROOT / cfg["experiment_name"])
    return cfg
