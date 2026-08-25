"""sim_common: the handful of facts the Isaac-side scripts need.

The Isaac venv has no SAPIEN, so the home configuration and the executable
trajectories are computed on the repo venv side and handed over as JSON.
"""

import json
from pathlib import Path

from config import RESULTS

HOME_Q_FILE = RESULTS / "home_q.json"


def home_record() -> dict:
    if not HOME_Q_FILE.exists():
        raise FileNotFoundError(
            f"{HOME_Q_FILE} is missing -- run validate_setup.py on the repo venv first")
    return json.loads(HOME_Q_FILE.read_text())


def arm_home_qpos():
    return [float(v) for v in home_record()["q_home"]]


def load_plans(path: Path) -> dict:
    return json.loads(Path(path).read_text())
