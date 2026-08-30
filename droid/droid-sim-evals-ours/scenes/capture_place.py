"""capture_place: capture_scene6 with the held block welded to the gripper.

capture_scene6.main() settles 100 steps before capturing; without the weld the
free block would drop 30 cm onto the table during that settle and every output
(layout PNGs, wrist_obs.h5, external_obs.h5, objects.json) would show it on the
table instead of in the gripper. Importing weld_held_block wraps settle_sim so
the joint is authored before the first settle step -- same seam batch_eval_v2
gets via place_eval.

    cd /root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u scenes/capture_place.py --scene 6 --variant 1
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # weld_held_block (also adds droid-sim-evals for src.*)
sys.path.insert(0, str(HERE))  # capture_scene6

import weld_held_block  # noqa: F401  -- wraps settle_sim before capture_scene6 imports it

import capture_scene6

if __name__ == "__main__":
    capture_scene6.main()
