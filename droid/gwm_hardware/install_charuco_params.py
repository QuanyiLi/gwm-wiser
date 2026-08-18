"""Point `calibrate-wrist-cam` at the Charuco board this rig actually owns.

`tiptop/scripts/calibrate_wrist_cam.py` hard-codes the DROID board: 14x9
squares, 20 mm checker, 15 mm marker. Our board is different, and the numbers
are not cosmetic -- the checker size is the scale factor of the whole hand-eye
solve, so an error there lands directly in every commanded grasp.

Board identified from a photo (`IMG_0873.jpeg`) rather than by eye:

    dictionary  DICT_5X5_100   (44 markers, ids 0..43 -- also valid in _50)
    grid        11 x 8 squares (CharucoDetector recovers all 70 interior
                corners with this configuration; 44 = floor(11*8/2))
    ratio       marker / checker = 0.7117, measured from the detected corners

The photo fixes the ratio but not the scale, so the checker size has to be
measured on the physical board and passed in. **Measure across the whole grid,
not one square**: span all 11 squares and divide by 11, which divides the
reading error by 11 as well.

Idempotent, keeps a `.orig`, `--restore` reverts.

    python -m gwm_hardware.install_charuco_params --checker-mm 35.0
    python -m gwm_hardware.install_charuco_params --restore
"""

import argparse
import re
import shutil
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "tiptop/tiptop/scripts/calibrate_wrist_cam.py"
BACKUP = TARGET.with_suffix(".py.orig")
MARKER = "# --- board params patched by gwm_hardware.install_charuco_params ---"

SQUARES_X, SQUARES_Y = 11, 8
MARKER_RATIO = 0.7117          # measured off the board photo
DICT = "DICT_5X5_100"

# Rotational excursion of the calibration sweep.
#
# Upstream uses angle_scale = 0.2 rad and the hand-camera branch divides it by
# 1.5, so the wrist only rotates +-7.6 deg across the whole sweep. Hand-eye
# (AX=XB) constrains the rotational part of X from the rotation BETWEEN poses,
# and 7.6 deg conditions it poorly. The first calibration here came out with a
# 2.6 deg residual: reconstructing the tabletop from three arm poses, the plane
# normal swung 3.0 deg with the wrist while its magnitude stayed put -- the
# signature of a rotational hand-eye error, not a tilted table.
#
# 0.45 rad gives +-17 deg. Still a small motion, and cuRobo plans and
# collision-checks it like any other.
ANGLE_SCALE = 0.45


def install(checker_mm: float) -> None:
    text = TARGET.read_text()
    if MARKER in text:
        print("already patched; restoring first so the values are replaced cleanly")
        restore()
        text = TARGET.read_text()
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"saved pristine copy to {BACKUP.name}")

    checker = checker_mm / 1000.0
    marker = round(checker * MARKER_RATIO, 5)

    block = f'''{MARKER}
# This rig's board, not DROID's 14x9 / 20 mm one. Grid and dictionary read off
# a photo with the aruco detector; checker size measured on the board itself.
CHARUCOBOARD_ROWCOUNT = SQUARES_Y = {SQUARES_Y}
CHARUCOBOARD_COLCOUNT = SQUARES_X = {SQUARES_X}
CHARUCOBOARD_CHECKER_SIZE = {checker}
CHARUCOBOARD_MARKER_SIZE = {marker}
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.{DICT})'''

    traj = re.search(r"def calibration_traj\(t, pos_scale=[\d.]+, angle_scale=([\d.]+)", text)
    if traj and float(traj.group(1)) != ANGLE_SCALE:
        text = text.replace(f"angle_scale={traj.group(1)}", f"angle_scale={ANGLE_SCALE}", 1)
        print(f"  sweep angle_scale {traj.group(1)} -> {ANGLE_SCALE} "
              f"(wrist +-{ANGLE_SCALE/1.5*57.3:.0f} deg, was "
              f"+-{float(traj.group(1))/1.5*57.3:.0f} deg)")

    pattern = re.compile(
        r"# Charuco Board Params #\n"
        r"CHARUCOBOARD_ROWCOUNT = SQUARES_Y = \d+\n"
        r"CHARUCOBOARD_COLCOUNT = SQUARES_X = \d+\n"
        r"CHARUCOBOARD_CHECKER_SIZE = [\d.]+\n"
        r"(?:# CHARUCOBOARD_MARKER_SIZE = [\d.]+\n)?"
        r"CHARUCOBOARD_MARKER_SIZE = [\d.]+\n"
        r"ARUCO_DICT = aruco\.getPredefinedDictionary\(aruco\.\w+\)")
    if not pattern.search(text):
        raise SystemExit("calibrate_wrist_cam.py's board block does not match what "
                         "this patch expects -- upstream changed. Failing rather "
                         "than guessing.")
    TARGET.write_text(pattern.sub(block, text, count=1))
    print(f"patched {TARGET.name}: {SQUARES_X}x{SQUARES_Y}, checker "
          f"{checker_mm:.2f} mm, marker {marker*1000:.2f} mm, {DICT}")
    print(f"  grid outline works out to {SQUARES_X*checker_mm/10:.1f} x "
          f"{SQUARES_Y*checker_mm/10:.1f} cm -- sanity-check that against the board")


def restore() -> None:
    if not BACKUP.exists():
        raise SystemExit(f"no pristine copy at {BACKUP}")
    shutil.copy2(BACKUP, TARGET)
    print(f"restored {TARGET.name}")


def verify() -> None:
    import sys
    sys.path.insert(0, str(TARGET.parents[2]))
    src = TARGET.read_text()
    for k in ("SQUARES_X", "SQUARES_Y", "CHARUCOBOARD_CHECKER_SIZE",
              "CHARUCOBOARD_MARKER_SIZE"):
        m = re.search(rf"{k} = ([\d.]+)", src)
        print(f"  {k:28s} {m.group(1) if m else '??'}")
    m = re.search(r"getPredefinedDictionary\(aruco\.(\w+)\)", src)
    print(f"  {'ARUCO_DICT':28s} {m.group(1)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--restore", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--checker-mm", type=float,
                    help="measured checker square size in mm (span the whole "
                         "grid and divide, do not measure one square)")
    a = ap.parse_args()
    if a.restore:
        restore()
    elif a.verify:
        verify()
    else:
        if a.checker_mm is None:
            raise SystemExit("--checker-mm is required; measure it first")
        install(a.checker_mm)
        verify()


if __name__ == "__main__":
    main()
