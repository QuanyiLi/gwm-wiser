"""validate_setup: post-capture checks for the pointing rig.

  1. IK reachability of the 4 cells and the whole scoring grid.
  2. Renderer/camera alignment: the FK render of the HOME pose composited
     over each captured photo (same URDF, K, c2w as the scoring path).
  3. A hover-over-dog composite.

Run in the repo venv after capture, before starting the server.
"""

import sys

import numpy as np
from PIL import Image

from config import CAPTURE_DIR, CELLS, GRID_STEP, REGION, REPO, URDF
from pointing import PointerKinematics, load_views

sys.path.insert(0, str(REPO))
from real_data_train.renderer.franka_renderer import FrankaRobotRenderer  # noqa: E402


def main() -> None:
    views, q_init = load_views()
    print("q_init:", np.round(q_init, 4))
    kin = PointerKinematics(q_init)
    print(f"d_tip={kin.d_tip:.4f} m  yaw_home={np.degrees(kin.yaw_home):.1f} deg")

    for name, (x, y) in CELLS.items():
        q = kin.ik(x, y)
        print(f"  {name} ({x:.2f},{y:.2f}): IK {'ok' if q is not None else 'FAIL'}")

    xs = np.round(np.arange(REGION[0], REGION[1] + 1e-9, GRID_STEP), 4)
    ys = np.round(np.arange(REGION[2], REGION[3] + 1e-9, GRID_STEP), 4)
    fails = [(float(x), float(y)) for x in xs for y in ys
             if kin.ik(float(x), float(y)) is None]
    print(f"grid {len(xs)}x{len(ys)} = {len(xs) * len(ys)} points, IK failures: {len(fails)}")
    if fails:
        print("  failed points:", fails[:20])

    # max joint step between 2 cm neighbours flags elbow-branch flips
    sols = {}
    for x in xs:
        for y in ys:
            q = kin.ik(float(x), float(y))
            if q is not None:
                sols[(round(float(x), 4), round(float(y), 4))] = q
    worst = 0.0
    for (x, y), q in sols.items():
        for nb in [(round(x + GRID_STEP, 4), y), (x, round(y + GRID_STEP, 4))]:
            if nb in sols:
                worst = max(worst, float(np.abs(q - sols[nb]).max()))
    print(f"max joint delta between neighbouring grid solutions: {worst:.3f} rad")

    renderer = FrankaRobotRenderer(str(URDF), arm="panda")
    q_dog = kin.ik(*CELLS["dog"])
    for cam, (rgb, K, c2w) in views.items():
        h, w = rgb.shape[:2]
        frames, alpha = renderer.render(
            np.stack([q_init, q_dog]), np.array([1.0, 1.0]), K, c2w,
            width=w, height=h, return_alpha=True)
        for tag, i in [("home", 0), ("hover_dog", 1)]:
            a = alpha[i][..., None]
            comp = (frames[i].astype(np.float32) * a
                    + rgb.astype(np.float32) * (1 - a)).astype(np.uint8)
            Image.fromarray(comp).save(CAPTURE_DIR / f"overlay_{tag}_{cam}.png")
    print(f"overlays saved to {CAPTURE_DIR}")


if __name__ == "__main__":
    main()
