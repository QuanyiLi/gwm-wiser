"""dump_strips: save the exact RAT strips the scorer sees for the 4 cell-centre hovers.

The server writes cand{i:02d}_s{score:+.4f}.png (photo + 5 renders, anchor-
resized) into --out; candidate order is the CELLS order.
"""

import argparse
import sys

from config import CELLS, REPO, RESULTS, SERVER_URL
from pointing import PointerKinematics, load_views

sys.path.insert(0, str(REPO / "droid"))
from gwm_tiptop.score_client import score_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=SERVER_URL)
    ap.add_argument("--out", default="strips_gripper")
    ap.add_argument("--cam", default="external_cam_2")
    ap.add_argument("--instruction", default="point at the image of the dog")
    args = ap.parse_args()

    views, q_init = load_views()
    kin = PointerKinematics(q_init)
    cands = [kin.candidate(*CELLS[t]) for t in CELLS]
    rgb, K, c2w = views[args.cam]
    out_dir = RESULTS / args.out / args.cam
    r = score_candidates(args.server_url, rgb, K, c2w, args.instruction, cands,
                         {"rat_scale": 3.0, "task_image": "current",
                          "dump_dir": str(out_dir)})
    print("order:", list(CELLS))
    print("scores:", [round(s, 4) for s in r["scores"]])
    print("strips in", out_dir)


if __name__ == "__main__":
    main()
