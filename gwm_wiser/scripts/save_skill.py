import os
import argparse

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.utils.rollout import rollout_with_mplib_expert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot", type=str, default="panda", choices=["panda", "xarm6"]
    )
    args = parser.parse_args()

    if args.robot == "panda":
        robot_uids = "my_panda_wristcam"
    elif args.robot == "xarm6":
        robot_uids = "xarm6_robotiq_wristcam"
    else:
        raise ValueError(f"Robot {args.robot} not supported")

    # collect 1 no noise demo from any env without RGB, but only robot segmentation
    demo_saving_dir = os.path.join(PROJECT_ROOT, "gwm_skills", "config_0_train")
    if args.robot != "panda":
        demo_saving_dir += f"_{args.robot}"

    rollout_with_mplib_expert(
        demo_saving_dir,
        "config_0",
        "train",
        robot_init_qpos=0.0,
        round_to_collect=1,
        only_save_robot_state_image=True,
        robot_uids=robot_uids,
    )
