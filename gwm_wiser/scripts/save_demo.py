import argparse
import os
from pathlib import Path

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.utils.merge_dataset import merge_train_test_datasets
from gwm_wiser.utils.rollout import rollout_with_mplib_expert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=24)
    parser.add_argument("--dataset_name", type=str, default="wiser_dataset")
    parser.add_argument("--no_merge", action="store_true", help="Skip merging datasets")
    args = parser.parse_args()

    for i in range(args.start_index, args.end_index):
        for mode in ["train", "test"]:
            cfg_name = f"config_{i}"
            # collect 1 no noise demo
            demo_saving_dir = os.path.join(
                PROJECT_ROOT,
                args.dataset_name,
                "no_noise_demo_1_round",
                f"{cfg_name}_{mode}",
            )
            if not os.path.exists(demo_saving_dir):
                rollout_with_mplib_expert(
                    demo_saving_dir,
                    cfg_name,
                    mode,
                    robot_init_qpos=0.0,
                    round_to_collect=1,
                )
            else:
                print(f"{demo_saving_dir} exists, skip!")

            # collect 5 noised demo
            if mode == "train":
                demo_saving_dir = os.path.join(
                    PROJECT_ROOT,
                    args.dataset_name,
                    "noise_demo_5_round",
                    f"{cfg_name}_{mode}",
                )
                if not os.path.exists(demo_saving_dir):
                    rollout_with_mplib_expert(
                        demo_saving_dir,
                        cfg_name,
                        mode,
                        robot_init_qpos=0.05,
                        round_to_collect=5,
                    )
                else:
                    print(f"{demo_saving_dir} exists, skip!")

    if not args.no_merge:
        root_directory = merged_output_directory = Path(
            os.path.join(PROJECT_ROOT, args.dataset_name)
        )
        print(f"Merging datasets from {root_directory} to {merged_output_directory}...")
        merge_train_test_datasets(root_directory, merged_output_directory)
