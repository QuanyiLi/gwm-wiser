import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets


def is_valid_lerobot_dataset(directory: Path) -> bool:
    """
    Check if a directory contains a valid LeRobot dataset.
    A valid dataset has lerobot_data/ with data/, meta/, videos/ subfolders.
    """
    lerobot_data = directory / "lerobot_data"
    if not lerobot_data.exists():
        return False

    required_subfolders = ["data", "meta", "videos"]
    for subfolder in required_subfolders:
        if not (lerobot_data / subfolder).exists():
            return False

    return True


def find_datasets(root_dir: Path):
    """
    Recursively walk through root_dir and find all valid LeRobot datasets.
    Returns two lists: train_datasets and test_datasets.
    """
    train_dirs = []
    test_dirs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_dir = Path(dirpath)

        if is_valid_lerobot_dataset(current_dir):
            dir_name = current_dir.name
            if "_train" in dir_name:
                train_dirs.append(current_dir)
            elif "_test" in dir_name:
                test_dirs.append(current_dir)
            else:
                print(
                    f"Warning: {dir_name} is a valid dataset but does not contain _train or _test, skipping."
                )

    return train_dirs, test_dirs


def merge_train_test_datasets(root_dir: Path, output_dir: Path):
    """
    Walk through root_dir, find all valid LeRobot datasets,
    split into train and test, and merge each group.
    """
    if not root_dir.exists():
        print(f"Root directory {root_dir} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root_dir} for valid LeRobot datasets...")
    train_dirs, test_dirs = find_datasets(root_dir)

    print(f"Found {len(train_dirs)} train datasets and {len(test_dirs)} test datasets.")

    # Merge train datasets
    if train_dirs:
        print("\nMerging train datasets...")
        train_datasets = []
        for train_dir in sorted(train_dirs):
            print(f"  Loading {train_dir}...")
            lerobot_data_dir = train_dir / "lerobot_data"
            ds = LeRobotDataset(repo_id=train_dir.name, root=lerobot_data_dir)
            train_datasets.append(ds)

        train_output_dir = output_dir / "merged_train"
        merged_train = merge_datasets(
            datasets=train_datasets,
            output_repo_id="merged_train",
            output_dir=train_output_dir,
        )
        print(f"Merged train dataset saved to {train_output_dir}")
        print(f"  Total episodes: {merged_train.meta.total_episodes}")

    # Merge test datasets
    if test_dirs:
        print("\nMerging test datasets...")
        test_datasets = []
        for test_dir in sorted(test_dirs):
            print(f"  Loading {test_dir.name}...")
            lerobot_data_dir = test_dir / "lerobot_data"
            ds = LeRobotDataset(repo_id=test_dir.name, root=lerobot_data_dir)
            test_datasets.append(ds)

        test_output_dir = output_dir / "merged_test"
        merged_test = merge_datasets(
            datasets=test_datasets,
            output_repo_id="merged_test",
            output_dir=test_output_dir,
        )
        print(f"Merged test dataset saved to {test_output_dir}")
        print(f"  Total episodes: {merged_test.meta.total_episodes}")

    print("\nMerging complete.")
