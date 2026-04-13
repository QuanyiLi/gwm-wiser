#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys


def _disable_tf_gpu() -> None:
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    except Exception:
        # Best-effort; TF may already be initialized or not installed in this env.
        pass


def main() -> int:
    p = argparse.ArgumentParser(
        description="Count transitions (steps) in an RLDS/TFDS dataset split."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="TFDS data_dir (e.g., data/wiser/rlds)",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="TFDS dataset name (builder name)",
    )
    p.add_argument("--split", type=str, default="train", help="Split name (train/val)")
    p.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Optional cap on number of episodes to count (0 = no cap).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON object with num_episodes/num_transitions instead of a single integer.",
    )
    args = p.parse_args()

    if args.max_episodes < 0:
        raise ValueError("--max_episodes must be >= 0")

    _disable_tf_gpu()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    ds = tfds.load(
        args.dataset_name, data_dir=args.data_dir, split=args.split, shuffle_files=False
    )

    num_episodes = 0
    num_transitions = 0

    for ep in ds:
        if args.max_episodes and num_episodes >= args.max_episodes:
            break

        steps = ep["steps"]
        card = tf.data.experimental.cardinality(steps).numpy()
        if card < 0:
            # Fallback if cardinality is unknown.
            card = sum(1 for _ in steps)
        num_transitions += int(card)
        num_episodes += 1

    if args.json:
        print(
            json.dumps(
                {"num_episodes": num_episodes, "num_transitions": num_transitions}
            )
        )
    else:
        print(num_transitions)

    return 0


if __name__ == "__main__":
    sys.exit(main())
