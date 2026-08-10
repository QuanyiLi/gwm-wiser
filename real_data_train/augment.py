"""Window augmentation per the plan of record.

Horizontal flip is DISABLED for good (decision D-13): a flipped full-RGB
stream cannot be reproduced by the state renderer at inference time, so flip
breaks train/inference render homology.

Color jitter is probability-gated and applies ONLY to full RGB: robot-only
color is an invariant of the RAT interface (one sampled transformation shared
by all six frames).
"""

import random

import torchvision.transforms.v2 as T

# Existing WISER jitter ranges.
COLOR_JITTER = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def jitter_window(
    sample: dict,
    jitter_prob: float,
    rng: random.Random = random,
) -> dict:
    if rng.random() < jitter_prob:
        fn_idx, b, c, s, h = COLOR_JITTER.get_params(
            COLOR_JITTER.brightness,
            COLOR_JITTER.contrast,
            COLOR_JITTER.saturation,
            COLOR_JITTER.hue,
        )
        rgb = sample["rgb"]
        adjust = {
            0: lambda x: T.functional.adjust_brightness(x, b),
            1: lambda x: T.functional.adjust_contrast(x, c),
            2: lambda x: T.functional.adjust_saturation(x, s),
            3: lambda x: T.functional.adjust_hue(x, h),
        }
        for i in fn_idx:
            rgb = adjust[int(i)](rgb)
        sample["rgb"] = rgb
    return sample
