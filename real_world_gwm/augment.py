"""Window augmentation per the phase-one plan.

- Horizontal flip applies the same spatial transform to full RGB, robot-only
  RGB, and masks.
- Color jitter is probability-gated (a deliberate change from the WISER
  trainer, whose ``jitter_prob`` is dead code) and applies ONLY to full RGB:
  robot-only color is an invariant of the RAT interface.
"""

import random

import torchvision.transforms.v2 as T

# Existing WISER jitter ranges.
COLOR_JITTER = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def augment_window(
    sample: dict,
    flip_prob: float,
    jitter_prob: float,
    rng: random.Random = random,
) -> dict:
    if rng.random() < flip_prob:
        for key in ("rgb", "robot_only", "mask"):
            sample[key] = T.functional.hflip(sample[key])

    if rng.random() < jitter_prob:
        fn_idx, b, c, s, h = COLOR_JITTER.get_params(
            COLOR_JITTER.brightness,
            COLOR_JITTER.contrast,
            COLOR_JITTER.saturation,
            COLOR_JITTER.hue,
        )
        # One sampled transformation shared by all six frames, full RGB only.
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
