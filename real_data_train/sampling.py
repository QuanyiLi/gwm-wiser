"""Deterministic step-granular sampling for resumable training.

Each epoch draws a fresh permutation from (seed, epoch); a global optimizer
step maps to (epoch, sample offset), so resume restores the exact sampler
position without storing the permutation itself.
"""

import torch


def epoch_permutation(n: int, seed: int, epoch: int) -> list:
    g = torch.Generator()
    g.manual_seed(seed * 100003 + epoch)
    return torch.randperm(n, generator=g).tolist()


def sample_position(step: int, dataset_len: int, batch_size: int) -> tuple:
    """(epoch, sample offset within the epoch) for a global optimizer step."""
    steps_per_epoch = dataset_len // batch_size
    return step // steps_per_epoch, (step % steps_per_epoch) * batch_size
