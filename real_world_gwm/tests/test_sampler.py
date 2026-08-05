"""Step-granular resumable sampling: deterministic epoch permutations."""

from real_world_gwm.sampling import epoch_permutation, sample_position


def test_epoch_permutation_is_deterministic_and_covers_dataset():
    p1 = epoch_permutation(10, seed=42, epoch=0)
    p2 = epoch_permutation(10, seed=42, epoch=0)
    assert p1 == p2
    assert sorted(p1) == list(range(10))


def test_different_epochs_reshuffle():
    assert epoch_permutation(10, seed=42, epoch=0) != epoch_permutation(
        10, seed=42, epoch=1
    )


def test_sample_position_maps_global_step_to_epoch_and_offset():
    # 10 samples, batch size 2 -> 5 optimizer steps per epoch
    assert sample_position(step=0, dataset_len=10, batch_size=2) == (0, 0)
    assert sample_position(step=4, dataset_len=10, batch_size=2) == (0, 8)
    assert sample_position(step=5, dataset_len=10, batch_size=2) == (1, 0)
    assert sample_position(step=12, dataset_len=10, batch_size=2) == (2, 4)
