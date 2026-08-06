import io
import tarfile

import zstandard as zstd

from real_world_gwm.scripts.setup_data import extract_shard
from real_world_gwm.scripts.setup_data import (
    prune_extracted,
    purge_cached_download,
)


def test_prune_extracted_keeps_admitted_cameras(tmp_path):
    keep = tmp_path / "episode_00000000_randomized_zed2_analogue_1_batch_1.mp4"
    drop_video = tmp_path / "episode_00000000_wrist_camera_batch_1.mp4"
    drop_backup = tmp_path / "trajectories_batch_1.h5.bak"
    state = tmp_path / "trajectories_batch_1.h5"
    for path in (keep, drop_video, drop_backup, state):
        path.write_bytes(b"content")

    assert prune_extracted(tmp_path) == len(b"content") * 2
    assert keep.exists()
    assert state.exists()
    assert not drop_video.exists()
    assert not drop_backup.exists()


def test_purge_cached_download_removes_snapshot_link_and_blob(tmp_path):
    blob = tmp_path / "blobs" / "digest"
    blob.parent.mkdir()
    blob.write_bytes(b"shard")
    snapshot = tmp_path / "snapshots" / "revision" / "00000.tar"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    purge_cached_download(snapshot)

    assert not snapshot.exists()
    assert not blob.exists()


def test_extract_shard_filters_unused_files_before_writing(tmp_path):
    files = {
        "house/episode_00000000_randomized_zed2_analogue_1_batch_1.mp4": b"keep",
        "house/episode_00000000_wrist_camera_batch_1.mp4": b"drop-video",
        "house/trajectories_batch_1.h5": b"state",
        "house/trajectories_batch_1.h5.bak": b"drop-backup",
    }
    package = io.BytesIO()
    with tarfile.open(fileobj=package, mode="w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

    compressed = zstd.ZstdCompressor().compress(package.getvalue())
    shard = tmp_path / "00000.tar"
    with tarfile.open(shard, mode="w") as archive:
        member = tarfile.TarInfo("package.tar.zst")
        member.size = len(compressed)
        archive.addfile(member, io.BytesIO(compressed))

    out = tmp_path / "out"
    count, freed = extract_shard(shard, out, prune=True)

    assert count == 1
    assert freed == len(b"drop-video") + len(b"drop-backup")
    assert (out / "house/trajectories_batch_1.h5").exists()
    assert (out / "house/episode_00000000_randomized_zed2_analogue_1_batch_1.mp4").exists()
    assert not (out / "house/trajectories_batch_1.h5.bak").exists()
    assert not (out / "house/episode_00000000_wrist_camera_batch_1.mp4").exists()
