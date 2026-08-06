"""Provision the selected training corpora under real_world_gwm/data/.

Two sources (plan of record, ADR-0016):

- molmoact2_droid: allenai/MolmoAct2-DROID-Dataset (LeRobot v3.0). Downloaded
  as repo files (meta/data parquets + concatenated AV1 mp4s); the reader in
  real_world_gwm.sources.molmoact2_droid discovers whichever episodes are
  complete on disk, so partial downloads are safe.
- molmobot: allenai/MolmoBot-Data (custom tar-of-tar.zst scene packages).
  Shards are pulled through the HF cache and scene packages are extracted to
  the same layout the authors' bulk_download.py produces:
      molmobot/<Config>/part0/<split>/<package_name>/...

--test-split provisions the fixed smoke-test subset used by the Stage-1 E2E
gate on the dev machine (~1.4 GB total):
  - DROID episodes 0..148 with both exterior streams (one file per modality)
  - MolmoBot FrankaPickOmniCamConfig val shard 0, first N scene packages

Full-scale provisioning uses the generic selectors (--configs/--split/
--shards/--data-files); the corpus budget itself is a plan-level knob and is
decided outside this script.

Examples:

    # dev machine, smoke subset
    python -m real_world_gwm.scripts.setup_data --test-split

    # cluster, MolmoBot Pick train shards 0-9
    python -m real_world_gwm.scripts.setup_data --source molmobot \\
        --configs FrankaPickOmniCamConfig --split train --shards 0 10

    # cluster, DROID data files 0-4 (episode coverage grows with the files)
    python -m real_world_gwm.scripts.setup_data --source molmoact2_droid \\
        --data-files 0 5 --cameras exterior_1_left exterior_2_left
"""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

DROID_REPO = "allenai/MolmoAct2-DROID-Dataset"
MOLMOBOT_REPO = "allenai/MolmoBot-Data"

FRANKA_CONFIGS = (
    "FrankaPickOmniCamConfig",
    "FrankaPickAndPlaceOmniCamConfig",
    "FrankaPickAndPlaceColorOmniCamConfig",
    "FrankaPickAndPlaceNextToOmniCamConfig",
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--source", choices=["molmoact2_droid", "molmobot", "all"],
                   default="all")
    p.add_argument("--test-split", action="store_true",
                   help="fixed smoke-test subset (see module docstring)")
    # molmoact2_droid selectors
    p.add_argument("--data-files", type=int, nargs=2, metavar=("FROM", "TO"),
                   default=None,
                   help="half-open range of data/chunk-000 parquet file indices")
    p.add_argument("--video-files", type=int, nargs=2, metavar=("FROM", "TO"),
                   default=None,
                   help="half-open range of video file indices per camera; "
                        "defaults to --data-files")
    p.add_argument("--cameras", nargs="+",
                   default=["exterior_1_left", "exterior_2_left"],
                   help="admitted exterior streams (wrist is never used)")
    # molmobot selectors
    p.add_argument("--configs", nargs="+", default=list(FRANKA_CONFIGS))
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--shards", type=int, nargs=2, metavar=("FROM", "TO"),
                   default=None, help="half-open shard index range")
    p.add_argument("--max-packages", type=int, default=None,
                   help="stop after extracting this many scene packages "
                        "per config (test/smoke use)")
    p.add_argument("--purge-shard-cache", action="store_true",
                   help="delete each cached shard tar right after extraction "
                        "(full-scale runs; halves peak disk)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- molmoact2


def setup_molmoact2_droid(args):
    from huggingface_hub import snapshot_download

    local_dir = args.data_root / "molmoact2_droid"
    data_files = args.data_files or (0, 1)
    video_files = args.video_files or data_files

    patterns = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        # smoke subset: episodes 0-148 all live in episodes file-000
        ("meta/episodes/chunk-000/file-000.parquet"
         if args.test_split else "meta/episodes/chunk-000/*.parquet"),
    ]
    patterns += [
        f"data/chunk-000/file-{i:03d}.parquet"
        for i in range(*data_files)
    ]
    for cam in args.cameras:
        patterns += [
            f"videos/observation.images.{cam}/chunk-000/file-{i:03d}.mp4"
            for i in range(*video_files)
        ]

    print(f"[molmoact2_droid] downloading {len(patterns)} patterns "
          f"-> {local_dir}")
    snapshot_download(
        DROID_REPO,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=patterns,
    )
    print(f"[molmoact2_droid] done: {local_dir}")


# ------------------------------------------------------------------ molmobot


def extract_shard(shard_path: Path, out_dir: Path, max_packages=None,
                  already: set = frozenset()):
    """Extract .tar.zst scene packages from one shard tar.

    Mirrors the authors' bulk_download.py layout: each package lands in
    out_dir/<package_stem>/. Returns the number of packages extracted.
    """
    import io

    import zstandard as zstd

    n = 0
    with tarfile.open(shard_path, "r") as shard_tar:
        for member in shard_tar:
            if not member.name.endswith(".tar.zst"):
                continue
            stem = Path(member.name).name[: -len(".tar.zst")]
            # aggregated stats / cache blobs ride along in some shards
            if stem.startswith("_"):
                continue
            if max_packages is not None and n >= max_packages:
                break
            if stem in already:
                n += 1
                continue
            fobj = shard_tar.extractfile(member)
            dctx = zstd.ZstdDecompressor()
            pkg_dir = out_dir
            pkg_dir.mkdir(parents=True, exist_ok=True)
            with dctx.stream_reader(fobj) as reader:
                buffered = io.BufferedReader(reader, buffer_size=1 << 20)
                with tarfile.open(fileobj=buffered, mode="r|") as pkg_tar:
                    pkg_tar.extractall(path=pkg_dir, filter="data")
            n += 1
    return n


def setup_molmobot(args):
    from huggingface_hub import hf_hub_download, list_repo_files

    shard_range = args.shards or (0, 1)
    repo_files = None

    for config in args.configs:
        out_dir = args.data_root / "molmobot" / config / "part0" / args.split
        out_dir.mkdir(parents=True, exist_ok=True)

        # index parquet: provenance + the authors' streaming-access entry table
        try:
            idx = hf_hub_download(
                MOLMOBOT_REPO, repo_type="dataset",
                filename=f"{config}/{args.split}_pkgs-00000-of-00001.parquet",
            )
            shutil.copy2(idx, args.data_root / "molmobot" / config /
                         f"{args.split}_pkgs-00000-of-00001.parquet")
        except Exception as e:  # index naming may vary across configs
            print(f"[molmobot] {config}: index parquet not copied ({e})")

        if repo_files is None:
            repo_files = set(list_repo_files(MOLMOBOT_REPO, repo_type="dataset"))

        done_marker = out_dir / ".extracted_shards.json"
        done = set(json.loads(done_marker.read_text())) if done_marker.exists() else set()
        already_pkgs = {p.name for p in out_dir.iterdir() if p.is_dir()}

        for shard_id in range(*shard_range):
            fname = f"{config}/{args.split}_shards/{shard_id:05d}.tar"
            if fname not in repo_files:
                print(f"[molmobot] {config}: shard {shard_id:05d} absent, skipping")
                continue
            if str(shard_id) in done and args.max_packages is None:
                print(f"[molmobot] {config}: shard {shard_id:05d} already extracted")
                continue
            print(f"[molmobot] {config}: fetching {fname}")
            shard_path = Path(
                hf_hub_download(MOLMOBOT_REPO, repo_type="dataset", filename=fname)
            )
            n = extract_shard(shard_path, out_dir,
                              max_packages=args.max_packages,
                              already=already_pkgs)
            print(f"[molmobot] {config}: shard {shard_id:05d} -> "
                  f"{n} scene packages in {out_dir}")
            done.add(str(shard_id))
            done_marker.write_text(json.dumps(sorted(done)))
            if args.purge_shard_cache:
                shard_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------- main


def main(argv=None):
    args = parse_args(argv)
    if args.test_split:
        # Fixed smoke subset (Round-2 decisions): DROID episodes 0-148 with
        # both exterior streams; MolmoBot Pick val shard 0, 12 scene packages.
        args.data_files = args.data_files or (0, 1)
        args.video_files = args.video_files or (0, 1)
        args.configs = (["FrankaPickOmniCamConfig"]
                        if args.configs == list(FRANKA_CONFIGS) else args.configs)
        args.split = "val"
        args.shards = args.shards or (0, 1)
        if args.max_packages is None:
            args.max_packages = 12

    args.data_root.mkdir(parents=True, exist_ok=True)
    if args.source in ("molmoact2_droid", "all"):
        setup_molmoact2_droid(args)
    if args.source in ("molmobot", "all"):
        setup_molmobot(args)
    print("setup complete")


if __name__ == "__main__":
    main()
