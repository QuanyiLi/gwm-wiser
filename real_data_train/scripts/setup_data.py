"""Provision the selected training corpora under real_data_train/data/.

Two sources (plan of record, ADR-0016):

- molmoact2_droid: allenai/MolmoAct2-DROID-Dataset (LeRobot v3.0). Downloaded
  as repo files (meta/data parquets + concatenated AV1 mp4s); the reader in
  real_data_train.sources.molmoact2_droid discovers whichever episodes are
  complete on disk, so partial downloads are safe.
- molmobot: allenai/MolmoBot-Data (custom tar-of-tar.zst scene packages).
  Shards are pulled through the HF cache and scene packages are extracted to
  the same layout the authors' bulk_download.py produces:
      molmobot/<Config>/part0/<split>/<package_name>/...
  Run-1 corpus (decision D-31): ONLY the plain Pick-and-Place config
  (train shards 0-100 of 1,598 — ~217 GB fetched, ~53 GB kept after
  pruning); Pick-only, Color, and NextTo are excluded. The Pick config
  survives solely inside the frozen --test-split smoke fixture.

--test-split provisions the fixed smoke-test subset used by the Stage-1 E2E
gate on the dev machine (~1.4 GB total):
  - DROID episodes 0..148 with both exterior streams (one file per modality)
  - MolmoBot FrankaPickOmniCamConfig val shard 0, first N scene packages

Full-scale provisioning uses the generic selectors (--configs/--split/
--shards/--data-files); the corpus budget itself is a plan-level knob and is
decided outside this script.

Examples:

    # dev machine, smoke subset
    python -m real_data_train.scripts.setup_data --test-split

    # cluster, the MolmoBot run-1 corpus (PnP-only, D-31)
    python -m real_data_train.scripts.setup_data --source molmobot \\
        --split train --shards 0 100 --prune-extracted --purge-shard-cache

    # cluster, DROID data files 0-4 (episode coverage grows with the files)
    python -m real_data_train.scripts.setup_data --source molmoact2_droid \\
        --data-files 0 5 --cameras exterior_1_left exterior_2_left
"""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

DROID_REPO = "allenai/MolmoAct2-DROID-Dataset"
MOLMOBOT_REPO = "allenai/MolmoBot-Data"

# The DROID authors' post-hoc calibration + annotation release (a MODEL repo,
# not a dataset repo). Camera recovery for molmoact2_droid joins against it
# (plan: camera-recovery section). cam2cam_extrinsics.json (180 MB) is not
# needed and skipped.
KARLP_REPO = "KarlP/droid"
KARLP_FILES = (
    "cam2base_extrinsics.json",
    "cam2base_extrinsic_superset.json",
    "intrinsics.json",
    "camera_serials.json",
    "droid_language_annotations.json",
    "keep_ranges_1_0_1.json",
    "episode_id_to_path.json",
)

# Run-1 MolmoBot corpus (decision D-31, supersedes D-13's all-four-configs):
# ONLY the plain Pick-and-Place config is admitted — the Pick-only, Color,
# and NextTo variants are excluded from the training corpus. The Pick config
# remains referenced solely by the frozen --test-split smoke fixture.
PNP_CONFIG = "FrankaPickAndPlaceOmniCamConfig"
SMOKE_CONFIG = "FrankaPickOmniCamConfig"

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
    p.add_argument("--configs", nargs="+", default=[PNP_CONFIG],
                   help="MolmoBot task configs (default: the PnP-only "
                        "run-1 corpus, decision D-31)")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--shards", type=int, nargs=2, metavar=("FROM", "TO"),
                   default=None,
                   help="half-open shard index range; omitted = ALL shards "
                        "of each selected config (test-split: shard 0 only)")
    p.add_argument("--max-packages", type=int, default=None,
                   help="stop after extracting this many scene packages "
                        "per config (test/smoke use)")
    p.add_argument("--purge-shard-cache", action="store_true",
                   help="delete each cached shard tar right after extraction "
                        "(full-scale runs; halves peak disk)")
    p.add_argument("--prune-extracted", action="store_true",
                   help="after extracting each molmobot shard, delete files "
                        "the training path never reads: wrist/gopro/depth "
                        "mp4s and the .h5.bak duplicates (~50%% disk saved; "
                        "gopro is re-downloadable if its ablation is revived)")
    p.add_argument("--no-assets", dest="assets", action="store_false",
                   help="skip cloning the URDF asset repos into data/assets "
                        "(they are fetched here by default because cluster "
                        "compute nodes often have no internet — rendering "
                        "must find them already on disk)")
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


def setup_karlp_calibration(args):
    """~216 MB of JSON; required for DROID camera recovery (idempotent)."""
    from huggingface_hub import snapshot_download

    local_dir = args.data_root / "karlp_droid"
    print(f"[karlp_droid] downloading {len(KARLP_FILES)} files -> {local_dir}")
    snapshot_download(
        KARLP_REPO,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=list(KARLP_FILES),
    )
    print(f"[karlp_droid] done: {local_dir}")


# ------------------------------------------------------------------ molmobot


def _unused_extracted_file(name: str, exterior_cameras) -> bool:
    if name.endswith(".h5.bak"):
        return True
    if name.endswith(".mp4"):
        # episode_<8d>_<camera>_batch_<...>.mp4
        cam = name.split("_", 2)[2].rsplit("_batch_", 1)[0]
        return cam not in exterior_cameras
    return False


def prune_extracted(out_dir: Path) -> int:
    """Delete extracted files the training path never reads.

    Keeps: h5 state files and the mp4s of the admitted exterior cameras
    (sources.molmobot.EXTERIOR_CAMERAS). Drops: wrist/gopro mp4s, depth
    streams, and trajectories_*.h5.bak duplicates. Returns bytes freed.
    """
    from real_data_train.sources.molmobot import EXTERIOR_CAMERAS

    freed = 0
    for f in out_dir.rglob("*"):
        if not f.is_file():
            continue
        if _unused_extracted_file(f.name, EXTERIOR_CAMERAS):
            freed += f.stat().st_size
            f.unlink()
    return freed


def extract_shard(shard_path: Path, out_dir: Path, max_packages=None,
                  already: set = frozenset(), prune=False):
    """Extract .tar.zst scene packages from one shard tar.

    Mirrors the authors' bulk_download.py layout: each package lands in
    out_dir/<package_stem>/. Returns the number of packages extracted.
    """
    import io

    import zstandard as zstd

    n = 0
    freed = 0
    if prune:
        from real_data_train.sources.molmobot import EXTERIOR_CAMERAS

        def extraction_filter(member, path):
            nonlocal freed
            if (member.isfile()
                    and _unused_extracted_file(
                        Path(member.name).name, EXTERIOR_CAMERAS
                    )):
                freed += member.size
                return None
            return tarfile.data_filter(member, path)
    else:
        extraction_filter = "data"

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
                    pkg_tar.extractall(path=pkg_dir, filter=extraction_filter)
            n += 1
    return n, freed


def purge_cached_download(download_path: Path) -> None:
    """Remove both an HF snapshot link and its content-addressed blob."""
    is_symlink = download_path.is_symlink()
    blob_path = download_path.resolve() if is_symlink else None
    download_path.unlink(missing_ok=True)
    if blob_path is not None:
        blob_path.unlink(missing_ok=True)


def setup_molmobot(args):
    from huggingface_hub import hf_hub_download, list_repo_files

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

        prefix = f"{config}/{args.split}_shards/"
        available = sorted(
            int(Path(f).stem) for f in repo_files
            if f.startswith(prefix) and f.endswith(".tar")
        )
        if args.shards is not None:
            wanted = set(range(*args.shards))
            shard_ids = [s for s in available if s in wanted]
        else:
            shard_ids = available
        print(f"[molmobot] {config} {args.split}: "
              f"{len(shard_ids)}/{len(available)} shards selected")

        done_marker = out_dir / ".extracted_shards.json"
        done = set(json.loads(done_marker.read_text())) if done_marker.exists() else set()
        already_pkgs = {p.name for p in out_dir.iterdir() if p.is_dir()}
        if args.prune_extracted:
            freed = prune_extracted(out_dir)
            print(f"[molmobot] {config}: initial prune freed "
                  f"{freed / 1e9:.2f} GB")

        for shard_id in shard_ids:
            fname = f"{config}/{args.split}_shards/{shard_id:05d}.tar"
            if str(shard_id) in done and args.max_packages is None:
                print(f"[molmobot] {config}: shard {shard_id:05d} already extracted")
                continue
            print(f"[molmobot] {config}: fetching {fname}")
            shard_path = Path(
                hf_hub_download(MOLMOBOT_REPO, repo_type="dataset", filename=fname)
            )
            n, freed = extract_shard(
                shard_path, out_dir,
                max_packages=args.max_packages,
                already=already_pkgs,
                prune=args.prune_extracted,
            )
            print(f"[molmobot] {config}: shard {shard_id:05d} -> "
                  f"{n} scene packages in {out_dir}")
            if args.prune_extracted:
                print(f"[molmobot] {config}: pruned {freed / 1e9:.2f} GB "
                      "(wrist/gopro/depth mp4s, .h5.bak)")
            done.add(str(shard_id))
            done_marker.write_text(json.dumps(sorted(done)))
            if args.purge_shard_cache:
                purge_cached_download(shard_path)


# ---------------------------------------------------------------------- main


def main(argv=None):
    args = parse_args(argv)
    if args.test_split:
        # Fixed smoke subset (Round-2 decisions): DROID episodes 0-148 with
        # both exterior streams; MolmoBot Pick val shard 0, 12 scene packages.
        args.data_files = args.data_files or (0, 1)
        args.video_files = args.video_files or (0, 1)
        args.configs = ([SMOKE_CONFIG]
                        if args.configs == [PNP_CONFIG] else args.configs)
        args.split = "val"
        args.shards = args.shards or (0, 1)
        if args.max_packages is None:
            args.max_packages = 12

    args.data_root.mkdir(parents=True, exist_ok=True)
    if args.source in ("molmoact2_droid", "all"):
        setup_molmoact2_droid(args)
        setup_karlp_calibration(args)
    if args.source in ("molmobot", "all"):
        setup_molmobot(args)
    if args.assets:
        from real_data_train.renderer.assets import (
            ensure_source_repos,
            provenance,
        )

        ensure_source_repos(args.data_root / "assets")
        print(f"[assets] URDF source repos ready: "
              f"{list(provenance(args.data_root / 'assets'))}")
    print("setup complete")


if __name__ == "__main__":
    main()
