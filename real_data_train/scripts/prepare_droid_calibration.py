"""Recover per-episode DROID camera calibration by joining MolmoAct2-DROID
against the KarlP/droid post-hoc calibration release (plan: camera-recovery
section; decision D-28).

The LeRobot conversion dropped DROID episode IDs, but kept the triple
language annotations verbatim — and the KarlP release is keyed by episode ID
with those same annotations. The join is therefore exact, not fuzzy:

    (language_instruction, _2, _3)  ->  DROID episode ID
        -> camera_serials.json      ext1/ext2 -> ZED serial
        -> cam2base_extrinsic_superset.json | cam2base_extrinsics.json
                                    serial -> 6D cam2base + quality metric
        -> intrinsics.json          serial -> [fx, cx, fy, cy] @ 1280x720
        -> keep_ranges_1_0_1.json   non-idle frame ranges (openpi-style)

Ambiguous triples (~1.4% of the annotation corpus) are dropped, never
guessed. The output is one JSON consumed by sources.molmoact2_droid:

    <data_root>/molmoact2_droid_calib/calibration.json
        {"episodes": {"ep<idx>": {"episode_id", "keep_ranges",
                                  "cameras": {"<camera>": {
                                      "serial", "extrinsic_cam2base_6d",
                                      "quality_metric", "metric_type",
                                      "extrinsic_source",
                                      "intrinsics": [fx, cx, fy, cy],
                                      "intrinsics_wh": [width, height]}}}}}

Every joined calibration is still verified per stream by the render-time
edge gate (render_actions.py) before any clip enters the rendered tree —
this script only assembles candidates.

CPU-only; run via slurm/submit_prepare_droid_calib.run on the cluster.
"""

import argparse
import json
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CAMERA_SERIAL_KEY = {
    "exterior_1_left": "ext1_cam_serial",
    "exterior_2_left": "ext2_cam_serial",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="drop calibrations below this KarlP quality metric "
                        "(default 0.0: keep all — the render-time edge gate "
                        "is the admission criterion, decision D-28)")
    return p.parse_args(argv)


def load_karlp(karlp_root: Path) -> dict:
    files = {}
    for name in ("droid_language_annotations", "camera_serials",
                 "cam2base_extrinsic_superset", "cam2base_extrinsics",
                 "intrinsics", "keep_ranges_1_0_1", "episode_id_to_path"):
        path = karlp_root / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing — run scripts/setup_data.py "
                "--source molmoact2_droid first"
            )
        files[name] = json.loads(path.read_text())
    return files


def triple_index(annotations: dict) -> tuple:
    """(unique triple -> episode ID, set of ambiguous triples)."""
    index, ambiguous = {}, set()
    for eid, v in annotations.items():
        key = (v.get("language_instruction1", ""),
               v.get("language_instruction2", ""),
               v.get("language_instruction3", ""))
        if key in index:
            ambiguous.add(key)
        index[key] = eid
    for key in ambiguous:
        del index[key]
    return index, ambiguous


def extrinsic_for(eid: str, serial: str, superset: dict, cam2base: dict):
    """(pose6d, quality, metric_type, source_file) or None.

    Superset entries carry per-serial suffixed metrics; cam2base entries hold
    one serial with top-level metrics. Superset wins (both cameras, newer).
    """
    entry = superset.get(eid)
    if entry and serial in entry:
        return (entry[serial], entry.get(f"{serial}_quality_metric"),
                entry.get(f"{serial}_metric_type"), "superset")
    entry = cam2base.get(eid)
    if entry and serial in entry:
        return (entry[serial], entry.get("quality_metric"),
                entry.get("metric_type"), "cam2base")
    return None


def keep_ranges_for(eid: str, id_to_path: dict, keep_ranges: dict):
    rel = id_to_path.get(eid)
    if rel is None:
        return None
    base = f"gs://xembodiment_data/r2d2/r2d2-data-full/{rel}"
    return keep_ranges.get(f"{base}/recordings/MP4--{base}/trajectory.h5")


def episode_language_triples(droid_root: Path):
    """Yield (episode_index, (l1, l2, l3)) from the on-disk data parquets."""
    import pyarrow.parquet as pq

    cols = ["episode_index", "language_instruction",
            "language_instruction_2", "language_instruction_3"]
    for path in sorted(droid_root.glob("data/chunk-*/file-*.parquet")):
        df = pq.read_table(path, columns=cols).to_pandas()
        firsts = df.groupby("episode_index").first()
        for idx, r in firsts.iterrows():
            yield int(idx), (r["language_instruction"],
                             r["language_instruction_2"],
                             r["language_instruction_3"])


def build_calibration(data_root: Path, min_quality: float) -> dict:
    karlp = load_karlp(data_root / "karlp_droid")
    index, ambiguous = triple_index(karlp["droid_language_annotations"])

    stats = {k: 0 for k in (
        "episodes_on_disk", "join_ambiguous", "join_miss", "joined",
        "no_serials", "no_extrinsic", "no_intrinsics", "below_quality",
        "streams_calibrated", "episodes_calibrated", "keep_ranges_present",
    )}
    episodes = {}
    for ep_idx, triple in episode_language_triples(
            data_root / "molmoact2_droid"):
        stats["episodes_on_disk"] += 1
        if triple in ambiguous:
            stats["join_ambiguous"] += 1
            continue
        eid = index.get(triple)
        if eid is None:
            stats["join_miss"] += 1
            continue
        stats["joined"] += 1
        serials = karlp["camera_serials"].get(eid)
        if serials is None:
            stats["no_serials"] += 1
            continue
        ep_intr = karlp["intrinsics"].get(eid, {})
        cameras = {}
        for camera, serial_key in CAMERA_SERIAL_KEY.items():
            serial = serials.get(serial_key)
            if not serial:
                continue
            ext = extrinsic_for(eid, serial, karlp["cam2base_extrinsic_superset"],
                                karlp["cam2base_extrinsics"])
            if ext is None:
                stats["no_extrinsic"] += 1
                continue
            pose6d, quality, metric_type, source = ext
            if quality is not None and quality < min_quality:
                stats["below_quality"] += 1
                continue
            intr = ep_intr.get(serial)
            if intr is None or intr["width"] <= 0 or intr["height"] <= 0 \
                    or intr["cameraMatrix"][0] <= 0 or intr["cameraMatrix"][2] <= 0:
                # ~2% of the release's SVO extractions are zero-filled
                stats["no_intrinsics"] += 1
                continue
            cameras[camera] = {
                "serial": serial,
                "extrinsic_cam2base_6d": pose6d,
                "quality_metric": quality,
                "metric_type": metric_type,
                "extrinsic_source": source,
                "intrinsics": intr["cameraMatrix"],       # [fx, cx, fy, cy]
                "intrinsics_wh": [intr["width"], intr["height"]],
            }
        if not cameras:
            continue
        keep = keep_ranges_for(eid, karlp["episode_id_to_path"],
                               karlp["keep_ranges_1_0_1"])
        if keep is not None:
            stats["keep_ranges_present"] += 1
        episodes[f"ep{ep_idx:06d}"] = {
            "episode_id": eid,
            "keep_ranges": keep,
            "cameras": cameras,
        }
        stats["episodes_calibrated"] += 1
        stats["streams_calibrated"] += len(cameras)

    return {
        "schema_version": 1,
        "min_quality": min_quality,
        "ambiguous_triples_in_annotations": len(ambiguous),
        "stats": stats,
        "episodes": episodes,
    }


def main(argv=None):
    args = parse_args(argv)
    result = build_calibration(args.data_root, args.min_quality)
    out_dir = args.data_root / "molmoact2_droid_calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(out_path)
    for k, v in result["stats"].items():
        print(f"{k:24s} {v}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
