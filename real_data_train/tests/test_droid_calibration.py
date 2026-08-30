"""The KarlP calibration join (prepare_droid_calibration) and its consumers."""

import json

import numpy as np
import pytest

from real_data_train.scripts.prepare_droid_calibration import (
    build_calibration,
    extrinsic_for,
    keep_ranges_for,
    triple_index,
)
from real_data_train.scripts.render_actions import segment_keep_ranges

EID = "LAB+abc123+2023-01-01-10h-00m-00s"
EID2 = "LAB+abc123+2023-01-01-11h-00m-00s"
REL = "LAB/success/2023-01-01/Sun_Jan__1_10:00:00_2023"


def test_triple_index_drops_ambiguous():
    ann = {
        "a": {"language_instruction1": "x", "language_instruction2": "y",
              "language_instruction3": "z"},
        "b": {"language_instruction1": "x", "language_instruction2": "y",
              "language_instruction3": "z"},
        "c": {"language_instruction1": "unique", "language_instruction2": "u2",
              "language_instruction3": "u3"},
    }
    index, ambiguous = triple_index(ann)
    assert ("x", "y", "z") in ambiguous
    assert index == {("unique", "u2", "u3"): "c"}


def test_extrinsic_prefers_superset_with_suffixed_metrics():
    superset = {EID: {"12345": [1, 2, 3, 0.1, 0.2, 0.3],
                      "12345_quality_metric": 0.9,
                      "12345_metric_type": "IoU"}}
    cam2base = {EID: {"12345": [9, 9, 9, 0, 0, 0],
                      "quality_metric": 0.5, "metric_type": "IoU"}}
    pose, q, mt, src = extrinsic_for(EID, "12345", superset, cam2base)
    assert pose == [1, 2, 3, 0.1, 0.2, 0.3] and q == 0.9 and src == "superset"
    pose, q, mt, src = extrinsic_for(EID, "12345", {}, cam2base)
    assert pose == [9, 9, 9, 0, 0, 0] and q == 0.5 and src == "cam2base"
    assert extrinsic_for(EID, "99999", superset, cam2base) is None


def test_keep_ranges_key_construction():
    base = f"gs://xembodiment_data/r2d2/r2d2-data-full/{REL}"
    keep = {f"{base}/recordings/MP4--{base}/trajectory.h5": [[0, 50]]}
    assert keep_ranges_for(EID, {EID: REL}, keep) == [[0, 50]]
    assert keep_ranges_for("unknown", {EID: REL}, keep) is None


def _write_synthetic_root(tmp_path):
    import pandas as pd

    root = tmp_path / "data"
    karlp = root / "karlp_droid"
    karlp.mkdir(parents=True)
    triple = {"language_instruction1": "pick the cube",
              "language_instruction2": "grab the cube",
              "language_instruction3": "lift the cube"}
    files = {
        "droid_language_annotations": {EID: triple, EID2: triple | {
            "language_instruction1": "other"}},
        "camera_serials": {EID: {"wrist_cam_serial": "111",
                                 "ext1_cam_serial": "222",
                                 "ext2_cam_serial": "333"},
                           EID2: {"ext1_cam_serial": "222"}},
        "cam2base_extrinsic_superset": {EID: {
            "222": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
            "222_quality_metric": 0.8, "222_metric_type": "IoU"}},
        "cam2base_extrinsics": {EID: {"333": [1, 1, 1, 0, 0, 0],
                                      "quality_metric": 0.6,
                                      "metric_type": "IoU"}},
        "intrinsics": {EID: {
            "222": {"cameraMatrix": [524.0, 640.0, 524.0, 370.0],
                    "distCoeffs": [0.0] * 12, "width": 1280, "height": 720},
            # zero-filled SVO extraction: must be dropped
            "333": {"cameraMatrix": [0.0, 0.0, 0.0, 0.0],
                    "distCoeffs": [0.0] * 12, "width": 0, "height": 0}}},
        "keep_ranges_1_0_1": {},
        "episode_id_to_path": {EID: REL},
    }
    for name, doc in files.items():
        (karlp / f"{name}.json").write_text(json.dumps(doc))

    droid = root / "molmoact2_droid" / "data" / "chunk-000"
    droid.mkdir(parents=True)
    pd.DataFrame({
        "episode_index": [7, 7, 8],
        "language_instruction": [triple["language_instruction1"]] * 2 + ["no match"],
        "language_instruction_2": [triple["language_instruction2"]] * 2 + ["no"],
        "language_instruction_3": [triple["language_instruction3"]] * 2 + ["no"],
    }).to_parquet(droid / "file-000.parquet")
    return root


def test_build_calibration_end_to_end(tmp_path):
    root = _write_synthetic_root(tmp_path)
    result = build_calibration(root, min_quality=0.0)
    stats = result["stats"]
    assert stats["episodes_on_disk"] == 2
    assert stats["joined"] == 1 and stats["join_miss"] == 1
    assert stats["episodes_calibrated"] == 1
    # ext1 has superset pose + valid intrinsics; ext2's intrinsics are
    # zero-filled and must be dropped
    ep = result["episodes"]["ep000007"]
    assert set(ep["cameras"]) == {"exterior_1_left"}
    cam = ep["cameras"]["exterior_1_left"]
    assert cam["serial"] == "222" and cam["extrinsic_source"] == "superset"
    assert cam["intrinsics_wh"] == [1280, 720]
    assert stats["no_intrinsics"] == 1
    assert ep["keep_ranges"] is None


def test_load_calibrations_scales_intrinsics(tmp_path):
    root = _write_synthetic_root(tmp_path)
    from real_data_train.scripts import prepare_droid_calibration as prep
    from real_data_train.sources.molmoact2_droid import load_calibrations

    prep.main(["--data-root", str(root)])
    flat = load_calibrations(root)
    calib = flat["molmoact2_droid/ep000007/exterior_1_left"]
    k = calib["intrinsics"]
    assert np.allclose(k[0], [524.0 / 4, 0, 640.0 / 4])
    assert np.allclose(k[1], [0, 524.0 / 4, 370.0 / 4])
    assert calib["cam2world_cv"].shape == (4, 4)
    # cam2world translation is the 6D translation verbatim
    assert np.allclose(calib["cam2world_cv"][:3, 3], [0.1, 0.2, 0.3])


def test_segment_keep_ranges():
    # None -> whole episode
    assert segment_keep_ranges(None, 100, min_len=45) == [(0, 100)]
    # clipping to the episode + min-length filter
    assert segment_keep_ranges([[0, 30], [40, 90], [95, 300]], 100,
                               min_len=45) == [(40, 90)]
    assert segment_keep_ranges([[0, 44]], 100, min_len=45) == []
    # too-short episode with no ranges
    assert segment_keep_ranges(None, 30, min_len=45) == []


def test_cv_pose_closed_loop_through_sapien():
    """cv_pose_to_sapien_pose must make SAPIEN's extrinsic matrix equal the
    inverse cam2world (OpenCV convention) — the render-path guarantee."""
    sapien = pytest.importorskip("sapien")
    from real_data_train.renderer.franka_renderer import (
        cv_pose_to_matrix,
        cv_pose_to_sapien_pose,
    )

    scene = sapien.Scene()
    cam = scene.add_camera("c", 64, 48, 1.0, 0.01, 100.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = np.concatenate([rng.uniform(-1, 1, 3),
                            rng.uniform(-np.pi, np.pi, 3)])
        cam.set_entity_pose(cv_pose_to_sapien_pose(v))
        scene.update_render()
        want = np.linalg.inv(cv_pose_to_matrix(v))[:3]
        assert np.abs(cam.get_extrinsic_matrix() - want).max() < 1e-5
