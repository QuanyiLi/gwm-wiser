"""Pre-flight end-to-end gate on real released VRS data (RTX 3090 class GPU).

Exercises the plan's local acceptance stages in order, all against the real
test tree and through the actual CLIs: audit -> visualization -> smoke
training (forward/backward/save) -> resume -> canonical strict load ->
one-batch overfit with a clear loss decrease -> throughput sanity.

Uses a reduced GWM configuration because the frozen embedder plus the full
4096-dim training state exceeds 24 GB; the full-size configuration is a
cluster concern.

Run:  pytest real_world_gwm/tests/test_preflight_e2e.py -v
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VRS_TEST_ROOT = Path(os.environ.get("VRS_TEST_ROOT", "/root/data/vrs/test"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not VRS_TEST_ROOT.is_dir(),
    reason="pre-flight needs a GPU and the real VRS test tree",
)

SMALL_MODEL = [
    "--model_dim", "512", "--model_ffn_dim", "1024", "--model_head_dim", "64",
    "--model_n_layer", "2", "--model_n_head", "8", "--model_n_kv_head", "4",
]


def run_cli(argv, **kw):
    proc = subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
        **kw,
    )
    assert proc.returncode == 0, (
        f"command failed: {' '.join(map(str, argv))}\n"
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    )
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("preflight")


@pytest.fixture(scope="module")
def manifest_path(workdir):
    path = workdir / "audit_manifest.json"
    run_cli(
        [
            "-m", "real_world_gwm.audit",
            "--roots", str(VRS_TEST_ROOT),
            "--out", str(path),
            "--candidate_steps", "1", "2",
            "--motion_sample_windows", "1",
        ]
    )
    return path


@pytest.fixture(scope="module")
def smoke_run(workdir, manifest_path):
    out_dir = workdir / "run_smoke"
    log = run_cli(
        [
            "-m", "real_world_gwm.train",
            "--dataset_roots", str(VRS_TEST_ROOT),
            "--manifest", str(manifest_path),
            "--output_dir", str(out_dir),
            "--limit_videos", "4",
            "--total_steps", "8", "--save_every", "4", "--log_every", "2",
            "--num_workers", "2",
            *SMALL_MODEL,
        ]
    )
    return out_dir, log


def test_audit_gate_passes_on_real_data(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    assert manifest["totals"]["clips"] == 105
    assert manifest["totals"]["frames"] == 7203
    assert manifest["totals"]["valid_windows"] > 6000
    assert manifest["token_ceiling_violations"] == []
    assert manifest["batch_shapes"]  # audited grids drive the batch policy
    assert all(c["mask_provenance"] == "mask_gt" for c in manifest["clips"])
    assert manifest["temporal_sampling"]["kind"] == "ordinal"


def test_visualization_renders_training_samples(workdir):
    viz = workdir / "viz"
    log = run_cli(
        [
            "-m", "real_world_gwm.adapters.vrs.visualize",
            "--roots", str(VRS_TEST_ROOT),
            "--out", str(viz),
            "--num_windows", "2",
        ]
    )
    assert len(list(viz.glob("*.png"))) == 2
    assert "excluded: 0" in log


def test_smoke_training_saves_step_checkpoints(smoke_run):
    out_dir, log = smoke_run
    for step in (4, 8):
        step_dir = out_dir / f"step_{step:07d}"
        assert (step_dir / "checkpoint.pt").is_file()
        assert (step_dir / "train_state.pt").is_file()
    assert (out_dir / "resolved_config.json").is_file()
    assert "End of training" in log


def test_throughput_is_sane(smoke_run):
    _, log = smoke_run
    rates = [float(m) for m in re.findall(r"steps/s=([0-9.]+)", log)]
    assert rates, "no throughput logged"
    # frozen vision-tower embedding is the training-loop floor; on the 3090
    # anything below this signals a broken pipeline, not normal variance
    assert max(rates) > 0.3, f"throughput collapsed: {rates}"
    print(f"\nthroughput (reduced GWM, 1620 tokens): {rates} steps/s")


def test_resume_restores_and_completes(workdir, manifest_path, smoke_run):
    out_dir, _ = smoke_run
    resume_dir = workdir / "run_resume"
    log = run_cli(
        [
            "-m", "real_world_gwm.train",
            "--dataset_roots", str(VRS_TEST_ROOT),
            "--manifest", str(manifest_path),
            "--output_dir", str(resume_dir),
            "--resume", str(out_dir / "step_0000004" / "train_state.pt"),
            "--limit_videos", "4",
            "--total_steps", "8", "--save_every", "4", "--log_every", "2",
            "--num_workers", "2",
            *SMALL_MODEL,
        ]
    )
    assert "resumed from" in log and "at step 4" in log
    assert (resume_dir / "step_0000008" / "checkpoint.pt").is_file()


def test_canonical_checkpoint_strict_loads_and_runs(smoke_run, manifest_path):
    from real_world_gwm.gwm_model import load_canonical_like_planner

    out_dir, _ = smoke_run
    model, checkpoint = load_canonical_like_planner(
        out_dir / "step_0000008" / "checkpoint.pt"
    )
    assert checkpoint["config"].dim == 512
    assert checkpoint["step"] == 8
    manifest = json.loads(manifest_path.read_text())
    assert checkpoint["metadata"]["manifest_hash"] == manifest["manifest_hash"]
    assert checkpoint["versions"]["transformers"] == "4.57.6"  # pinned by plan
    model.eval()
    x = torch.randn(1, 1620, 4096, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        assert model(x).shape == (1, 1620, 4096)


WISER_ROOT = REPO_ROOT / "wiser_dataset"


@pytest.mark.skipif(
    not (WISER_ROOT / "merged_test").is_dir(),
    reason="WISER merged_test not downloaded (see repo README)",
)
def test_wiser_dev_open_loop_paths(workdir, manifest_path, smoke_run):
    out_dir, _ = smoke_run
    ckpt = out_dir / "step_0000008" / "checkpoint.pt"
    # standalone evaluator on a canonical checkpoint
    log = run_cli(
        [
            "-m", "real_world_gwm.tests.evaluate_open_loop",
            "--checkpoint", str(ckpt),
            "--wiser_dev_dataset_root", str(WISER_ROOT),
            "--max_batches", "2",
        ]
    )
    result = json.loads(log[log.index("{"):log.index("}") + 1])
    assert result["batches"] == 2 and 0 < result["open_loop_mse"] < 10
    # in-training dev metrics
    log = run_cli(
        [
            "-m", "real_world_gwm.train",
            "--dataset_roots", str(VRS_TEST_ROOT),
            "--manifest", str(manifest_path),
            "--output_dir", str(workdir / "run_dev_eval"),
            "--limit_videos", "1", "--limit_windows", "2",
            "--total_steps", "4", "--save_every", "100", "--log_every", "2",
            "--wiser_dev_dataset_root", str(WISER_ROOT),
            "--eval_every", "4", "--eval_batches", "2",
            "--num_workers", "0",
            *SMALL_MODEL,
        ]
    )
    assert "[wiser-dev open-loop]" in log


@pytest.mark.skipif(
    not (WISER_ROOT / "merged_train").is_dir(),
    reason="WISER merged_train not downloaded (see repo README)",
)
def test_wiser_repro_adapter_smoke(workdir):
    """--dataset_adapter wiser: the gwm_train.py-reproduction debug path."""
    out_dir = workdir / "run_wiser_repro"
    log = run_cli(
        [
            "-m", "real_world_gwm.train",
            "--dataset_adapter", "wiser",
            "--dataset_roots", str(WISER_ROOT),
            "--output_dir", str(out_dir),
            "--dataset_subsample_ratio", "2000",
            "--total_steps", "4", "--save_every", "4", "--log_every", "2",
            "--batch_size", "2", "--num_workers", "2",
            *SMALL_MODEL,
        ]
    )
    assert "WISER-contaminated" in log  # debug path is clearly marked
    assert "tokens=1620" in log  # WISER grid, uniform -> batch_size 2 works
    assert (out_dir / "step_0000004" / "checkpoint.pt").is_file()
    assert "End of training" in log


def test_one_batch_overfit_shows_clear_loss_decrease(workdir):
    # no --manifest: covers the automatic startup audit (single-command slurm)
    out_dir = workdir / "run_overfit"
    log = run_cli(
        [
            "-m", "real_world_gwm.train",
            "--dataset_roots", str(VRS_TEST_ROOT),
            "--output_dir", str(out_dir),
            "--limit_videos", "1", "--limit_windows", "1",
            "--overfit_one_batch", "--flip_prob", "0", "--jitter_prob", "0",
            "--total_steps", "40", "--save_every", "100", "--log_every", "5",
            "--num_workers", "0",
            *SMALL_MODEL,
        ]
    )
    assert (out_dir / "audit_manifest.json").is_file()  # written by auto-audit
    mses = [float(m) for m in re.findall(r"mse=([0-9.]+)", log)]
    assert len(mses) >= 4
    assert mses[-1] < 0.7 * mses[0], f"no clear decrease: {mses}"
    print(f"\noverfit mse: {mses[0]:.4f} -> {mses[-1]:.4f}")
