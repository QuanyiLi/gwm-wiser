"""Per-clip video for the rendered tree.

PNG-per-frame storage put the corpus at ~60 M files / ~1.5 TB — past cluster inode
and byte quotas. One FFV1 MKV per clip is bit-exact through the torchcodec
decode path (verified: encode rgb24 -> ffv1/bgr0 -> decode == input, maxdiff
0), ~55% of the PNG bytes, and ~1/260th of the files.

For REAL sources losslessness is a hard requirement, never an assumption:
writers must call verify_lossless (render_actions does, per clip, before
meta.json lands) and the unit tests round-trip synthetic and real-shaped
frames. YUV-subsampled "lossless" modes (e.g. x264 yuv444) measurably fail
bit-exactness and are banned.

The high-volume SIM tree (molmobot) instead uses near-lossless VP9:
a codec shootout on real MolmoBot renders showed FFV1 is the
LOSSLESS floor (inter-frame lossless VP9/AV1 came out 24-43% larger), while
vp9 crf4/yuv444p is 5.4x smaller at maxdiff 23 / mean 0.067 / 99.9% of
pixels within +-2. Near-lossless clips are verified per write against the
tolerance gates below; inference-time candidate rendering stays live and
uncompressed, so the accepted artifact is train-time and sim-only.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ENCODE_ARGS = ("-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0")

# Near-lossless VP9 for the sim rendered tree. CRF calibrated on real
# MolmoBot renders (sweep over 3 streams / 723 frames @624x352):
#   crf4/yuv444p = 2.4 KB/frame, maxdiff 23, mean 0.067, p99.9 = 2.
NEAR_LOSSLESS_ARGS = ("-c:v", "libvpx-vp9", "-crf", "4", "-b:v", "0",
                      "-pix_fmt", "yuv444p", "-deadline", "good",
                      "-cpu-used", "2", "-row-mt", "1")
NEAR_LOSSLESS_CODEC = "vp9-crf4/yuv444p"
NEAR_LOSSLESS_MAX_ABS = 48    # 2x the calibration maximum
NEAR_LOSSLESS_MEAN_ABS = 0.5  # ~7x the calibration mean


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg") or str(Path(sys.prefix) / "bin" / "ffmpeg")
    if not Path(exe).is_file():
        raise RuntimeError("ffmpeg binary not found (conda-forge 'ffmpeg' package)")
    return exe


def _encode(frames: np.ndarray, path, fps: float, args) -> Path:
    """uint8 (T, H, W, 3) RGB -> video at `path` (written atomically)."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    t, h, w, c = frames.shape
    assert c == 3, f"expected RGB frames, got {frames.shape}"
    path = Path(path)
    tmp = path.with_suffix(".tmp.mkv")
    proc = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", *args, str(tmp)],
        input=frames.tobytes(), capture_output=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode()[-500:]}")
    tmp.rename(path)
    return path


def write_lossless_video(frames: np.ndarray, path, fps: float = 15.0) -> Path:
    """uint8 (T, H, W, 3) RGB -> bit-exact FFV1 MKV (real sources)."""
    return _encode(frames, path, fps, ENCODE_ARGS)


def write_near_lossless_video(frames: np.ndarray, path,
                              fps: float = 15.0) -> Path:
    """uint8 (T, H, W, 3) RGB -> near-lossless VP9 MKV (sim tree)."""
    return _encode(frames, path, fps, NEAR_LOSSLESS_ARGS)


def read_video_frames(path, indices=None) -> np.ndarray:
    """Decode frames (all, or the given indices) -> uint8 (N, H, W, 3)."""
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(path), num_ffmpeg_threads=1)
    if indices is None:
        indices = list(range(dec.metadata.num_frames))
    return dec.get_frames_at(list(indices)).data.permute(0, 2, 3, 1).numpy()


def verify_lossless(path, frames: np.ndarray) -> None:
    """Raise unless the file decodes bit-exactly to `frames`."""
    got = read_video_frames(path)
    if got.shape != frames.shape or not np.array_equal(got, frames):
        raise RuntimeError(
            f"lossless verification FAILED for {path}: decoded shape "
            f"{got.shape} vs {frames.shape}, maxdiff "
            f"{int(np.abs(got.astype(int) - frames.astype(int)).max())}"
        )


def verify_near_lossless(path, frames: np.ndarray,
                         max_abs: int = NEAR_LOSSLESS_MAX_ABS,
                         mean_abs: float = NEAR_LOSSLESS_MEAN_ABS) -> None:
    """Raise unless the file decodes to `frames` within the near-lossless tolerance
    gates — permits calibrated quantization, rejects encode accidents."""
    got = read_video_frames(path)
    if got.shape != frames.shape:
        raise RuntimeError(
            f"near-lossless verification FAILED for {path}: decoded shape "
            f"{got.shape} vs {frames.shape}"
        )
    diff = np.abs(got.astype(np.int16) - frames.astype(np.int16))
    got_max, got_mean = int(diff.max()), float(diff.mean())
    if got_max > max_abs or got_mean > mean_abs:
        raise RuntimeError(
            f"near-lossless verification FAILED for {path}: maxdiff "
            f"{got_max} (limit {max_abs}), meandiff {got_mean:.4f} "
            f"(limit {mean_abs})"
        )
