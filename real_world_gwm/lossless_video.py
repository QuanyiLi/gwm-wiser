"""Per-clip lossless video for the rendered tree (plan decision D-27).

PNG-per-frame storage put run-1 at ~60 M files / ~1.5 TB — past cluster inode
and byte quotas. One FFV1 MKV per clip is bit-exact through the torchcodec
decode path (verified: encode rgb24 -> ffv1/bgr0 -> decode == input, maxdiff
0), ~55% of the PNG bytes, and ~1/260th of the files.

Losslessness is a hard requirement, never an assumption: writers must call
verify_lossless (render_actions does, per clip, before meta.json lands) and
the unit tests round-trip synthetic and real-shaped frames. YUV-subsampled
"lossless" modes (e.g. x264 yuv444) measurably fail bit-exactness and are
banned.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ENCODE_ARGS = ("-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0")


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg") or str(Path(sys.prefix) / "bin" / "ffmpeg")
    if not Path(exe).is_file():
        raise RuntimeError("ffmpeg binary not found (conda-forge 'ffmpeg' package)")
    return exe


def write_lossless_video(frames: np.ndarray, path, fps: float = 15.0) -> Path:
    """uint8 (T, H, W, 3) RGB -> FFV1 MKV at `path` (written atomically)."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    t, h, w, c = frames.shape
    assert c == 3, f"expected RGB frames, got {frames.shape}"
    path = Path(path)
    tmp = path.with_suffix(".tmp.mkv")
    proc = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", *ENCODE_ARGS, str(tmp)],
        input=frames.tobytes(), capture_output=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode()[-500:]}")
    tmp.rename(path)
    return path


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
