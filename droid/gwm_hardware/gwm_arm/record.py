"""Record a camera while the robot moves.

The turn already saves what the scorer SAW -- one wrist capture and one
external frame, both taken before anything moved. What it never kept is what
happened next, and that is the only record of whether a plan that scored well
actually did the right thing. Reading a failure off a still photograph and a
joint trajectory is guesswork; a video of the descent is not.

The WRIST camera is the default because it is the view that shows the thing
the plan is about -- the object arriving in the gripper, or the object leaving
it -- and because it is free at exactly this moment: capture has finished with
it and nothing else opens it until the next turn. An external camera shows the
whole arm instead and is a better choice for judging a collision or a reach;
`--record-cam external` switches.

Recording never blocks the robot. A camera that will not open, a codec that
will not write, a frame that does not arrive -- all of them log and step
aside. The plan runs either way.

    with Recorder(run_dir / "exec.mp4", serial) as rec:
        ...move the robot...
    print(rec.summary())
"""

import logging
import threading
import time
from pathlib import Path

_log = logging.getLogger(__name__)

NOMINAL_FPS = 15.0          # what the file is stamped with; the real rate is measured
POLL_TIMEOUT_S = 1.0


def camera_serial(which: str) -> str:
    """`wrist` | `external` | `external_2` -> the serial in tiptop.yml."""
    from tiptop.config import tiptop_cfg

    cams = tiptop_cfg().cameras
    key = {"wrist": "hand", "external": "external", "external_2": "external_2"}[which]
    if not hasattr(cams, key):
        raise KeyError(f"tiptop.yml declares no camera {key!r}")
    return str(getattr(cams, key).serial)


class Recorder:
    """Grab frames in a background thread and write them to an mp4.

    Frames are written as they arrive rather than buffered: 720p RGB at 15 fps
    is 2.7 MB a frame, so a minute of buffering would be 2.4 GB of RAM on a
    machine already short of it.
    """

    def __init__(self, path, serial: str, fps: float = NOMINAL_FPS, enabled: bool = True):
        self.path = Path(path)
        self.serial = str(serial)
        self.fps = float(fps)
        self.enabled = bool(enabled)
        self._cam = None
        self._writer = None
        self._stop = threading.Event()
        self._thread = None
        self.frames = 0
        self.started = None
        self.stopped = None
        self.error = None

    # ------------------------------------------------------------------ loop

    def _run(self):
        import cv2

        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                rgb = self._cam.read_camera().rgb
            except Exception as e:      # noqa: BLE001 - never take the robot down
                self.error = self.error or f"frame read failed: {e}"
                break
            if self._writer is None:
                h, w = rgb.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                if not self._writer.isOpened():
                    self.error = "cv2.VideoWriter would not open (codec?)"
                    break
            self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.frames += 1
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    # --------------------------------------------------------------- context

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            from tiptop.perception.cameras.rs_camera import RealsenseCamera

            from gwm_hardware.common.rs_open import open_with_retry

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._cam = open_with_retry(
                lambda: RealsenseCamera(self.serial, enable_depth=False), self.serial)
            self.started = time.perf_counter()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            _log.info(f"recording {self.serial} -> {self.path}")
        except Exception as e:      # noqa: BLE001 - recording is never worth a turn
            self.error = f"could not start: {e}"
            _log.warning(f"recording OFF ({self.error}); the plan runs regardless")
            self._cam = None
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=POLL_TIMEOUT_S + 2.0)
        self.stopped = time.perf_counter()
        if self._writer is not None:
            self._writer.release()
        if self._cam is not None:
            # Same rule as everywhere else on this rig: a RealSense admits one
            # process at a time, so whatever opened it closes it.
            try:
                self._cam.close()
            except Exception as e:      # noqa: BLE001
                _log.debug(f"camera close: {e}")
        return False        # never swallow an execution failure

    # --------------------------------------------------------------- report

    def summary(self) -> str:
        if not self.enabled:
            return ""
        if self.error and not self.frames:
            return f"  recording      : FAILED ({self.error})"
        dur = (self.stopped or time.perf_counter()) - (self.started or 0.0)
        real = self.frames / dur if dur > 0 else 0.0
        note = "" if abs(real - self.fps) < 1.5 else (
            f"  [file is stamped {self.fps:.0f} fps, so it plays "
            f"{'fast' if real < self.fps else 'slow'}]")
        return (f"  recording      : {self.frames} frames, {dur:.1f} s, "
                f"{real:.1f} fps -> {self.path}{note}")
