"""Open a RealSense that did not start, by resetting it and trying again.

tiptop's `RealsenseCamera.__init__` starts the pipeline and then blocks for 30
frames before the object exists. When a frame never arrives --

    RuntimeError: Frame didn't arrive within 5000

-- the exception escapes the constructor, so nothing owns the pipeline and
nothing stops it. The process keeps the device claimed, and because the
pipeline stages run IN-PROCESS the "process" is the whole session: every
remaining instruction fails the same way, and the only cure was to quit.

Observed on the zhiwei rig 2026-08-19 on the wrist D435i (035422072950,
fw 5.12.15.50): one failed open left /dev/video0, /dev/video2 and /dev/video4
held by the session for the rest of its life.

So: drop the orphaned pipeline, power-cycle the device, wait for it to
re-enumerate, and try again. A hardware reset is the documented remedy for a
RealSense that streams no frames, and it costs a few seconds against losing
the session.
"""

import gc
import logging
import time

_log = logging.getLogger(__name__)

RESET_SETTLE_S = 2.0
REENUMERATE_TIMEOUT_S = 20.0


def reset_device(serial: str) -> bool:
    """Power-cycle one camera and wait for it to come back. True if it did."""
    import pyrealsense2 as rs

    serial = str(serial)
    ctx = rs.context()
    dev = next((d for d in ctx.query_devices()
                if d.get_info(rs.camera_info.serial_number) == serial), None)
    if dev is None:
        _log.warning(f"camera {serial} is not enumerated; nothing to reset")
        return False
    _log.info(f"hardware-resetting camera {serial}")
    try:
        dev.hardware_reset()
    except RuntimeError as e:      # already gone, or mid-reset
        _log.debug(f"reset call returned {e}")
    time.sleep(RESET_SETTLE_S)

    deadline = time.monotonic() + REENUMERATE_TIMEOUT_S
    while time.monotonic() < deadline:
        if any(d.get_info(rs.camera_info.serial_number) == serial
               for d in rs.context().query_devices()):
            _log.info(f"camera {serial} is back")
            return True
        time.sleep(0.5)
    _log.error(f"camera {serial} did not re-enumerate within {REENUMERATE_TIMEOUT_S:.0f} s")
    return False


def open_with_retry(opener, serial: str, attempts: int = 3):
    """`opener()`, resetting `serial` between failed attempts.

    `opener` is whatever normally constructs the camera -- passed in rather
    than imported, so this stays usable for the wrist camera, an external one,
    or anything else that wraps the same constructor.
    """
    last = None
    for i in range(1, attempts + 1):
        try:
            return opener()
        except RuntimeError as e:
            last = e
            _log.warning(f"camera {serial} did not start on attempt {i}/{attempts}: {e}")
            if i == attempts:
                break
            # The failed constructor left the pipeline unreferenced except by
            # this frame's traceback; collecting it is what actually stops it.
            gc.collect()
            reset_device(serial)
    raise RuntimeError(
        f"camera {serial} would not start after {attempts} attempts (last: {last}). "
        "Its USB link or firmware is the suspect, not this code -- replug it, or "
        "check `rs-enumerate-devices`."
    ) from last
