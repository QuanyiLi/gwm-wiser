"""Make FoundationStereo give its 5.7 GB back after each inference.

On this rig's GPU:

    FoundationStereo, weights loaded, before any inference   3658 MiB
    after ONE 1280x720 stereo forward                        9342 MiB
    ------------------------------------------------------------
    retained by PyTorch's caching allocator                   5684 MiB

That 5.7 GB is cache, not model. PyTorch keeps freed blocks so the next
allocation is fast, which is the right default for a training loop and the
wrong one for a service that runs a handful of times per session and then sits
idle next to a 19 GB scorer.

It is the difference between "the modules take turns" and "everything runs at
once". On this 32 GB card:

    FoundationStereo 9342 + gwm-server 19100 + M2T2 1180 + cuRobo 740
        = 30.4 GB of 32.6 -> cuRobo cannot get in (CUDA OOM)
    FoundationStereo 3658 + gwm-server 19100 + M2T2 1180 + cuRobo 740
        = 24.7 GB -> ~8 GB spare, all four co-resident

The cost is that the next inference re-allocates its working set. Against a
~2-3 s stereo forward that runs once per scene capture, that is noise.

The patch is one `torch.cuda.empty_cache()` at the end of `run_inference`,
after the result is already a numpy array on the host. The FoundationStereo
clone is gitignored and rebuilt by its own setup, so this is a versioned,
idempotent installer with a backup and a `--restore`, not a hand edit.

    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.install_fs_memory_release
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.install_fs_memory_release --restore
"""

import argparse
import shutil

from gwm_hardware.common.paths import DROID_ROOT

TARGET = DROID_ROOT / "FoundationStereo/scripts/server.py"
BACKUP = TARGET.with_suffix(".py.orig")
MARKER = "# --- patched by gwm_hardware.common.install_fs_memory_release ---"

BEFORE = """    # Convert to depth
    depth = K[0, 0] * baseline / disp

    return depth"""

AFTER = f"""    # Convert to depth
    depth = K[0, 0] * baseline / disp

    {MARKER}
    # `depth` is already a host numpy array here, so nothing below needs the
    # GPU copies. Dropping them and returning the allocator's cached blocks
    # takes this server from 9342 MiB resident to 3658 MiB, which is what lets
    # it stay up alongside the 19 GB scorer instead of being torn down between
    # stages. Re-allocation costs a fraction of the 2-3 s forward.
    del img0_tensor, img1_tensor
    torch.cuda.empty_cache()

    return depth"""


def install() -> None:
    text = TARGET.read_text()
    if MARKER in text:
        print("already patched")
        return
    if BEFORE not in text:
        raise SystemExit(
            f"{TARGET} does not contain the expected `run_inference` tail. Upstream "
            "changed; re-derive the patch rather than forcing it."
        )
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"saved pristine copy to {BACKUP.name}")
    TARGET.write_text(text.replace(BEFORE, AFTER, 1))
    print(f"patched {TARGET}")
    print("restart the server for it to take effect")


def restore() -> None:
    if not BACKUP.exists():
        raise SystemExit(f"no backup at {BACKUP}")
    shutil.copy2(BACKUP, TARGET)
    print(f"restored {TARGET} from {BACKUP.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} not found -- is FoundationStereo installed?")
    restore() if args.restore else install()


if __name__ == "__main__":
    main()
