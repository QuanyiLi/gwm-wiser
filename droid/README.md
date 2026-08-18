# droid/ — the droid-side stack (absorbed 2026-08-10)

Former standalone clones under `/root/code/gwm/`, absorbed into this monorepo so
GWM-side customizations to all of them are versioned in one place (decision
G-16 in `gwm_integrate_doc/plan.md`).

| Dir | Role | Absorbed at (upstream HEAD) | Upstream remote |
|---|---|---|---|
| `tiptop/` | TiPToP planner — pristine upstream worktree since 2026-08-10; GWM additions live in the sibling dirs below | `d8f5afd` (main) | `git@github.com:QuanyiLi/tiptop.git` (own fork) |
| `tiptop/curobo/` | cuRobo (nested clone inside tiptop) | `b5fad1d` | `https://github.com/williamshen-nz/curobo.git` |
| `tiptop/cutamp/` | cuTAMP (nested clone, used as unmodified library per G-7) | `e206ab8` | `https://github.com/tiptop-robot/cuTAMP.git` |
| `M2T2/` | M2T2 grasp server (`:8123`) | `96201ac` (master) | `https://github.com/williamshen-nz/M2T2.git` |
| `droid-sim-evals/` | DROID sim eval harness (batch_eval_v2, SuccessTracker; 16 files carried local eval mods at absorption) | `69a914c` (main) | `https://github.com/tiptop-robot/droid-sim-evals.git` |
| `server/` | gwm-server scoring microservice (`:8901`, GI-4), moved out of `real_data_train/` (ex `real_world_gwm/`); imports `real_data_train.renderer` | — (was never committed upstream) | — |
| `gwm_tiptop/` | GWM×TiPToP integration package (ex `tiptop/gwm_tiptop/`); resolves in the tiptop pixi env via `gwm_tiptop.pth`, see below | — (ours) | — |
| `gwm_integrate_doc/` | GWM×TiPToP plan of record + decision ledger (ex `tiptop/gwm_integrate_doc/`) | — (ours) | — |
| `droid-sim-evals-ours/` | custom eval tasks layered on droid-sim-evals (grasp-and-hold, G-17) | — (ours) | — |
| `gwm_hardware/` | hardware-rig code for `zhiwei`: Panda + Robotiq **2F-140** model generation, RealSense pre-flight, perception-server warm-up, bring-up docs. Separate from `gwm_tiptop/` so the droid-sim results stay reproducible from an unchanged tree | — (ours) | — |
| `FoundationStereo/` | stereo-depth server (`:1234`) — **real-robot only**; droid-sim used GT depth, so this arrived with the hardware bring-up (2026-08-17). RealSense feeds it the IR pair, not the ASIC depth | `b8c58bd` | `https://github.com/williamshen-nz/FoundationStereo.git` (own `.git` kept; gitignored wholesale) |

## Original `.git` dirs → `/root/code/gwm/upstream-git-backups/`

The absorbed repos' own `.git` dirs (full history + remotes) are preserved at
`/root/code/gwm/upstream-git-backups/{tiptop,M2T2,droid-sim-evals,tiptop-curobo,tiptop-cutamp}.git`.
To detach a repo again: `mv /root/code/gwm/upstream-git-backups/<name>.git <dir>/.git`.
To diff against or pull upstream without detaching:
`git --git-dir=/root/code/gwm/upstream-git-backups/<name>.git --work-tree=<dir> diff`.

## Path-compat symlinks (why nothing needed reinstalling)

The old locations are symlinks to here:

```
/root/code/gwm/tiptop          -> /root/code/gwm/gwm-wiser/droid/tiptop
/root/code/gwm/M2T2            -> /root/code/gwm/gwm-wiser/droid/M2T2
/root/code/gwm/droid-sim-evals -> /root/code/gwm/gwm-wiser/droid/droid-sim-evals
```

The 47 GB of environments (`tiptop/.pixi` 16G, `M2T2/.pixi` 13G,
`droid-sim-evals/.venv` 18G) have absolute paths baked into shebangs, activation
scripts, and conda prefixes; they all resolve through these symlinks, so the
environments work unchanged from both the old and new paths. **Do not delete the
symlinks** without recreating the environments.

## Launch commands (canonical, new paths)

```bash
# M2T2 grasp server (:8123)
cd /root/code/gwm/gwm-wiser/droid/M2T2 && pixi run server
# TiPToP server (:8765; GOOGLE_API_KEY in tiptop/.env, chmod 600, gitignored)
cd /root/code/gwm/gwm-wiser/droid/tiptop && source .env && pixi run tiptop-server
# gwm-server scoring service (:8901; --backend dummy for mechanics-only)
cd /root/code/gwm/gwm-wiser && .venv/bin/python -m droid.server.gwm_server \
    --backend gwm --urdf droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf \
    --ckpt /root/exp_ret/0810_gwm/checkpoint.pt
# selection-policy plan server for A/B batches (:8766+, G-19; random | fixed)
/root/code/gwm/gwm-wiser/droid/droid-sim-evals/.venv/bin/python \
    /root/code/gwm/gwm-wiser/droid/gwm_tiptop/policy_server.py \
    --select random --proposals-dir droid/gwm_integrate_doc/proposals/scene1 --port 8766
# 50-trial paper repro batch
cd /root/code/gwm/gwm-wiser/droid/droid-sim-evals && bash run_batch_v2_paper.sh
# grasp-and-hold eval tasks (scene 1, G-17)
cd /root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours && ./run_grasp_tasks.sh
# gwm_tiptop drivers: tiptop env's python, run from the gwm-wiser repo root
cd /root/code/gwm/gwm-wiser && droid/tiptop/.pixi/envs/default/bin/python -m gwm_tiptop.propose_from_h5 --help
```

`gwm_tiptop` is installed into the tiptop pixi env as a site-packages symlink
(G-21). If the env is ever recreated, restore it with:

```bash
ln -sfn /root/code/gwm/gwm-wiser/droid/gwm_tiptop \
    "$(/root/code/gwm/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python -c 'import site; print(site.getsitepackages()[0])')/gwm_tiptop"
```

Do NOT reinstate the earlier `gwm_tiptop.pth` mechanism: `tiptop` and `cutamp`
are editable installs served by setuptools *meta-path* finders, and any
sys.path dir containing a `tiptop/`-shaped subdir (the .pth's `droid/`, but
also cwd `/root/code/gwm` via its compat symlinks, or cwd `droid/tiptop` for
`cutamp/`) shadows them with an `__file__ = None` namespace package — that is
what broke tiptop-server's SAM2 cache-dir lookup (G-21). Console scripts
(`pixi run tiptop-server`) put no cwd on sys.path and are immune; python
`-c`/`-m` runs should use a neutral cwd such as the gwm-wiser root.

Not tracked by git (see root `.gitignore` + per-dir ones): `.pixi/`, `.venv/`,
`tiptop/.env` (secret), `tiptop/tiptop/.cache/` (SAM-2 weights), `M2T2/weights/`
(model weights, itself a git-lfs clone), `droid-sim-evals/runs/` and
`droid-sim-evals-ours/runs/` (eval results), `droid-sim-evals/assets*`
(downloaded scenes).

---

## Rig 2 — `zhiwei` (real robot, bring-up 2026-08-17)

The paths above are the **first machine's** (`/root/code/gwm/...`, droid-sim rig).
This repo now also lives on the hardware workstation at `/home/quanyi/gwm-wiser`,
where nothing was carried over: `.pixi/`, `.venv/`, weights and the compat
symlinks are all gitignored, so every environment was rebuilt from scratch.
See [`gwm_hardware/docs/hardware-bringup.md`](gwm_hardware/docs/hardware-bringup.md)
for the full bring-up procedure.

Rig: Franka + Robotiq 2F-85, Bamboo on a separate RT-kernel NUC, two RealSense
D400s (wrist + external), RTX 5090 32 GB, Ubuntu 22.04.

### Environment rebuild — three things that are not in the upstream docs

**1. `tiptop`'s editable install fails without a pretend version.** `droid/tiptop`
has no `.git` of its own since the absorption, so `setuptools-scm` cannot infer a
version and the editable install dies *while `pixi install` still exits 0* — a
silent failure. Export the version recorded at absorption:

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP="0.0.post1.dev23+gd8f5afdaa"
pixi install
```

**2. cuRobo must be pinned by hand.** `install/install-curobo.sh` does
`git pull --ff-only origin main`, which would drift off the GI-0 pin. Clone and
check out the pin yourself, then build:

```bash
cd droid/tiptop && git clone https://github.com/williamshen-nz/curobo.git curobo
cd curobo && git checkout b5fad1df2a3ac4d3e33e369918b7d62d0e59ebd1
cd .. && pixi run bash -c 'export CUDA_HOME=$CONDA_PREFIX; \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; \
  export TORCH_CUDA_ARCH_LIST="12.0"; \
  cd curobo && pip install -e . --no-build-isolation --no-deps'
pixi run install-cutamp     # pinned by tag from REQUIRED_CUTAMP_VERSION, safe as-is
```

**3. RTX 5090 (`sm_120`) is fine.** The lockfile resolves torch 2.7.1 / CUDA 12.9,
whose `arch_list` includes `sm_120`; cuRobo and cuTAMP build and run. Verified
2026-08-17 with `pixi run cutamp-demo --motion_plan` ("Successful plan found!",
13 motion-plan solves in 0.93 s). Pass `TORCH_CUDA_ARCH_LIST="12.0"` to any
CUDA-extension build (cuRobo, M2T2's `pointnet2_ops`).

### Launch commands on this rig

```bash
cd /home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"

# Camera pre-flight — ALWAYS run first, see gwm_hardware/rs_preflight.py
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.rs_preflight

# M2T2 grasp server (:8123)
cd droid/M2T2 && pixi run server
# FoundationStereo depth server (:1234) — real-robot only
cd droid/FoundationStereo && pixi run server
# TiPToP server / demo (GOOGLE_API_KEY in droid/tiptop/.env, chmod 600)
cd droid/tiptop && source .env && pixi run tiptop-run
# gwm-server scoring service (:8901)
cd /home/quanyi/gwm-wiser && .venv/bin/python -m droid.server.gwm_server \
    --backend gwm --urdf droid/gwm_tiptop/assets/<rig>.urdf \
    --ckpt /home/quanyi/0810_gwm/checkpoint.pt
```

`gwm_tiptop` site-packages symlink for this machine:

```bash
ln -sfn /home/quanyi/gwm-wiser/droid/gwm_tiptop \
    "$(/home/quanyi/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python \
       -c 'import site; print(site.getsitepackages()[0])')/gwm_tiptop"
```

### Rerun spawns a GUI viewer that blocks piped output

`cutamp-demo` / `tiptop-run` spawn a `rerun` viewer that inherits stdout, so a
piped or backgrounded run never sees EOF even after the task finishes. Kill the
viewer (`pkill -f "rerun --port=9876"`) or run with the viewer disabled.
