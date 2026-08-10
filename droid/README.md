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
# gwm_tiptop drivers run inside tiptop's pixi env (cwd must be droid/tiptop):
cd /root/code/gwm/gwm-wiser/droid/tiptop && pixi run python -m gwm_tiptop.propose_from_h5 --help
```

`gwm_tiptop` is not installed into the env; it resolves through a one-line
`gwm_tiptop.pth` (containing `/root/code/gwm/gwm-wiser/droid`) in the tiptop pixi
env's site-packages. If the env is ever recreated, restore it with:

```bash
cd /root/code/gwm/gwm-wiser/droid/tiptop && echo /root/code/gwm/gwm-wiser/droid > \
    "$(.pixi/envs/default/bin/python -c 'import site; print(site.getsitepackages()[0])')/gwm_tiptop.pth"
```

Not tracked by git (see root `.gitignore` + per-dir ones): `.pixi/`, `.venv/`,
`tiptop/.env` (secret), `tiptop/tiptop/.cache/` (SAM-2 weights), `M2T2/weights/`
(model weights, itself a git-lfs clone), `droid-sim-evals/runs/` and
`droid-sim-evals-ours/runs/` (eval results), `droid-sim-evals/assets*`
(downloaded scenes).
