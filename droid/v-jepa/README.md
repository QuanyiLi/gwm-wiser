# droid/v-jepa — V-JEPA 2-AC as a goal-image trajectory selector on DROID-sim scene 6

*(standalone experiment folder; nothing outside it is modified. The results
bundle is NOT in the repo: `make_bundle.py` writes it to `vjepa_ret/`
(gitignored), archived as `/root/code/gwm/v-jepa_ret/vjepa_ret_2026-08-23.tar.gz`
on the remote box — its `README.md` carries the full tables, diagnostics and
caveats, `CONCLUSION.md` the adopted number.)*

Handover experiment B: the lineage's own action-conditioned latent world model
(V-JEPA 2-AC, Assran et al. 2025, post-trained on ~62 h of DROID) is given the
goal image it requires and asked to pick, among the same 16 candidate plans per
scene that GWM / TiPToP choose from, the one that reaches the goal. No language
is involved anywhere: the instruction only names which candidates count as
correct.

**Adopted number for the scene-6 table: V-JEPA 2-AC = 25/70** (pick 15/50,
place 10/20; `CONCLUSION.md` in the archived bundle).

Headline (faithful preprocessing: whole frame squashed to 256×256 as in the
authors' stored DROID clips and robot example; goal-producing candidate
excluded from the pool; one designated goal per task): the predicted arm
selects the right object in 0/10 pick and 2/4 place tasks with the
per-candidate argmin, 3/10 and 2/4 with GWM's two-stage rule — every one a
goal-independent constant choice (candidate×goal interaction ≤ 4 % of the
energy variance, ≤ 12 % in the best short-horizon cell, same-object energy
never more than 0.008 below different-object). The oracle arm — the same
encoder and L1 goal cost on the candidates' actually executed frames — is
10/10 and 4/4 (object-correct) and already 20/24 / 12/12 at a 3 s horizon.
Short horizons (1.5 / 3 / 6 s goals), the 8-frame context, 2× / 4× faster
action steps, the other camera, the gripper-TCP state frame and the 1.35:1
crop do not change the verdict; one-step energy landscapes that are correct
on real DROID frames are uninformative on sim frames. Full tables,
diagnostics and caveats: `vjepa_ret/README.md`.

## Layout

| path | what |
|---|---|
| `vjepa_sel/model.py` | V-JEPA 2-AC wrapper: local checkpoint load (ViT-g encoder + 305 M AC predictor), frame encoding, autoregressive rollout with a sliding context window, L1 energy |
| `vjepa_sel/preprocess.py` | image preprocessing: `full_aa` (faithful: whole frame → antialiased 256×256, what the stored DROID clips were) and the `train` fallback crop (1.35:1) as an ablation |
| `vjepa_sel/plan_stepper.py` | numpy port of the harness's plan stepping (15 Hz, waypoint stride 3, 20-step gripper, 30-step hold) |
| `vjepa_sel/fk.py` | Panda modified-DH FK to `panda_link8` (validated to 0.00 mm against the sim) |
| `vjepa_sel/traj.py` | plan → EEF state / action sequence at the model's cadence |
| `vjepa_sel/tasks.py` | the 14 tasks and their target clusters |
| `sim/replay_candidates.py` | Isaac replay of every candidate; records frames, states, verdicts |
| `sim/validate_fk.py` | FK / timeline check against a replay |
| `score_vjepa.py` | energies of every candidate rollout vs every goal (predicted + oracle arms; `final`, `lift` and 1.5 / 3 / 6 s horizon goal banks; `--tcp-offset`, `--crop-mode`) |
| `diag_onestep.py` | one-step energy-landscape check, real DROID example vs sim frames |
| `analyze_selection.py` | selection accuracy under the protocols; CSVs in the harness schema |
| `plot_figs.py`, `make_tables.py`, `make_headline.py`, `make_bundle.py` | figures; README tables from `selection.json`; the adopted-number CSVs; assemble `vjepa_ret/` |
| `run_scoring.sh`, `sim/run_replays.sh` | the drivers (a finished replay prints no final line — Isaac's shutdown swallows it; completion = 16 `judge.json` files) |
| `vjepa2/`, `checkpoints/`, `.venv/`, `runs/`, `logs/`, `vjepa_ret/` | upstream clone, 11.7 GB checkpoint, env, raw outputs, results bundle — all gitignored (results live outside the repo) |

## Repro

```bash
cd droid/v-jepa
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
    "torch==2.7.1" "torchvision==0.22.1" timm einops scipy numpy pillow h5py opencv-python-headless pyyaml tqdm matplotlib imageio
git clone --depth 1 https://github.com/facebookresearch/vjepa2.git vjepa2
curl -L -o checkpoints/vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt   # 11.7 GB
# 1. replay the candidate pools in Isaac (~35 min pick, ~15 min place; 3090)
bash sim/run_replays.sh
# 2. score + analyse all configs (~30 min), figures, bundle
bash run_scoring.sh all
for f in pick place; do for c in w32_s4 w8_s4 w32_s8 w32_s16 w32_s4_cam1 w32_s4_tcp; do \
  .venv/bin/python plot_figs.py --family $f --energy-dir runs/vjepa_$f/$c --out-dir vjepa_ret/figs; done; done
.venv/bin/python diag_onestep.py --out runs/diag_onestep_full_aa.json
.venv/bin/python make_tables.py > runs/tables.md
.venv/bin/python make_bundle.py
.venv/bin/python make_headline.py
```
