# References

## Primary external sources

- [Grounded World Model for Semantically Generalizable Planning](https://arxiv.org/html/2604.11751) — GWM, RAT, WISER, and the reported planning results.
- [RobotSeg repository](https://github.com/showlab/RobotSeg) — VRS download, dataset summary, masks, code license, and training configuration.
- [RobotSeg paper, section 3](https://arxiv.org/html/2511.22950v2#S3) — VRS construction, embodiments, clip/frame counts, and annotation protocol.
- [RobotSeg VRS loader](https://github.com/showlab/RobotSeg/blob/main/train/dataset/vos_raw_dataset.py) — ordered JPG frames and mask categories `000=arm`, `001=gripper`, `002=whole robot`.
- [RobotSeg training configuration](https://github.com/showlab/RobotSeg/blob/main/robotseg/configs/robotseg-train.yaml) — released VRS paths and pseudo-mask training input.
- [Qwen3-VL-Embedding-8B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — multimodal embedding model and representation dimensions.
- [Qwen3-VL-Embedding-8B configuration](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/config.json) — visual patch, merge, DeepStack, and hidden-size configuration.

## Dataset-survey leads

First-tier leads (one adapter each, validating three signal hypotheses; see the phase-one plan):

- [RobotSeg repository / VRS](https://github.com/showlab/RobotSeg) — ready-now real-video source: 2,812 videos / 138,707 frames (train 2,707 / 131,504; test 105 / 7,203), ten upstream datasets, 80% Franka from DROID; masks `000=arm, 001=gripper, 002=whole robot`; train-side dense masks are DINOv3 pseudo labels (`mask_gt_dinov3`), test side densely human-annotated; no fps or timestamps published; OneDrive/Baidu release only, no HuggingFace mirror.
- [DROID](https://droid-dataset.github.io/) — 76k Franka episodes under CC-BY 4.0 with synchronized joint states and calibration; phase-one entry via RobotSeg-predicted whole-robot masks (ADR-0015), state bundle counted as phase-two readiness.
- [RoboCasa](https://robocasa.ai/) — large simulation corpus (2,200+ hours of demonstrations, mobile manipulators); engine reported as robosuite/MuJoCo (verify in survey); ground-truth segmentation expected from the simulator.
- [MolmoBot](https://github.com/allenai/MolmoBot) — AI2 large-scale simulation for zero-shot manipulation; MolmoBot-Data on Hugging Face carries `obs/agent/qpos` for Franka/RBY1/DROID-style setups; observation naming suggests ManiSkill, so the engine-proximity tag applies.

Second-tier leads (surveyed on paper, no adapter yet):

- [RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) — 107k real trajectories across 479 tasks, including 52.9k Franka with RGB and joint states.
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) — WidowX real corpus; non-Franka.
- [NVIDIA PhysicalAI Robotics Manipulation Augmented](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented) — simulation lead; qpos, calibration, label, and license suitability require revalidation.
- [LIBERO](https://libero-project.github.io/) — Franka simulation benchmark; modest scale.
- [Exylos Bimanual Table Spill Cleanup](https://huggingface.co/datasets/ExylosAi/table_spill_cleanup_bimanual) — unverified scale and suitability.
- [AgiBot World](https://agibot-world.com/) — large humanoid dual-arm real corpus; non-Franka.
- [Dexora Real-World Dataset](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset) — low-confidence lead; embodiment, qpos, main-camera, mask, calibration, and license claims require inspection.

Reference-only (not training corpora): [RoboEngine/RoboSeg](https://github.com/michaelyuancb/roboengine) and [RoVi-Aug](https://github.com/BerkeleyAutomation/rovi-aug) as segmentation/data-generation references; [datasets.bot](https://datasets.bot/) as a catalog index whose entries require direct upstream inspection.

## Relevant local sources

- [Repository overview](../../README.md) — current GWM/WISER workflows and dataset roles.
- [GWM model](../../gwm_wiser/models/gwm.py) — fixed 1,620-token wrapper and canonical learned modules.
- [Qwen embedding wrapper](../../gwm_wiser/models/qwen3_vl_embedding.py) — local Qwen preprocessing, visual latent extraction, and pooling.
- [Qwen video preprocessing](../../gwm_wiser/models/qwen_video_utils.py) — aspect-aware smart resizing and video-grid construction.
- [GWM data pipeline](../../gwm_wiser/utils/gwm_data.py) — six-frame condition/target construction and latent concatenation.
- [GWM training](../../gwm_wiser/scripts/gwm_train.py) — frozen Qwen, MSE optimization, cosine diagnostics, optimizer, scheduler, and checkpoint format.
- [GWM closed-loop evaluation](../../gwm_wiser/scripts/gwm_eval.py) — unchanged phase-one development evaluation entry point.
- [Retrieval planner](../../gwm_wiser/planner/retrieval.py) — logged candidate retrieval, semantic scoring, and action-chunk execution.
- [Robot-only renderer](../../gwm_wiser/utils/robot_renderer.py) — WISER-aligned rendering from explicit robot configurations or supported absolute joint-position commands.
- [Retired VRS prototype review](prior-dev-vrs-prototype.md) — evidence retained from the former `origin/dev` branch, including what its loader fixture did and did not establish.

## Established facts affecting the design

- VRS contains ordered video frames but publishes no per-video timestamps, FPS, extraction stride, action, proprioception, camera calibration, or task labels.
- VRS training masks after the first manually annotated frame are propagated pseudo-labels; the released test masks are densely annotated.
- The committed VRS training manifest is heavily Franka-weighted. If VRS is selected, its current adapter plan uses all released embodiments without balancing; this is a data-use choice, not a cross-embodiment claim.
- The existing GWM trains on Qwen internal visual tokens, not the final pooled retrieval vector.
- The current GWM wrapper fixes the concatenated latent at 1,620 tokens, while its learned projections and transformer layers are sequence-length agnostic.
- The existing preprocessing maps WISER `448x224` frames to a `(3, 18, 30)` video grid — 405 tokens per level, matching the 1,620-token canonical interface (verified empirically). Per-frame sizing first rounds each dimension to a multiple of 64 (round-half-even), then applies a `[131072, 786432]` per-frame pixel window at factor 32, so unbudgeted sources produce 384–2,304 tokens per level (1,536–9,216 concatenated); a 640x480 source yields 3,840 concatenated tokens.
- The WISER robot-only renderer sets articulation configurations directly. Its action conversion is specific to the supported `pd_joint_pos` conventions and does not make arbitrary action representations renderable without state reconstruction.
