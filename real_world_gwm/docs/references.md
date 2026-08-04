# References

## Primary external sources

- [Grounded World Model for Semantically Generalizable Planning](https://arxiv.org/html/2604.11751) — GWM, RAT, WISER, and the reported planning results.
- [RobotSeg repository](https://github.com/showlab/RobotSeg) — VRS download, dataset summary, masks, code license, and training configuration.
- [RobotSeg paper, section 3](https://arxiv.org/html/2511.22950v2#S3) — VRS construction, embodiments, clip/frame counts, and annotation protocol.
- [RobotSeg VRS loader](https://github.com/showlab/RobotSeg/blob/main/train/dataset/vos_raw_dataset.py) — ordered JPG frames and mask categories `000=arm`, `001=gripper`, `002=whole robot`.
- [RobotSeg training configuration](https://github.com/showlab/RobotSeg/blob/main/robotseg/configs/robotseg-train.yaml) — released VRS paths and pseudo-mask training input.
- [Qwen3-VL-Embedding-8B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — multimodal embedding model and representation dimensions.
- [Qwen3-VL-Embedding-8B configuration](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/config.json) — visual patch, merge, DeepStack, and hidden-size configuration.

## Deferred dataset-survey leads

These links are discovery leads supplied for a later survey, not yet validated as suitable training corpora:

- [datasets.bot](https://datasets.bot/) — catalog to search for robot datasets with RGB, synchronized robot state, and calibration metadata.
- [MolmoBot](https://github.com/allenai/MolmoBot) — candidate simulation-data source to inspect for recoverable RAT supervision.
- [Exylos Bimanual Table Spill Cleanup](https://huggingface.co/datasets/ExylosAi/table_spill_cleanup_bimanual) — state- and mask-rich lead retained from the former `origin/dev` survey; current suitability is unverified.
- [NVIDIA PhysicalAI Robotics Manipulation Augmented](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented) — simulation lead retained from the former `origin/dev` survey; qpos, calibration, label, and license suitability require revalidation.
- [RoboEngine/RoboSeg](https://github.com/michaelyuancb/roboengine) — segmentation reference retained from the old survey; it was not established as a consecutive phase-one training corpus.
- [Dexora Real-World Dataset](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset) — low-confidence state-rich lead from the old survey; embodiment, qpos, main-camera, mask, calibration, and license claims require inspection.
- [RoVi-Aug](https://github.com/BerkeleyAutomation/rovi-aug) — robot-mask/data-generation reference from the old survey, not a validated ready-to-train corpus.

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
- The WISER robot-only renderer sets articulation configurations directly. Its action conversion is specific to the supported `pd_joint_pos` conventions and does not make arbitrary action representations renderable without state reconstruction.
