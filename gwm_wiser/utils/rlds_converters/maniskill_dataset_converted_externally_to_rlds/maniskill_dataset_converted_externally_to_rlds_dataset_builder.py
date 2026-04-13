from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import tensorflow_datasets as tfds


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError(
            f"Expected quaternion shapes (4,), got {q1.shape} and {q2.shape}"
        )
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_wxyz_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion (wxyz) to XYZ Euler angles (radians).

    Uses ManiSkill's internal rotation conversion utilities for consistency with its PDEEPoseController.
    """
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")

    import torch

    from mani_skill.utils.geometry import rotation_conversions

    qt = torch.from_numpy(q)
    m = rotation_conversions.quaternion_to_matrix(qt)
    e = rotation_conversions.matrix_to_euler_angles(m, "XYZ")
    return e.detach().cpu().numpy().astype(np.float32)


def _clip_rot_action_unit_norm(rot_action: np.ndarray) -> np.ndarray:
    rot_action = np.asarray(rot_action, dtype=np.float32)
    if rot_action.shape != (3,):
        raise ValueError(f"Expected rot_action shape (3,), got {rot_action.shape}")
    norm = float(np.linalg.norm(rot_action))
    if norm > 1.0:
        rot_action = rot_action / (norm + 1e-8)
    return np.clip(rot_action, -1.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=1)
def _get_panda_pinocchio_model() -> tuple[Any, int]:
    """
    Lazily build a Panda Pinocchio model for forward kinematics (no physics scene needed).

    Returns:
      (pin_model, tcp_link_index)
    """
    try:
        from sapien.wrapper.pinocchio_model import PinocchioModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SAPIEN is required for FK-based dataset conversion. "
            "Make sure you are running inside the conda env created by scripts/setup_conda_env.sh."
        ) from exc

    repo_root = Path(__file__).resolve().parents[3]
    urdf_path = (
        repo_root
        / ".deps"
        / "ManiSkill"
        / "mani_skill"
        / "assets"
        / "robots"
        / "panda"
        / "panda_v2.urdf"
    )
    if not urdf_path.exists():
        raise FileNotFoundError(
            "Could not locate Panda URDF for FK conversion. Expected: "
            f"{urdf_path}. Did you run scripts/setup_conda_env.sh to populate .deps/ManiSkill?"
        )

    urdf_xml = urdf_path.read_text()
    # Use Python stdlib XML parsing to avoid extra binary deps / potential thread-safety issues.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_xml.encode("utf-8"))
    link_order = [
        node.attrib["name"]
        for node in root
        if node.tag == "link" and "name" in node.attrib
    ]
    joint_order = [
        node.attrib["name"]
        for node in root
        if node.tag == "joint"
        and node.attrib.get("type") in {"revolute", "prismatic"}
        and "name" in node.attrib
    ]

    if not joint_order:
        raise RuntimeError(
            "Failed to infer joint_order from Panda URDF; FK conversion cannot continue."
        )
    if "panda_hand_tcp" not in link_order:
        raise RuntimeError(
            "Failed to find link 'panda_hand_tcp' in Panda URDF; FK conversion cannot continue."
        )

    pin_model = PinocchioModel(urdf_xml, [0, 0, -9.81])
    pin_model.set_joint_order(joint_order)
    pin_model.set_link_order(link_order)
    tcp_link_index = link_order.index("panda_hand_tcp")

    return pin_model, tcp_link_index


def _tcp_pose7_from_qpos9(qpos9: np.ndarray) -> np.ndarray:
    """
    Compute TCP pose (xyz + wxyz quaternion) in the robot base frame from Panda joint positions (9-dof).
    """
    qpos9 = np.asarray(qpos9, dtype=np.float32)
    if qpos9.shape != (9,):
        raise ValueError(f"Expected qpos9 shape (9,), got {qpos9.shape}")

    pin_model, tcp_link_index = _get_panda_pinocchio_model()
    pin_model.compute_forward_kinematics(qpos9)
    pose = pin_model.get_link_pose(tcp_link_index)
    p = np.asarray(pose.p, dtype=np.float32)
    q = np.asarray(pose.q, dtype=np.float32)  # wxyz
    tcp_pose = np.concatenate([p, q], axis=0).astype(np.float32)
    if tcp_pose.shape != (7,):
        raise ValueError(f"Expected tcp_pose shape (7,), got {tcp_pose.shape}")
    return tcp_pose


def _eef_delta_action7_from_qpos9_and_joint_target(
    qpos9: np.ndarray, *, target_arm_qpos7: np.ndarray, gripper_action: float
) -> np.ndarray:
    """
    Convert a Panda PD-joint-position target (7-d arm) into a ManiSkill `pd_ee_delta_pose` action (7-d, normalized).

    We:
      1) FK current qpos9 -> tcp pose
      2) FK target qpos9 (arm from target, fingers from current) -> tcp pose
      3) Convert pose delta into normalized controller action:
         - pos normalized by +/- pos_scale (default 0.1m)
         - rot normalized by rot_scale (default -0.1 rad) and clipped to unit norm (matches ManiSkill controller)
         - keep gripper action as-is (already normalized in [-1, 1] in LeRobot dumps)
    """
    qpos9 = np.asarray(qpos9, dtype=np.float32)
    target_arm_qpos7 = np.asarray(target_arm_qpos7, dtype=np.float32)
    if qpos9.shape != (9,):
        raise ValueError(f"Expected qpos9 shape (9,), got {qpos9.shape}")
    if target_arm_qpos7.shape != (7,):
        raise ValueError(
            f"Expected target_arm_qpos7 shape (7,), got {target_arm_qpos7.shape}"
        )

    pos_scale = float(os.environ.get("GWM_EE_DELTA_POS_SCALE", "0.1"))
    rot_scale = float(os.environ.get("GWM_EE_DELTA_ROT_SCALE", "-0.1"))
    if pos_scale <= 0:
        raise ValueError("GWM_EE_DELTA_POS_SCALE must be > 0")
    if rot_scale == 0:
        raise ValueError("GWM_EE_DELTA_ROT_SCALE must be != 0")

    tcp0 = _tcp_pose7_from_qpos9(qpos9)

    target_qpos9 = qpos9.copy()
    target_qpos9[:7] = target_arm_qpos7
    tcp1 = _tcp_pose7_from_qpos9(target_qpos9)

    dp = (tcp1[:3] - tcp0[:3]).astype(np.float32)
    q0 = tcp0[3:7]
    q1 = tcp1[3:7]

    # Root-aligned body rotation: q_target = q_delta * q_current  => q_delta = q_target * inv(q_current)
    q_delta = _quat_mul_wxyz(q1, _quat_inv_wxyz(q0))
    de = _quat_wxyz_to_euler_xyz(q_delta)

    pos_action = np.clip(dp / pos_scale, -1.0, 1.0).astype(np.float32)
    rot_action = _clip_rot_action_unit_norm((de / rot_scale).astype(np.float32))

    action = np.concatenate(
        [pos_action, rot_action, np.array([gripper_action], dtype=np.float32)], axis=0
    )
    if action.shape != (7,):
        raise ValueError(f"Expected action shape (7,), got {action.shape}")
    return action.astype(np.float32)


class ManiskillDatasetConvertedExternallyToRlds(tfds.core.GeneratorBasedBuilder):
    """TFDS/RLDS builder for WISER ManiSkill demos exported from LeRobot (real conversion)."""

    VERSION = tfds.core.Version("0.0.3")
    RELEASE_NOTES = {
        "0.0.3": "Keep raw Panda joint-position actions from LeRobot demos for joint-space VLA training.",
        "0.0.2": "Convert LeRobot qpos/actions to real tcp_pose + pd_ee_delta_pose actions.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Pass a LeRobot dataset directory via --manual_dir (the 'lerobot_data' folder)."
    )

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    # Matches OpenVLA/InstructVLA OXE config keys:
                                    #   image_obs_keys: primary -> "image", wrist -> "wrist_image"
                                    "image": tfds.features.Image(
                                        shape=(224, 448, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Workspace RGB observation (from LeRobot image_1).",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Wrist RGB observation (from LeRobot wrist_image).",
                                    ),
                                    # Depth keys exist in the upstream configs; we store empty strings for padding.
                                    "depth": tfds.features.Text(
                                        doc="Optional depth (PNG bytes). Empty for smoke."
                                    ),
                                    "wrist_depth": tfds.features.Text(
                                        doc="Optional wrist depth (PNG bytes). Empty for smoke."
                                    ),
                                    "joint_state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Panda arm joint positions (7-dof).",
                                    ),
                                    "gripper_state": tfds.features.Tensor(
                                        shape=(1,),
                                        dtype=np.float32,
                                        doc="Gripper state (mean finger qpos, meters).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="joint_state (7) + gripper_state (1).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float32,
                                doc="Absolute Panda joint-position target (7) + gripper command (1).",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language instruction (from LeRobot task)."
                            ),
                            "reasonings": tfds.features.Text(
                                doc="JSON annotation for InstructVLA (minimal placeholders)."
                            ),
                            # Required by RLDS format (kept simple for smoke).
                            "discount": tfds.features.Scalar(
                                dtype=np.float32, doc="Discount; defaults to 1.0."
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward; uses LeRobot next.reward if available.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on terminal step (last for demos).",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Episode id / source identifier."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        train_manual_dir = self._resolve_manual_dir(dl_manager.manual_dir)

        max_episodes = int(os.environ.get("GWM_RLDS_MAX_EPISODES", "0"))
        if max_episodes < 0:
            raise ValueError("GWM_RLDS_MAX_EPISODES must be >= 0")

        train_episode_ids = self._get_episode_ids_with_limit(
            train_manual_dir, max_episodes=max_episodes
        )
        return {
            "train": self._generate_examples(
                train_manual_dir, allowed_episode_ids=train_episode_ids
            ),
        }

    @staticmethod
    def _resolve_manual_dir(path: Optional[str]) -> str:
        if not path:
            raise ValueError(
                "You must provide the dataset path using --manual_dir (LeRobot 'lerobot_data')."
            )
        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(resolved):
            raise ValueError(f"Dataset directory does not exist: {resolved}")
        return resolved

    @classmethod
    def _get_episode_ids_with_limit(
        cls, manual_dir: str, *, max_episodes: int
    ) -> list[int]:
        episode_ids_from_env = os.environ.get("GWM_RLDS_EPISODE_IDS", "").strip()
        if episode_ids_from_env:
            episode_ids = [
                int(x.strip()) for x in episode_ids_from_env.split(",") if x.strip()
            ]
            if any(ep_id < 0 for ep_id in episode_ids):
                raise ValueError(
                    "GWM_RLDS_EPISODE_IDS must contain only non-negative episode ids"
                )
            if max_episodes > 0:
                episode_ids = episode_ids[:max_episodes]
            if len(episode_ids) == 0:
                raise ValueError(
                    "GWM_RLDS_EPISODE_IDS resolved to an empty episode list"
                )
            return episode_ids

        episode_ids = cls._infer_episode_ids(manual_dir) or cls._list_episode_ids(
            manual_dir
        )
        if max_episodes > 0:
            episode_ids = episode_ids[:max_episodes]
        if len(episode_ids) == 0:
            raise ValueError(f"No episodes found under manual_dir={manual_dir}")
        return episode_ids

    @staticmethod
    def _infer_episode_ids(manual_dir: str) -> Optional[list[int]]:
        """
        Fast path: infer episode ids from LeRobot `meta/info.json` without iterating frames (which decodes videos).

        We assume episode indices are contiguous `[0, total_episodes)`, which is how LeRobot writes splits like
        `"train": "0:<total_episodes>"`.
        """
        info_path = Path(manual_dir) / "meta" / "info.json"
        if not info_path.exists():
            return None
        try:
            info = json.loads(info_path.read_text())
        except Exception:  # noqa: BLE001
            return None

        total_episodes = info.get("total_episodes")
        if not isinstance(total_episodes, int) or total_episodes <= 0:
            return None
        return list(range(total_episodes))

    @staticmethod
    def _list_episode_ids(manual_dir: str) -> list[int]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset("wiser_smoke", root=manual_dir, video_backend="pyav")
        episode_ids: list[int] = []
        current: Optional[int] = None
        for frame in dataset:
            idx = int(frame["episode_index"])
            if current is None or idx != current:
                episode_ids.append(idx)
                current = idx
        return episode_ids

    @staticmethod
    def _to_uint8_hwc(image_chw) -> np.ndarray:
        # LeRobot commonly stores images as CHW float32 in [0, 1]. Handle uint8 too.
        if hasattr(image_chw, "cpu"):
            image_chw = image_chw.cpu()
        if hasattr(image_chw, "numpy"):
            image_chw = image_chw.numpy()
        image_chw = np.asarray(image_chw)

        if image_chw.ndim != 3:
            raise ValueError(
                f"Expected image with 3 dims (C,H,W); got shape={image_chw.shape}"
            )

        if image_chw.dtype == np.float32 or image_chw.dtype == np.float64:
            image_chw = np.clip(image_chw, 0.0, 1.0)
            image_chw = (255.0 * image_chw).astype(np.uint8)
        elif image_chw.dtype != np.uint8:
            image_chw = image_chw.astype(np.uint8)

        return np.transpose(image_chw, (1, 2, 0))

    @classmethod
    def _generate_examples(
        cls, manual_dir: str, *, allowed_episode_ids: Optional[list[int]] = None
    ) -> Iterator[tuple[str, Any]]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        allowed = set(allowed_episode_ids) if allowed_episode_ids is not None else None
        max_steps_per_episode = int(
            os.environ.get("GWM_RLDS_MAX_STEPS_PER_EPISODE", "32")
        )
        if max_steps_per_episode < 0:
            raise ValueError("GWM_RLDS_MAX_STEPS_PER_EPISODE must be >= 0")

        dataset_kwargs = {}
        if allowed_episode_ids is not None:
            dataset_kwargs["episodes"] = sorted(allowed_episode_ids)

        dataset = LeRobotDataset(
            "wiser_smoke", root=manual_dir, video_backend="pyav", **dataset_kwargs
        )

        # Remove video keys we don't need to avoid decoding them (speeds up conversion ~2x)
        _EXCLUDE_VIDEO_KEYS = [
            "observation.images.image_1_robot_state",
            "observation.images.image_1_segmentation_mask",
        ]
        for key in _EXCLUDE_VIDEO_KEYS:
            dataset.meta.info["features"].pop(key, None)

        steps = []
        current_episode_id: Optional[int] = None

        def maybe_yield_episode(ep_id: int, episode_steps: list[dict]):
            if allowed is not None and ep_id not in allowed:
                return None

            episode_steps[0]["is_first"] = True
            episode_steps[-1]["is_last"] = True
            episode_steps[-1]["is_terminal"] = True

            return str(ep_id), {
                "steps": episode_steps,
                "episode_metadata": {"file_path": str(ep_id)},
            }

        for frame in dataset:
            ep_id = int(frame["episode_index"])
            if current_episode_id is None:
                current_episode_id = ep_id

            if ep_id != current_episode_id:
                if steps:
                    maybe = maybe_yield_episode(current_episode_id, steps)
                    if maybe is not None:
                        yield maybe
                steps = []
                current_episode_id = ep_id

            if allowed is not None and current_episode_id not in allowed:
                continue

            if max_steps_per_episode > 0 and len(steps) >= max_steps_per_episode:
                continue

            # Language instruction
            language_instruction = frame.get("task", "")
            if isinstance(language_instruction, bytes):
                language_instruction = language_instruction.decode(
                    "utf-8", errors="ignore"
                )
            language_instruction = str(language_instruction)

            # Reward (optional; defaults to 0.0)
            reward = frame.get("next.reward", 0.0)
            if hasattr(reward, "item"):
                reward = reward.item()

            # Keep raw joint-space proprio + action semantics from expert demos.
            qpos9 = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
            if qpos9.shape != (9,):
                raise ValueError(
                    f"Expected LeRobot observation.state shape (9,), got {qpos9.shape}"
                )

            action_raw = np.asarray(frame["action"], dtype=np.float32).reshape(-1)
            if action_raw.shape != (8,):
                raise ValueError(
                    f"Expected LeRobot action shape (8,), got {action_raw.shape}"
                )

            joint_state = qpos9[:7].astype(np.float32)
            gripper_state = np.mean(qpos9[7:9], keepdims=True).astype(np.float32)
            state = np.concatenate([joint_state, gripper_state], axis=0).astype(
                np.float32
            )
            action = action_raw.astype(np.float32)

            steps.append(
                {
                    "observation": {
                        "image": cls._to_uint8_hwc(frame["observation.images.image_1"]),
                        "wrist_image": cls._to_uint8_hwc(
                            frame["observation.images.wrist_image"]
                        ),
                        "depth": "",
                        "wrist_depth": "",
                        "joint_state": joint_state,
                        "gripper_state": gripper_state,
                        "state": state,
                    },
                    "action": action,
                    "language_instruction": language_instruction,
                    "reasonings": '{"move_primitive": null, "alt_instruction": null}',
                    "discount": 1.0,
                    "reward": float(reward),
                    "is_first": False,
                    "is_last": False,
                    "is_terminal": False,
                }
            )

        if current_episode_id is not None and steps:
            maybe = maybe_yield_episode(current_episode_id, steps)
            if maybe is not None:
                yield maybe
