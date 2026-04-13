import torch
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tensordict import TensorDict

from gwm_wiser.env.wiser_env import WISEREnv


class EnvWrapper:
    """Implement some necessary interfaces according to rsl_rl.vec_env"""

    def __init__(self, *, env):
        self.mani_skill_env = env
        self.num_actions = env.base_env.action_space.shape[-1]
        self.num_envs = env.base_env.num_envs
        self.device = env.base_env.device
        self._current_obs = None
        self.unwrapped = self.mani_skill_env.unwrapped

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """
        Unlike Done, Truncate doesn't mean the over of the episode.
        """
        o, r, terminations, truncations, info = self.mani_skill_env.step(actions)
        assert truncations is self.mani_skill_env.base_env.truncated
        dones = torch.logical_or(terminations, truncations)

        # for rsl-rl
        info["time_outs"] = self.mani_skill_env.base_env.truncated
        return self._build_observations(o), r, dones, info

    def reset(self, *args, **kwargs):
        o, info = self.mani_skill_env.reset(*args, **kwargs)
        return self._build_observations(o), info

    def get_observations(self):
        """
        For RSL-RL to build model and so on. Only called once
        :return: dummy observations
        """
        return self._current_obs

    def _build_observations(self, observations) -> TensorDict:
        self._current_obs = TensorDict(
            observations, batch_size=self.num_envs, device=self.device
        )
        return self._current_obs


def build_endless_env(env_cfg, data_record_dir=None, record_video=False):
    """
    Build environment with optional Maniskill recording wrapper. It will not be reset automatically in env.step.
    :param env_cfg: env cfg
    :param data_record_dir: directory to save recorded data. If None, no data will be recorded.
    :param record_video: whether to save videos during recording.
    :return: Wrapped env class
    """
    env = WISEREnv(**env_cfg)
    if record_video:
        assert data_record_dir is not None, (
            "Please provide data_record_dir to save recorded video!"
        )

    if data_record_dir is not None:
        assert data_record_dir is not None, (
            "Please provide data_record_dir to save recorded data!"
        )
        env = RecordEpisode(
            env,
            output_dir="{}".format(data_record_dir),
            save_trajectory=False,
            info_on_video=False,
            trajectory_name="trajectory",
            max_steps_per_video=env_cfg["max_episode_steps"],
            video_fps=15,
            save_video=record_video,
        )

    logger.info(
        "Endless env will not be terminated and reset automatically in env.step! "
        "Please control its termination with loop outside and check the success from info dict."
    )
    env = ManiSkillVectorEnv(
        env,
        num_envs=env_cfg["num_envs"],
        ignore_terminations=True,  # still terminate episode!
        auto_reset=False,
        record_metrics=True,
    )
    env = EnvWrapper(env=env)
    env.reset()
    return env
