# envs/tasks/__init__.py
from __future__ import annotations
import gymnasium as gym
from . import agents  # noqa: F401

gym.register(
    id="UR5-PickCube-v0",
    entry_point="envs.tasks.task_pick_cube_env:TaskPickCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.tasks.task_pick_cube_cfg:TaskPickCubeEnvCfg",
        #  RL library
        "rsl_rl_cfg_entry_point": "envs.tasks.agents.rsl_rl_ppo_cfg:PickCubeRslRlPpoCfg",
    },
)

from .task_pick_cube_env import TaskPickCubeEnv
from .task_pick_cube_cfg import TaskPickCubeEnvCfg