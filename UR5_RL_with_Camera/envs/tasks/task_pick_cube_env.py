# envs/tasks/task_pick_cube_env.py

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .task_pick_cube_cfg import TaskPickCubeEnvCfg
from ..base.base_UR5_env import BaseUR5Env
from ..vision_encoder import DepthEncoder

class TaskPickCubeEnv(BaseUR5Env):

    cfg: TaskPickCubeEnvCfg

    def __init__(self, cfg: TaskPickCubeEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 初始化視覺 encoder，放到與 env 相同的 device
        self.vision_encoder = DepthEncoder(
            img_size=cfg.camera_cfg.height,   # 從 cfg 讀，避免硬編碼
            feature_dim=64,
        ).to(self.device)

        # 凍結 encoder 參數（目前採用 freeze 策略）
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    # ============================================================
    # 1. Observation
    # ============================================================
    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self.get_sim_state()

        ee_pos    = state["ee_pos"]
        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]

        top_depth = state["top_camera_depth"]
        ee_depth  = state["ee_camera_depth"]

        top_depth_feat = self.encode_depth(top_depth)
        ee_depth_feat  = self.encode_depth(ee_depth)

        obs = torch.cat(
            [
                top_depth_feat,
                ee_depth_feat,
                ee_pos,
                joint_pos,
                joint_vel,
            ],
            dim=-1,
        )

        return {"policy": obs}

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        x = depth.permute(0, 3, 1, 2).contiguous().float()

        x = torch.nan_to_num(x, nan=5.0, posinf=5.0)
        x = torch.clamp(x, 0.0, 5.0) / 5.0

        with torch.no_grad():
            return self.vision_encoder(x)
    # ============================================================
    # 2. Reward
    # ============================================================

    def _get_rewards(self) -> torch.Tensor:
        """
        計算抓方塊任務的 reward。

        目前 reward 設計：

            reward = grasp_reward + lift_reward - smash_penalty

        其中：

            grasp_reward:
                鼓勵夾爪收合，讓 policy 有動力學會夾緊。

            lift_reward:
                當方塊被抬起時給予大量獎勵。

            smash_penalty:
                若方塊被壓進地板，給予懲罰。
        """

        state = self.get_sim_state()

        cube_pos = state["cube_pos"]
        gripper_joint_pos = state["gripper_joint_pos"]

        # ------------------------------------------------------------
        # 1. Grasp reward
        # ------------------------------------------------------------
        gripper_closed_amount = torch.sum(gripper_joint_pos, dim=1)

        grasp_reward = (
            gripper_closed_amount
            * float(self.cfg.grasp_reward_weight)
        )

        # ------------------------------------------------------------
        # 2. Lift reward
        # ------------------------------------------------------------
        cube_height = cube_pos[:, 2]
        lift_height = cube_height - float(self.cfg.cube_rest_height)

        lift_reward = torch.where(
            lift_height > float(self.cfg.lift_reward_threshold),
            lift_height * float(self.cfg.lift_reward_weight),
            torch.zeros_like(lift_height),
        )

        # ------------------------------------------------------------
        # 3. Smash penalty
        # ------------------------------------------------------------
        smash_penalty = torch.where(
            cube_height < float(self.cfg.cube_smash_height),
            torch.full_like(cube_height, float(self.cfg.smash_penalty_value)),
            torch.zeros_like(cube_height),
        )

        # ------------------------------------------------------------
        # 4. Total reward
        # ------------------------------------------------------------
        reward = grasp_reward + lift_reward - smash_penalty

        reward = torch.clamp(
            reward,
            min=float(self.cfg.reward_min_value),
        )

        # ------------------------------------------------------------
        # 5. Debug 
        # ------------------------------------------------------------
        if (
            bool(self.cfg.debug_print)
            and int(self.cfg.reward_print_interval) > 0
            and self.common_step_counter % int(self.cfg.reward_print_interval) == 0
        ):
            print(
                f" [env 0] "
                f"gripper_closed = {gripper_closed_amount[0].item():.3f} | "
                f"cube_z = {cube_height[0].item():.4f} m | "
                f"lift_height = {lift_height[0].item():.4f} m | "
                f"grasp_reward = {grasp_reward[0].item():.2f} | "
                f"lift_reward = {lift_reward[0].item():.2f} | "
                f"smash_penalty = {smash_penalty[0].item():.2f} | "
                f"reward = {reward[0].item():.2f}"
            )

        return reward

    # ============================================================
    # 3. Done
    # ============================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        判斷 episode 是否結束。

        回傳：
            died:
                非時間到的終止條件，例如：
                    - 方塊被推太遠
                    - 方塊掉到地板以下
                    - 可選：成功抓起後結束

            time_out:
                episode 時間到
        """

        state = self.get_sim_state()

        cube_pos = state["cube_pos"]
        env_origins = state["env_origins"]

        # ------------------------------------------------------------
        # 1. Timeout
        # ------------------------------------------------------------
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # ------------------------------------------------------------
        # 2. Cube pushed too far
        # ------------------------------------------------------------
        cube_xy = cube_pos[:, :2]
        env_origin_xy = env_origins[:, :2]

        cube_dist_from_origin = torch.norm(
            cube_xy - env_origin_xy,
            dim=1,
        )

        pushed_too_far = (
            cube_dist_from_origin
            > float(self.cfg.cube_push_done_dist)
        )

        # ------------------------------------------------------------
        # 3. Cube fell below ground
        # ------------------------------------------------------------
        fell_under_ground = (
            cube_pos[:, 2]
            < float(self.cfg.cube_fall_done_height)
        )

        # ------------------------------------------------------------
        # 4. Optional success termination
        # ------------------------------------------------------------
        if bool(self.cfg.done_on_success):
            lift_height = cube_pos[:, 2] - float(self.cfg.cube_rest_height)

            success = (
                lift_height
                > float(self.cfg.success_lift_height)
            )
        else:
            success = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )

        died = pushed_too_far | fell_under_ground | success

        return died, time_out

    # ============================================================
    # 4. Reset
    # ============================================================

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):

        super()._reset_idx(env_ids)


    def get_task_info(self) -> dict[str, Any]:


        state = self.get_sim_state()

        cube_pos = state["cube_pos"]
        ee_pos = state["ee_pos"]
        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]
        gripper_joint_pos = state["gripper_joint_pos"]

        cube_height = cube_pos[:, 2]
        lift_height = cube_height - float(self.cfg.cube_rest_height)

        ee_to_cube = cube_pos - ee_pos
        ee_cube_dist = torch.norm(ee_to_cube, dim=1)

        gripper_closed_amount = torch.sum(gripper_joint_pos, dim=1)

        info = {
            "cube_pos": cube_pos,
            "ee_pos": ee_pos,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "lift_height": lift_height,
            "ee_to_cube": ee_to_cube,
            "ee_cube_dist": ee_cube_dist,
            "gripper_closed_amount": gripper_closed_amount,
        }

        return info