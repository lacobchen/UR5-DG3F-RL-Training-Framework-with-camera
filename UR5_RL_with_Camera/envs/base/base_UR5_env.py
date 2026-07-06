# envs/base/base_ur5_env.py

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import torch
from isaaclab.envs import DirectRLEnv
from .base_UR5_env_cfg import BaseUR5EnvCfg
from ..backend.simulation_backend import SimulationBackend


class BaseUR5Env(DirectRLEnv):

    cfg: BaseUR5EnvCfg

    def __init__(
        self,
        cfg: BaseUR5EnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # DirectRLEnv.__init__() 內部會呼叫 self._setup_scene()
        super().__init__(cfg, render_mode, **kwargs)

       
        if not getattr(self.backend, "_handles_initialized", False):
            self.backend.initialize_handles()

        # 將常用 handle 快取到 env 本身，讓子類別可以直接使用。
        self._robot_dof_idx = self.backend.robot_dof_idx
        self._robot_dof_names = self.backend.robot_dof_names

        self._ee_frame_idx = self.backend.ee_frame_idx
        self._ee_frame_names = self.backend.ee_frame_names
        self._ee_body_idx = self.backend.ee_body_idx
        self._ee_jacobi_idx = self.backend.ee_jacobi_idx

        self.arm_dof_count = int(self.cfg.arm_dof_count)

        # 暫存 action。
        self.actions = torch.zeros(
            (self.num_envs, int(self.cfg.action_space)),
            device=self.device,
            dtype=torch.float32,
        )

        # 預先建立 gripper mask，避免每個 step 重複建立 tensor。
        self._gripper_mask_cache: torch.Tensor | None = None

        if self.cfg.debug_print:
            print(">>> BaseUR5Env initialized")
            print(f"    num_envs          = {self.num_envs}")
            print(f"    action_space      = {self.cfg.action_space}")
            print(f"    observation_space = {self.cfg.observation_space}")
            print(f"    robot_dof_idx     = {self._robot_dof_idx}")
            print(f"    ee_body_idx       = {self._ee_body_idx}")
            print(f"    ee_jacobi_idx     = {self._ee_jacobi_idx}")

    # ============================================================
    # 1. Scene setup
    # ============================================================

    def _setup_scene(self):
        
        if not hasattr(self.cfg, "cube_cfg"):
            raise AttributeError("需要 cfg.cube_cfg")

         # 改成 optional，camera 只在 task cfg 定義時才啟用
        has_camera = hasattr(self.cfg, "camera_cfg")

        if not has_camera:
             import warnings
             warnings.warn(
                    "cfg.camera_cfg 未定義，camera 不會被建立。"
                    "視覺 observation 將無法使用。"
            )
        self.backend = SimulationBackend(
            cfg=self.cfg,
            scene=self.scene,
            device=self.device,
            robot_key="robot",
            cube_key="cube",
            ee_body_name=self.cfg.ee_body_name,
            robot_joint_expr=self.cfg.robot_joint_expr,
            arm_dof_count=self.cfg.arm_dof_count,
            enable_lula_ik=self.cfg.enable_lula_ik,
            jacobian_body_index_offset=self.cfg.jacobian_body_index_offset,
        )

        self.robot, self.cube = self.backend.setup_scene()

    # ============================================================
    # 2. RL step: receive action
    # ============================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Isaac Lab 每個 step 在 physics simulation 前會先呼叫這個函式。

        這裡只負責暫存 RL policy 給的 action。
        真正把 action 轉成 joint target 的工作在 _apply_action() 做。
        """

        self.actions = actions.clone().to(device=self.device, dtype=torch.float32)

    def _apply_action(self):
        """
        將 RL action 轉換成 robot joint target，並寫入 Isaac Sim。

        流程：
            action
              ↓
            compute_control_targets()
              ↓
            target_joint_pos
              ↓
            backend.apply_settings()
              ↓
            Isaac Sim / Isaac Lab
        """

        target_joint_pos = self.compute_control_targets(self.actions)
        self.backend.apply_settings(target_joint_pos)

    # ============================================================
    # 3. Action -> joint target
    # ============================================================

    def compute_control_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """
        將 RL action 轉成 UR5 + DG3F 的 joint position target。
        """

        if actions.ndim != 2:
            raise ValueError(
                f"actions 必須是 2D tensor [num_envs, action_dim]，"
                f"但收到 shape = {tuple(actions.shape)}"
            )

        if actions.shape[1] < 3:
            raise ValueError(
                f"actions 至少需要前 3 維作為 TCP delta xyz，"
                f"但目前 action_dim = {actions.shape[1]}"
            )

        current_joint_pos = self.backend.get_current_control_joint_pos()
        target_joint_pos = current_joint_pos.clone()

        # ------------------------------------------------------------
        # 1. Cartesian delta action
        # ------------------------------------------------------------
        delta_x = actions[:, 0:3] * float(self.cfg.tcp_delta_scale)

        # ------------------------------------------------------------
        # 2. DiffIK: delta x -> delta q
        # ------------------------------------------------------------
        delta_q = self.compute_diffik_delta_q(delta_x)

        arm_dof_count = int(self.cfg.arm_dof_count)
        target_joint_pos[:, :arm_dof_count] += delta_q

        # ------------------------------------------------------------
        # 3. Gripper control
        # ------------------------------------------------------------
        gripper_target = self.compute_gripper_target(
            actions=actions,
            target_joint_pos=target_joint_pos,
        )

        target_joint_pos[:, arm_dof_count:] = gripper_target

        return target_joint_pos

    def compute_diffik_delta_q(self, delta_x: torch.Tensor) -> torch.Tensor:
        """
        使用 end-effector translational Jacobian 計算 UR5 前 6 軸的 delta_q。

        原理：
            J * delta_q = delta_x

        使用 pseudo-inverse：
            delta_q = pinv(J) * delta_x

        Args:
            delta_x:
                shape = [num_envs, 3]

        Returns:
            delta_q:
                shape = [num_envs, arm_dof_count]
        """

        if delta_x.ndim != 2 or delta_x.shape[1] != 3:
            raise ValueError(
                f"delta_x 必須是 [num_envs, 3]，但收到 shape = {tuple(delta_x.shape)}"
            )

        J = self.backend.get_ee_translation_jacobian(arm_only=True)
        # J shape: [num_envs, 3, arm_dof_count]

        if bool(self.cfg.diffik_use_damped_pinv):
            J_pinv = self._damped_pseudo_inverse(
                J,
                damping=float(self.cfg.diffik_damping_lambda),
            )
        else:
            J_pinv = torch.linalg.pinv(J)

        # J_pinv shape: [num_envs, arm_dof_count, 3]
        # delta_x.unsqueeze(-1): [num_envs, 3, 1]
        delta_q = torch.bmm(J_pinv, delta_x.unsqueeze(-1)).squeeze(-1)

        return delta_q

    def compute_gripper_target(
        self,
        actions: torch.Tensor,
        target_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        根據 action 計算 DG3F 夾爪 target。
            raw_actions[:, 6] > 0  -> 夾緊
            raw_actions[:, 6] <= 0 -> 張開
        套用 gripper_joint_mask：
            mask = 0 的關節保持 open value
            mask = 1 的關節移動到 close value
        """

        arm_dof_count = int(self.cfg.arm_dof_count)
        num_gripper_dofs = target_joint_pos.shape[1] - arm_dof_count

        if num_gripper_dofs <= 0:
            return torch.empty(
                (target_joint_pos.shape[0], 0),
                device=self.device,
                dtype=torch.float32,
            )

        signal = self._extract_gripper_signal(actions)

        if bool(self.cfg.use_binary_gripper):
            close_ratio = torch.where(
                signal > 0.0,
                torch.ones_like(signal),
                torch.zeros_like(signal),
            )
        else:
            # 將 action 由 [-1, 1] 線性映射到 [0, 1]
            close_ratio = torch.clamp((signal + 1.0) * 0.5, 0.0, 1.0)

        close_ratio = close_ratio.unsqueeze(-1)

        gripper_mask = self._get_gripper_mask(num_gripper_dofs)
        gripper_mask = gripper_mask.unsqueeze(0)

        open_value = float(self.cfg.gripper_open_value)
        close_value = float(self.cfg.gripper_close_value)

        # mask = 0:
        #   target = open_value
        #
        # mask = 1:
        #   target = open + ratio * (close - open)
        gripper_target = open_value + close_ratio * (close_value - open_value) * gripper_mask

        return gripper_target

    # ============================================================
    # 4. Reset
    # ============================================================

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """
        共用 reset 流程。

        目前預設：
            1. 呼叫 DirectRLEnv 的 reset buffer 邏輯
            2. reset cube
            3. reset robot 到 cube 上方
        task env 若有更複雜的任務，可以 override 這個函式。
        """

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self.backend.reset_all(
            env_ids=env_ids,
            randomize_cube_xy=bool(self.cfg.randomize_cube_xy),
            cube_xy_range=float(self.cfg.cube_xy_range),
            robot_above_cube=bool(self.cfg.reset_robot_above_cube),
            height_above_cube=float(self.cfg.height_above_cube),
        )

    # ============================================================
    # 5. Task-specific functions
    # ============================================================
    # 這三個函式理論上應該由 task env override。
    # 例如：
    #   TaskPickCubeEnv(BaseUR5Env)
    #       _get_observations()
    #       _get_rewards()
    #       _get_dones()
    # ============================================================

    def _get_observations(self) -> dict:
        """
        真正訓練時，建議在 task_pick_cube_env.py override。
        """

        state = self.backend.get_information()

        joint_pos = state["joint_pos"]
        joint_vel = state["joint_vel"]
        ee_pos = state["ee_pos"]

        # 如果 task 沒有 cube，這裡會出問題。
        # 目前你的架構是 pick cube，所以保留 cube_pos。
        cube_pos = state["cube_pos"]

        obs = torch.cat(
            [
                ee_pos,
                cube_pos,
                joint_pos,
                joint_vel,
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        真正訓練時，請在 task_pick_cube_env.py override。
        """

        return torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        請在 task_pick_cube_env.py override。
        """

        died = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    # ============================================================
    # 6. Helper functions
    # ============================================================

    def get_sim_state(self) -> dict[str, Any]:
        """
        給 task env 使用的簡潔 state 介面。
        """

        return self.backend.get_information()

    def _extract_gripper_signal(self, actions: torch.Tensor) -> torch.Tensor:

        action_dim = actions.shape[1]
        gripper_action_index = int(self.cfg.gripper_action_index)

        if action_dim > gripper_action_index:
            return actions[:, gripper_action_index]

        # 如果之後把 action_space 改成 4：
        #   [dx, dy, dz, gripper]
        # 這裡會自動使用最後一維作為 gripper signal。
        if action_dim >= 4:
            return actions[:, -1]

        # 如果 action 只有 3 維，代表沒有 gripper action。
        # 預設保持張開。
        return torch.zeros(
            actions.shape[0],
            device=self.device,
            dtype=torch.float32,
        )

    def _get_gripper_mask(self, num_gripper_dofs: int) -> torch.Tensor:

        if (
            self._gripper_mask_cache is not None
            and self._gripper_mask_cache.numel() == num_gripper_dofs
        ):
            return self._gripper_mask_cache

        mask = torch.tensor(
            self.cfg.gripper_joint_mask,
            device=self.device,
            dtype=torch.float32,
        )

        if mask.numel() > num_gripper_dofs:
            mask = mask[:num_gripper_dofs]

        elif mask.numel() < num_gripper_dofs:
            pad = torch.ones(
                num_gripper_dofs - mask.numel(),
                device=self.device,
                dtype=torch.float32,
            )
            mask = torch.cat([mask, pad], dim=0)

        self._gripper_mask_cache = mask

        return mask

    def _damped_pseudo_inverse(
        self,
        J: torch.Tensor,
        damping: float = 0.05,
    ) -> torch.Tensor:

        if J.ndim != 3:
            raise ValueError(
                f"J 必須是 3D tensor [num_envs, task_dim, dof]，"
                f"但收到 shape = {tuple(J.shape)}"
            )

        batch_size = J.shape[0]
        task_dim = J.shape[1]

        J_T = J.transpose(1, 2)

        identity = torch.eye(
            task_dim,
            device=J.device,
            dtype=J.dtype,
        ).unsqueeze(0).repeat(batch_size, 1, 1)

        damping_matrix = (damping ** 2) * identity

        # J^T (J J^T + λ^2 I)^-1
        J_pinv = torch.bmm(
            J_T,
            torch.linalg.inv(torch.bmm(J, J_T) + damping_matrix),
        )

        return J_pinv