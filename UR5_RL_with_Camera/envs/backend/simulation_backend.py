from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import TiledCamera, TiledCameraCfg



class SimulationBackend:
    """
    負責：
        1. 建立 Isaac Lab 場景中的 robot / cube / ground / light
        2. 讀取 robot / cube 的真實模擬狀態
        3. 寫入 robot 的 joint target
        4. 提供 Jacobian 給 controller 使用
        5. reset robot / cube
        6. 提供get_information() / apply_settings() 介面
    """

    def __init__(
        self,
        cfg,
        scene,
        device: str | torch.device,
        *,
        robot_key: str = "robot",
        cube_key: str = "cube",
        ee_body_name: str = "l_dg_mount",
        robot_joint_expr: str = ".*",
        arm_dof_count: int = 6,
        enable_lula_ik: bool = True,
        jacobian_body_index_offset: int = -1,
        ground_prim_path: str = "/World/ground",
        light_prim_path: str = "/World/Light",
        light_intensity: float = 2000.0,
    ):
        self.cfg = cfg
        self.scene = scene
        self.device = device

        self.robot_key = robot_key
        self.cube_key = cube_key

        self.ee_body_name = ee_body_name
        self.robot_joint_expr = robot_joint_expr
        self.arm_dof_count = arm_dof_count

        self.enable_lula_ik = enable_lula_ik
        self.jacobian_body_index_offset = jacobian_body_index_offset

        self.ground_prim_path = ground_prim_path
        self.light_prim_path = light_prim_path
        self.light_intensity = light_intensity

        self.robot: Articulation | None = None
        self.cube: RigidObject | None = None

        self.robot_dof_idx = None
        self.robot_dof_names = None

        self.ee_frame_idx = None
        self.ee_frame_names = None
        self.ee_body_idx: int | None = None
        self.ee_jacobi_idx: int | None = None

        self.lula_ik = None
        self.target_quat = None

        self._handles_initialized = False
        self.top_camera: TiledCamera | None = None
        self.ee_camera: TiledCamera | None = None
    # ============================================================
    # 1. 場景設置
    # ============================================================

    def setup_scene(self) -> tuple[Articulation, RigidObject]:
        """
        建立 Isaac Lab 場景中的 robot、cube、ground、light。

        在 BaseUR5Env._setup_scene() 裡呼叫：

            self.backend = SimulationBackend(self.cfg, self.scene, self.device)
            self.robot, self.cube = self.backend.setup_scene()
        """

        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)
        # Top-view camera
        if hasattr(self.cfg, "camera_cfg"):
            self.top_camera = TiledCamera(self.cfg.camera_cfg)
            self.scene.sensors["top_camera"] = self.top_camera

        # End-effector camera
        if hasattr(self.cfg, "ee_camera_cfg"):
            self.ee_camera = TiledCamera(self.cfg.ee_camera_cfg)
            self.scene.sensors["ee_camera"] = self.ee_camera
            
        spawn_ground_plane(
            prim_path=self.ground_prim_path,
            cfg=GroundPlaneCfg(),
        )

        self.scene.clone_environments(copy_from_source=False)

        if self._device_is_cpu():
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations[self.robot_key] = self.robot
        self.scene.rigid_objects[self.cube_key] = self.cube

        light_cfg = sim_utils.DomeLightCfg(
            intensity=self.light_intensity,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func(self.light_prim_path, light_cfg)

        return self.robot, self.cube
    def _check_camera_exists(self) -> None:
        if self.top_camera is None:
            raise RuntimeError("top_camera 尚未建立，請確認 cfg.camera_cfg。")
        if self.ee_camera is None:
            raise RuntimeError("ee_camera 尚未建立，請確認 cfg.ee_camera_cfg。")
        
    def initialize_handles(self) -> None:
        """
        初始化 robot joint index、end-effector body index、Jacobian index、Lula IK。

        """

        self._check_robot_and_cube_exist()

        self.robot_dof_idx, self.robot_dof_names = self.robot.find_joints(
            self.robot_joint_expr
        )

        self.ee_frame_idx, self.ee_frame_names = self.robot.find_bodies(
            self.ee_body_name
        )

        if len(self.ee_frame_idx) == 0:
            raise RuntimeError(
                f"找不到 end-effector body：{self.ee_body_name}。"
                "請確認 USD 裡的 body 名稱是否正確。"
            )

        self.ee_body_idx = int(self.ee_frame_idx[0])

        # Isaac Lab / PhysX 中，如果 robot 是 fixed-base，
        # Jacobian link index 通常會比 body index 少 1。
        self.ee_jacobi_idx = self.ee_body_idx + self.jacobian_body_index_offset

        if self.ee_jacobi_idx < 0:
            raise RuntimeError(
                f"ee_jacobi_idx = {self.ee_jacobi_idx}，數值不合理。"
                "請檢查 jacobian_body_index_offset 是否需要調整。"
            )

        if self.enable_lula_ik:
            self._init_lula_ik()

        self._handles_initialized = True

    # ============================================================
    # 2. 輸入:apply_settings / set target
    # ============================================================

    def apply_settings(self, target_joint_pos: torch.Tensor) -> None:
        """
        這裡的 settings 為「控制目標」，
        目前就是 joint position target。
        Args:
            target_joint_pos:
                shape = [num_envs, num_controlled_joints]
        """

        self.set_joint_position_target(target_joint_pos)

    def set_joint_position_target(
        self,
        target_joint_pos: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """
        將 joint position target 寫入 Isaac Sim。

        Args:
            target_joint_pos:
                目標關節角度。
                如果 joint_ids 是 robot_dof_idx，shape 通常為：
                [num_envs, len(robot_dof_idx)]

            joint_ids:
                要控制的 joint id。
                若為 None，預設使用 self.robot_dof_idx。
        """

        self._ensure_handles()

        if joint_ids is None:
            joint_ids = self.robot_dof_idx

        target_joint_pos = target_joint_pos.to(self.device)

        self.robot.set_joint_position_target(
            target_joint_pos,
            joint_ids=joint_ids,
        )

    def write_joint_state_to_sim(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """
        直接把 joint position / velocity 寫進模擬器。
        通常只在 reset 時使用，不建議每個 step 都用。
        """

        self._ensure_handles()

        env_ids = self._normalize_env_ids(env_ids)

        self.robot.write_joint_state_to_sim(
            joint_pos.to(self.device),
            joint_vel.to(self.device),
            env_ids=env_ids,
        )

    # ============================================================
    # 3. 輸出：get_information / get_state
    # ============================================================

    def get_information(self) -> dict[str, Any]:

        return self.get_state()

    def get_state(self) -> dict[str, Any]:

        self._ensure_handles()
        self._check_camera_exists()
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()

        cube_pos = self.get_cube_pos()
        cube_quat = self._safe_get_tensor(self.cube.data, "root_quat_w")
        cube_lin_vel = self._safe_get_tensor(self.cube.data, "root_lin_vel_w")
        cube_ang_vel = self._safe_get_tensor(self.cube.data, "root_ang_vel_w")
        cube_root_state = self._safe_get_tensor(self.cube.data, "root_state_w")

        ee_pos = self.get_ee_pos()
        ee_quat = self.get_ee_quat()
        
        # data.output 是 dict，key 由 cfg.data_types 決定
        top_output = self.top_camera.data.output
        ee_output = self.ee_camera.data.output

        top_depth = top_output["depth"]
        ee_depth = ee_output["depth"]

        top_rgb = top_output.get("rgb", None)
        ee_rgb = ee_output.get("rgb", None)

        state = {
            # raw objects
            "robot": self.robot,
            "cube": self.cube,
            "scene": self.scene,

            # indices
            "robot_dof_idx": self.robot_dof_idx,
            "robot_dof_names": self.robot_dof_names,
            "ee_body_idx": self.ee_body_idx,
            "ee_jacobi_idx": self.ee_jacobi_idx,

            # robot state
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "arm_joint_pos": joint_pos[:, : self.arm_dof_count],
            "arm_joint_vel": joint_vel[:, : self.arm_dof_count],
            "gripper_joint_pos": joint_pos[:, self.arm_dof_count :],
            "gripper_joint_vel": joint_vel[:, self.arm_dof_count :],

            # end-effector state
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,

            # cube state
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "cube_lin_vel": cube_lin_vel,
            "cube_ang_vel": cube_ang_vel,
            "cube_root_state": cube_root_state,

            # env information
            "env_origins": self.scene.env_origins,

            "top_camera_rgb": top_rgb,
            "top_camera_depth": top_depth,

            "ee_camera_rgb": ee_rgb,
            "ee_camera_depth": ee_depth,

            # 保留舊 key，避免其他舊程式壞掉
            "camera_rgb": top_rgb,
            "camera_depth": top_depth,
        }

        return state

    def get_joint_pos(self, controlled_dofs_only: bool = True) -> torch.Tensor:
        self._ensure_handles()

        if controlled_dofs_only:
            return self.robot.data.joint_pos[:, self.robot_dof_idx]

        return self.robot.data.joint_pos

    def get_joint_vel(self, controlled_dofs_only: bool = True) -> torch.Tensor:
        self._ensure_handles()

        if controlled_dofs_only:
            return self.robot.data.joint_vel[:, self.robot_dof_idx]

        return self.robot.data.joint_vel

    def get_ee_pos(self) -> torch.Tensor:
        self._ensure_handles()
        return self.robot.data.body_pos_w[:, self.ee_body_idx]

    def get_ee_quat(self) -> torch.Tensor | None:
        self._ensure_handles()
        return self._safe_get_tensor(self.robot.data, "body_quat_w", index=self.ee_body_idx)

    def get_cube_pos(self) -> torch.Tensor:
        self._check_robot_and_cube_exist()
        return self.cube.data.root_pos_w

    # ============================================================
    # 4. Jacobian interface
    # ============================================================

    def get_jacobian(
        self,
        *,
        body_jacobian_idx: int | None = None,
        translational_only: bool = False,
        arm_only: bool = False,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        取得 Jacobian。

        Args:
            body_jacobian_idx:
                要取哪一個 body 的 Jacobian。
                若為 None，預設取 end-effector 的 Jacobian。

            translational_only:
                True 時只取 XYZ 平移 Jacobian，shape = [num_envs, 3, dof]

            arm_only:
                True 時只取 UR5 前 6 軸的 Jacobian。

            joint_ids:
                指定要取哪些 joint columns。
                若 arm_only=True，則會優先使用前 arm_dof_count 個 joint。
        """

        self._ensure_handles()

        if body_jacobian_idx is None:
            body_jacobian_idx = self.ee_jacobi_idx

        jacobian = self.robot.root_physx_view.get_jacobians()
        jacobian = jacobian[:, body_jacobian_idx, :, :]

        if translational_only:
            jacobian = jacobian[:, :3, :]

        if arm_only:
            arm_joint_ids = self.robot_dof_idx[: self.arm_dof_count]
            jacobian = jacobian[..., arm_joint_ids]
        elif joint_ids is not None:
            jacobian = jacobian[..., joint_ids]

        return jacobian

    def get_ee_translation_jacobian(self, arm_only: bool = True) -> torch.Tensor:
        """
        取得 end-effector 的平移 Jacobian。

        預設回傳：
            shape = [num_envs, 3, 6]

        這個函式可以給 DiffIK controller 使用。
        """

        return self.get_jacobian(
            body_jacobian_idx=self.ee_jacobi_idx,
            translational_only=True,
            arm_only=arm_only,
        )

    # ============================================================
    # 5. Reset interface
    # ============================================================

    def reset_all(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        randomize_cube_xy: bool = True,
        cube_xy_range: float = 0.05,
        robot_above_cube: bool = True,
        height_above_cube: float = 0.25,
    ) -> dict[str, Any]:
        """
        reset cube + reset robot。

        注意：
            這個函式不會呼叫 super()._reset_idx(env_ids)。
            super()._reset_idx(env_ids) 應該在 BaseUR5Env / TaskEnv 裡呼叫。
        """

        env_ids = self._normalize_env_ids(env_ids)

        cube_state = self.reset_cube(
            env_ids,
            randomize_xy=randomize_cube_xy,
            xy_range=cube_xy_range,
        )

        if robot_above_cube:
            joint_pos, joint_vel = self.reset_robot_above_cube(
                env_ids,
                height_above_cube=height_above_cube,
            )
        else:
            joint_pos, joint_vel = self.reset_robot_to_default(env_ids)

        return {
            "env_ids": env_ids,
            "cube_state": cube_state,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def reset_cube(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        randomize_xy: bool = True,
        xy_range: float = 0.05,
    ) -> torch.Tensor:
        """
        reset 方塊位置。

        Args:
            env_ids:
                要 reset 哪些 environments。

            randomize_xy:
                是否在 x/y 方向加入隨機偏移。

            xy_range:
                x/y 隨機範圍。
                例如 0.05 代表 [-0.05, 0.05] m。
        """

        self._check_robot_and_cube_exist()

        env_ids = self._normalize_env_ids(env_ids)
        num_reset = len(env_ids)

        default_cube_state = self.cube.data.default_root_state[env_ids].clone()

        # 加上每個 vectorized env 的原點
        default_cube_state[:, :3] += self.scene.env_origins[env_ids]

        if randomize_xy:
            default_cube_state[:, 0] += (
                torch.rand(num_reset, device=self.device) * 2.0 - 1.0
            ) * xy_range
            default_cube_state[:, 1] += (
                torch.rand(num_reset, device=self.device) * 2.0 - 1.0
            ) * xy_range

        self.cube.write_root_state_to_sim(
            default_cube_state,
            env_ids=env_ids,
        )
        self.cube.reset(env_ids)

        return default_cube_state

    def reset_robot_to_default(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        gripper_open_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        reset robot 到 cfg 裡的 default joint pose。
        """

        self._ensure_handles()

        env_ids = self._normalize_env_ids(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self.arm_dof_count :] = gripper_open_value

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )
        self.robot.reset(env_ids)

        return joint_pos, joint_vel

    def reset_robot_above_cube(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        height_above_cube: float = 0.25,
        gripper_open_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        reset robot，使 UR5 末端大致位在 cube 上方。

        如果 Lula IK 成功：
            使用 IK 算出 UR5 前 6 軸角度。

        如果 Lula IK 失敗：
            使用安全 fallback 姿態。
        """

        self._ensure_handles()

        env_ids = self._normalize_env_ids(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.lula_ik is not None:
            target_positions = self.cube.data.root_pos_w[env_ids].detach().cpu().numpy()

            for local_i, _env_id in enumerate(env_ids):
                target_pos = target_positions[local_i].copy()
                target_pos[2] += height_above_cube

                success = False

                try:
                    ik_result, success = self.lula_ik.compute_inverse_kinematics(
                        target_position=target_pos,
                        target_orientation=self.target_quat,
                    )
                except Exception as err:
                    print(f"!! Lula IK 計算失敗，改用 fallback posture：{err}")
                    success = False

                if success:
                    angles = torch.tensor(
                        ik_result.joint_positions,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    joint_pos[local_i, : self.arm_dof_count] = angles[
                        : self.arm_dof_count
                    ]
                else:
                    joint_pos[local_i, : self.arm_dof_count] = (
                        self.get_fallback_arm_joint_pos()
                    )
        else:
            joint_pos[:, : self.arm_dof_count] = self.get_fallback_arm_joint_pos()

        # 夾爪打開
        joint_pos[:, self.arm_dof_count :] = gripper_open_value

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )
        self.robot.reset(env_ids)

        return joint_pos, joint_vel

    # ============================================================
    # 6. Utility for controller
    # ============================================================

    def get_current_control_joint_pos(self) -> torch.Tensor:
        """
        給 controller 使用。
        回傳目前被控制的所有 joint position。
        """

        return self.get_joint_pos(controlled_dofs_only=True)

    def get_current_control_joint_vel(self) -> torch.Tensor:
        """
        給 controller 使用。
        回傳目前被控制的所有 joint velocity。
        """

        return self.get_joint_vel(controlled_dofs_only=True)

    def get_fallback_arm_joint_pos(self) -> torch.Tensor:
        """
        IK 失敗時使用的安全姿態。

        對應你原本程式中的：
            [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        """

        return torch.tensor(
            [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )

    # ============================================================
    # 7. Lula IK initialization
    # ============================================================

    def _init_lula_ik(self) -> None:
        """
        初始化 Lula IK。

        若找不到 UR5 / UR5e 的 description 或 URDF，
        self.lula_ik 會保持 None，reset 時自動使用 fallback posture。
        """

        try:
            import numpy as np
            from omni.isaac.core.utils.extensions import (
                enable_extension,
                get_extension_path_from_name,
            )
            from omni.isaac.core.utils.rotations import euler_angles_to_quat
            from omni.isaac.motion_generation.lula.kinematics import (
                LulaKinematicsSolver,
            )

            enable_extension("omni.isaac.motion_generation")
            enable_extension("omni.isaac.universal_robots")

            ext_path = get_extension_path_from_name("omni.isaac.universal_robots")

            if not ext_path:
                ext_path = get_extension_path_from_name("isaacsim.robot.manipulators")

            if not ext_path:
                print("!! 警告：找不到 UR robot extension path，Lula IK 關閉 !!")
                self.lula_ik = None
                return

            desc_path = (
                self._find_file("ur5_robot_description.yaml", ext_path)
                or self._find_file("ur5e_robot_description.yaml", ext_path)
            )
            urdf_path = (
                self._find_file("ur5.urdf", ext_path)
                or self._find_file("ur5e.urdf", ext_path)
            )

            if desc_path and urdf_path:
                self.lula_ik = LulaKinematicsSolver(
                    robot_description_path=desc_path,
                    urdf_path=urdf_path,
                )

                # 讓夾爪朝下
                self.target_quat = euler_angles_to_quat(
                    np.array([0.0, np.pi, 0.0])
                )

                print(">>> 成功載入 Lula IK 引擎")
                print(f">>> robot_description: {desc_path}")
                print(f">>> urdf: {urdf_path}")
            else:
                print("!! 警告：找不到 UR5 / UR5e 的 Lula 設定檔，Lula IK 關閉 !!")
                self.lula_ik = None

        except Exception as err:
            print(f"!! 載入 Lula IK 引擎失敗，Lula IK 關閉：{err}")
            self.lula_ik = None

    # ============================================================
    # 8. Internal helpers
    # ============================================================

    def _ensure_handles(self) -> None:
        """
        確認 robot/cube 和常用 index 都已經初始化。
        """

        self._check_robot_and_cube_exist()

        if not self._handles_initialized:
            self.initialize_handles()

    def _check_robot_and_cube_exist(self) -> None:
        if self.robot is None:
            raise RuntimeError(
                "SimulationBackend.robot 尚未建立。"
                "請先在 _setup_scene() 中呼叫 backend.setup_scene()。"
            )

        if self.cube is None:
            raise RuntimeError(
                "SimulationBackend.cube 尚未建立。"
                "請先在 _setup_scene() 中呼叫 backend.setup_scene()。"
            )

    def _normalize_env_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        """
        將 env_ids 統一轉成 torch.LongTensor。
        """

        self._check_robot_and_cube_exist()

        if env_ids is None:
            all_indices = getattr(self.robot, "_ALL_INDICES", None)

            if all_indices is not None:
                return all_indices.to(device=self.device, dtype=torch.long)

            num_envs = getattr(self.scene, "num_envs", None)

            if num_envs is None:
                raise RuntimeError(
                    "無法推定 num_envs。請明確傳入 env_ids，"
                    "或確認 scene.num_envs 是否存在。"
                )

            return torch.arange(
                num_envs,
                device=self.device,
                dtype=torch.long,
            )

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)

        return torch.tensor(
            list(env_ids),
            device=self.device,
            dtype=torch.long,
        )

    def _device_is_cpu(self) -> bool:
        return str(self.device).startswith("cpu")

    @staticmethod
    def _find_file(filename: str, search_path: str) -> str | None:
        for root, _dirs, files in os.walk(search_path):
            if filename in files:
                return os.path.normpath(os.path.join(root, filename))
        return None

    @staticmethod
    def _safe_get_tensor(
        data_obj: Any,
        attr_name: str,
        index: int | None = None,
    ) -> torch.Tensor | None:
        """
        安全讀取 Isaac Lab data 欄位。

        有些版本或物件不一定有 root_quat_w / body_quat_w 等欄位，
        因此這裡用 getattr 避免直接噴錯。
        """

        value = getattr(data_obj, attr_name, None)

        if value is None:
            return None

        if index is not None:
            return value[:, index]

        return value