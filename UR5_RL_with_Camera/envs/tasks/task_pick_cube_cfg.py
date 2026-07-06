# envs/tasks/task_pick_cube_cfg.py

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

from ..base.base_UR5_env_cfg import BaseUR5EnvCfg


# ============================================================
# 1. Pick Cube Task Object Config
# ============================================================
CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Cube",

    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),

        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.1,
        ),

        collision_props=sim_utils.CollisionPropertiesCfg(),

        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0),
        ),
    ),

    init_state=RigidObjectCfg.InitialStateCfg(
        # 方塊中心在 z = 0.025，代表 5 cm 方塊剛好放在地板上
        pos=(0.5, 0.0, 0.025),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)


# ============================================================
# 2. Pick Cube Task Config
# ============================================================

@configclass
class TaskPickCubeEnvCfg(BaseUR5EnvCfg):

    # ★ 攝影機設定
    """"
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 1.2),      # 攝影機位置（相對 env origin）
            rot=(0.7071, 0.0, 0.7071, 0.0),  # 朝下看 (wxyz)
            convention="opengl",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        width=84,    # ★ 解析度，建議從小開始（84x84 or 64x64）
        height=84,
    )
    """
    # 上方固定相機：看整個工作空間
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/TopCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 1.2),
            rot=(0.7071, 0.0, 0.7071, 0.0),
            convention="opengl",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        width=84,
        height=84,
    )   

    # 末端相機：接在 end-effector 附近
    ee_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        
        prim_path="/World/envs/env_.*/Robot/dg3f/l_dg_mount/EECamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.08),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 3.0),
        ),
        width=84,
        height=84,
    )
    # ------------------------------------------------------------
    # Task object
    # ------------------------------------------------------------
    cube_cfg: RigidObjectCfg = CUBE_CFG.replace(
        prim_path="/World/envs/env_.*/Cube"
    )

    # ------------------------------------------------------------
    # Env length
    # ------------------------------------------------------------
    episode_length_s = 5.0

    # ------------------------------------------------------------
    # Action / observation space
    # ------------------------------------------------------------

    action_space = 7
    observation_space = 167
    state_space = 0

    # ------------------------------------------------------------
    # Reset setting
    # ------------------------------------------------------------
    reset_robot_above_cube = True
    height_above_cube = 0.25

    randomize_cube_xy = True
    cube_xy_range = 0.05

    # ------------------------------------------------------------
    # Reward parameters
    # ------------------------------------------------------------
    # 方塊放在地上時，中心高度約為 0.025 m
    cube_rest_height = 0.025

    # 夾爪收合獎勵：
    # reward_grasp = sum(gripper_joint_pos) * grasp_reward_weight
    grasp_reward_weight = 5.0

    # 抬起獎勵：
    # lift_height = cube_z - cube_rest_height
    # 若 lift_height > lift_reward_threshold，給予 lift reward
    lift_reward_threshold = 0.005
    lift_reward_weight = 3000.0

    # 壓入地板懲罰：
    # 若 cube_z < cube_smash_height，代表方塊可能被壓進地板
    cube_smash_height = 0.024
    smash_penalty_value = 10.0

    # reward 最低限制
    reward_min_value = 0.0

    # ------------------------------------------------------------
    # Done / termination parameters
    # ------------------------------------------------------------
    # 方塊如果被推離 env origin 太遠，提早結束
    cube_push_done_dist = 0.5

    # 方塊若掉到地板以下，提早結束
    cube_fall_done_height = 0.0

    # 是否在成功抬起後直接結束 episode
    done_on_success = False

    # 若 done_on_success = True，
    # lift_height 超過此高度就視為成功
    success_lift_height = 0.08

    # ------------------------------------------------------------
    # Debug / logging
    # ------------------------------------------------------------
    debug_print = False

    # 每隔幾個 simulation step 印一次 reward 資訊
    reward_print_interval = 60