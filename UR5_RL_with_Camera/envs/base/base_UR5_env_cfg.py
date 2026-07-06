# envs/base/base_ur5_env_cfg.py

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


UR5_DG3F_USD_PATH = r"D:/UR5_Project/UR5_DG3F/ur5_DG3F_v05.usd"


UR5_DG3F_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"D:/UR5_Project/UR5_DG3F/ur5_DG3F_v05.usd",

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),

        # 安全初始姿態：
        # UR5 手臂立起，DG3F 夾爪張開。
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 0.0,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,

            # 除了 shoulder / elbow / wrist 以外，都視為 DG3F 相關關節
            "(?!shoulder|elbow|wrist).*": 0.0,
        },
    ),

    actuators={
        "ur5_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit=150.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=40.0,
        ),

        "dg3f_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "(?!shoulder|elbow|wrist).*",
            ],
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


# ============================================================
# 2. Base UR5 Env Config
# ============================================================

@configclass
class BaseUR5EnvCfg(DirectRLEnvCfg):
    # Basic RL env setting
    decimation = 2
    episode_length_s = 5.0

    action_space = 18
    observation_space = 42
    state_space = 0

    # -----------------------------------------------------------
    # Simulation setting
    # -----------------------------------------------------------
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------
    # Robot setting
    # ------------------------------------------------------------
    robot_cfg: ArticulationCfg = UR5_DG3F_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ------------------------------------------------------------
    # Backend / robot handle setting
    # ------------------------------------------------------------
    robot_joint_expr = ".*"
    ee_body_name = "l_dg_mount"
    arm_dof_count = 6

    # Isaac Lab / PhysX 在 fixed-base robot 時，
    # Jacobian link index 通常會比 body index 少 1。
    jacobian_body_index_offset = -1

    enable_lula_ik = False

    # ------------------------------------------------------------
    # Cartesian DiffIK action setting
    # ------------------------------------------------------------
    tcp_delta_scale = 0.01

    # 是否使用 damped least-squares pseudo-inverse。
    # False 時使用 torch.linalg.pinv
    diffik_use_damped_pinv = False
    diffik_damping_lambda = 0.05

    # ------------------------------------------------------------
    # Gripper action setting
    # ------------------------------------------------------------
    gripper_action_index = 6

    use_binary_gripper = True

    gripper_open_value = 0.0
    gripper_close_value = 0.8

    # mask 
    # [0,0,0,0,0,0,1,1,1,1,1,1]
    #   DG3F 前 6 個夾爪關節保持 0
    #   後 6 個夾爪關節會跟著 close/open
    gripper_joint_mask = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    # ------------------------------------------------------------
    # Reset setting
    # ------------------------------------------------------------
    reset_robot_above_cube = True
    height_above_cube = 0.25

    randomize_cube_xy = True
    cube_xy_range = 0.05
    # ------------------------------------------------------------
    # Debug setting
    # ------------------------------------------------------------
    debug_print = False