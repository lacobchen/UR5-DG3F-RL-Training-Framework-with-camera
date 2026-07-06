# UR5 + DG3F Vision-Based Reinforcement Learning Project

本專案是在 **NVIDIA Isaac Sim /Isaac Lab** 中建立 UR5 六軸機械手臂與 DG3F 夾爪的抓取方塊任務，並使用 **Isaac Lab 官方 RSL-RL `train.py`** 執行 PPO 強化學習訓練

## 1.Project Structure

UR5_RL_with_Camera/
├── envs/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_UR5_env.py         # Base RL environment (DirectRLEnv)
│   │   └── base_UR5_env_cfg.py     # Base config (robot, actuators, DiffIK)
│   ├── tasks/
│   │   ├── __init__.py             # Gym registration for 我的任務ID
│   │   ├── task_pick_cube_env.py   # Task env (observations, rewards, dones)
│   │   ├── task_pick_cube_cfg.py   # Task config (cameras, cube, reward weights)
│   │   └── agents/
│   │       ├── __init__.py
│   │       └── rsl_rl_ppo_cfg.py   # PPO (actor/critic/algorithm)
│   ├── backend/
│   │   ├── __init__.py
│   │   └── simulation_backend.py   # 模擬環境及溝通(Isaac Lab scene, state, Jacobian, reset)
│   └── vision_encoder.py           # CNN depth encoder (DepthEncoder)
└── run_official_train.bat          # 訓練啟動腳本


RL Policy (PPO)
     │
     ▼
TaskPickCubeEnv._get_observations()
     │  ┌──────────────────────────────────────────────────┐
     │  │  top_depth (84×84) ─► DepthEncoder ─► 64-dim feat│
     │  │  ee_depth  (84×84) ─► DepthEncoder ─► 64-dim feat│
     │  │  ee_pos (3) + joint_pos (18) + joint_vel (18)    │
     │  └────────────────── obs: 167-dim ──────────────────┘
     │
     ▼
TaskPickCubeEnv._apply_action()
     │  action[0:3]  → Cartesian delta (DiffIK → UR5 6-DOF)
     │  action[6]    → Gripper open/close 
     │
     ▼
SimulationBackend (Isaac Lab / PhysX)

## 2.各檔案功能
### 2.1 run_official_train.bat

啟動檔，用來設定 Windows / Isaac Sim / Isaac Lab 的環境變數，並呼叫官方 RSL-RL 訓練腳本。

1. 啟動 conda isaaclab 環境
2. 設定 PROJECT_ROOT
3. 設定 ISAACLAB_ROOT
4. 設定 ISAACSIM_PATH
5. 呼叫 Isaac Sim 的 setup_python_env.bat
6. 補上 PATH / PYTHONPATH
7. 執行 Isaac Lab 官方 train.py
8. 加上 --enable_cameras

目前指令為(可自行調整環境個數、迭代次數、是否可視化、是否啟用攝影機等等)
```bat
call "%ISAACLAB_ROOT%\isaaclab.bat" -p "%ISAACLAB_ROOT%\scripts\reinforcement_learning\rsl_rl\train.py" ^
  --task UR5-PickCube-v0 ^
  --num_envs 1 ^
  --max_iterations 1 ^
  --headless ^
  --enable_cameras
```

### 2.2 `envs/tasks/__init__.py`

此檔案負責將自訂環境註冊成 Gymnasium task，讓 Isaac Lab 官方 `train.py` 可以透過 `--task` 找到環境。

目前 task id：UR5-PickCube-v0

```python
gym.register(
    id="UR5-PickCube-v0",
    entry_point="envs.tasks.task_pick_cube_env:TaskPickCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.tasks.task_pick_cube_cfg:TaskPickCubeEnvCfg",
        "rsl_rl_cfg_entry_point": "envs.tasks.agents.rsl_rl_ppo_cfg:PickCubeRslRlPpoCfg",
    },
)
```

### 2.3 `envs/tasks/task_pick_cube_cfg.py`

此檔案定義抓取方塊任務的環境設定，繼承自 `BaseUR5EnvCfg`。

1. Cube 物件設定
2. Camera 設定
3. Episode 長度
4. Action space / observation space
5. Reset 隨機化設定
6. Reward 參數
7. Done / termination 條件

目前 camera 設定：
```python
camera_cfg: TiledCameraCfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Camera",
    data_types=["rgb", "depth"],
    width=84,
    height=84,
)
```
雖然目前 `data_types` 包含 RGB 與 depth，但 RL policy 實際只使用 depth。

### 2.4 `envs/tasks/task_pick_cube_env.py`

此檔案是實際的 RL task environment，繼承自 `BaseUR5Env`。

1. 建立 DepthEncoder
2. 產生 observation
3. 計算 reward
4. 判斷 done / timeout
5. reset 任務狀態
6. 提供 task info

目前 depth encoder 是 frozen：

```python
for param in self.vision_encoder.parameters():
    param.requires_grad = False
```

因此目前的視覺特徵是：
Depth image → frozen CNN feature → PPO MLP policy

`DepthEncoder` 沒有預訓練，目前是 random frozen feature extractor。訓練流程可跑通，但視覺特徵不一定具有足夠的任務語意。

### 2.5 `envs/vision_encoder.py`

此檔案定義 `DepthEncoder`，是一個小型 CNN，用來將單通道 depth image 壓縮成固定長度 feature vector。

輸入與輸出：
input : [N, 1, 84, 84]
output: [N, 64]

結構為:
Conv(1→32, k=8, s=4) → ReLU
Conv(32→64, k=4, s=2) → ReLU
Conv(64→64, k=3, s=1) → ReLU → Flatten
Linear(3136→256) → ReLU → Linear(256→64)

### 2.6 `envs/base/base_UR5_env_cfg.py`

此檔案定義 UR5 + DG3F 的通用 base environment config。

1. UR5_DG3F USD 路徑
2. Robot articulation config
3. 初始 joint pose
4. actuator 設定
5. simulation dt / render interval
6. scene num_envs / env_spacing
7. robot prim path
8. end-effector body name
9. DiffIK 控制參數
10. gripper 控制參數
11. reset 與 debug 設定

目前 USD 路徑：

```python
UR5_DG3F_USD_PATH = r"D:/UR5_Project/UR5_DG3F/ur5_DG3F_v05.usd"
```
目前 robot 會被掛載到：
```python
prim_path="/World/envs/env_.*/Robot"
```
camera 掛到 end-effector 的 `l_dg_mount` 下":
```python
prim_path="/World/envs/env_0/Robot/dg3f/l_dg_mount"
```

### 2.7 `envs/base/base_UR5_env.py`

此檔案定義 UR5 + DG3F 的共用 RL environment base class。

1. 建立 SimulationBackend
2. 接收 RL action
3. 將 action 轉成 UR5 joint target
4. DiffIK 計算 TCP delta 對應的 delta q
5. gripper open / close 控制
6. reset robot / cube
7. 提供 simulation state 給 task environment

目前 action 到控制目標的流程：

RL action
→ _pre_physics_step()
→ _apply_action()
→ compute_control_targets()
→ compute_diffik_delta_q()
→ backend.apply_settings()
→ Isaac Sim simulation

### 2.8 `envs/backend/simulation_backend.py`

此檔案負責較底層的 Isaac Lab / Isaac Sim scene 操作。

1. 建立 Articulation robot
2. 建立 RigidObject cube
3. 建立 camera sensor
4. 建立 ground plane
5. 建立 light
6. clone vectorized environments
7. 初始化 robot / cube / camera handles
8. 讀取 joint state / cube state / ee state / camera image
9. 提供 Jacobian 給 DiffIK
10. 執行 reset cube / reset robot
11. 將 joint target 寫入 simulation

### 2.9 `envs/tasks/agents/rsl_rl_ppo_cfg.py`

此檔案定義 RSL-RL PPO 訓練設定。

## 3. How to Train
### 3.1 Basic training test

```bat
cd /d D:\UR5_Project\UR5_RL_with_Camera
run_official_train.bat
```

目前 bat 預設：

```bat
--num_envs 1
--max_iterations 1
--headless
--enable_cameras
```
後續訓練可增加env數量，及迭代次數
