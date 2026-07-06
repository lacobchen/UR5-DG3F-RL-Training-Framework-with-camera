# envs/tasks/agents/rsl_rl_ppo_cfg.py

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PickCubeRslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
    seed = 42
    device = "cuda:0"

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100

    experiment_name = "ur5_pick_cube"
    run_name = ""
    logger = "tensorboard"

    empirical_normalization = False
    clip_actions = 1.0
    check_for_nan = True

    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # ------------------------------------------------------------
    # Observation groups
    # ------------------------------------------------------------
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # ------------------------------------------------------------
    # Actor model
    # ------------------------------------------------------------
    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # ------------------------------------------------------------
    # Critic model
    # ------------------------------------------------------------
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )

    # ------------------------------------------------------------
    # PPO algorithm
    # ------------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )

    def to_dict(self) -> dict:
        cfg = super().to_dict()


        deprecated_model_keys = [
            "stochastic",
            "init_noise_std",
            "noise_std_type",
            "state_dependent_std",
        ]

        for model_key in ["actor", "critic"]:
            if model_key in cfg and isinstance(cfg[model_key], dict):
                for key in deprecated_model_keys:
                    cfg[model_key].pop(key, None)

        # 保留 policy 相容欄位，避免部分 runner 仍讀 policy。
        cfg.setdefault(
            "policy",
            {
                "class_name": "ActorCritic",
                "init_noise_std": 1.0,
                "actor_hidden_dims": [256, 128, 64],
                "critic_hidden_dims": [256, 128, 64],
                "activation": "elu",
            },
        )

        return cfg