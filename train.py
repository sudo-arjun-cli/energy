"""
Training Script — PPO Agent for Heat Pump Control.

This script trains a PPO agent on the HeatPumpControlEnv using
Stable-Baselines3. It supports:
- Multiple building configurations from i4b's TABULA database
- Configurable reward weights and hyperparameters
- TensorBoard logging for training monitoring
- Checkpoint saving and best model tracking
- Evaluation callbacks during training

Usage:
    python train.py                          # Default config
    python train.py --building sfh_2016_now_0_soc
    python train.py --total_timesteps 1000000 --days 60
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from config import (
    SIMULATION_CONFIG,
    PPO_CONFIG,
    EVAL_CONFIG,
    REWARD_WEIGHTS,
    RUNS_DIR,
    MODELS_DIR,
)
from heat_pump_env import HeatPumpControlEnv


def make_env(args, rank=0):
    """Factory function to create a monitored environment instance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    rank : int
        Environment index for vectorized envs.

    Returns
    -------
    callable
        Thunk that returns a Monitor-wrapped HeatPumpControlEnv.
    """
    def _init():
        env = HeatPumpControlEnv(
            building=args.building,
            hp_model=args.hp_model,
            method=args.method,
            mdot_hp=args.mdot_hp,
            delta_t=args.delta_t,
            days=args.days,
            forecast_steps=args.forecast_steps,
            random_init=args.random_init,
            goal_based=args.goal_based,
            goal_temp_range=(args.goal_temp_min, args.goal_temp_max),
            temp_deviation_weight=args.temp_deviation_weight,
        )
        log_dir = os.path.join(args.logdir, f"env_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        return env
    return _init


def parse_args():
    """Parse command-line arguments with defaults from config."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for heat pump control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Environment parameters ──────────────────────────────────────────
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument(
        "--building", type=str,
        default=SIMULATION_CONFIG["building"],
        help="Building model from i4b TABULA database",
    )
    env_group.add_argument(
        "--hp_model", type=str,
        default=SIMULATION_CONFIG["hp_model"],
        help="Heat pump model class name",
    )
    env_group.add_argument(
        "--method", type=str,
        default=SIMULATION_CONFIG["method"],
        help="RC-network model method (e.g., 4R3C, 7R5C)",
    )
    env_group.add_argument(
        "--mdot_hp", type=float,
        default=SIMULATION_CONFIG["mdot_hp"],
        help="Heat pump mass flow rate [kg/s]",
    )
    env_group.add_argument(
        "--delta_t", type=int,
        default=SIMULATION_CONFIG["delta_t"],
        help="Simulation timestep [seconds]",
    )
    env_group.add_argument(
        "--days", type=int,
        default=SIMULATION_CONFIG["days"],
        help="Episode length [days]",
    )
    env_group.add_argument(
        "--forecast_steps", type=int,
        default=SIMULATION_CONFIG["forecast_steps"],
        help="Number of weather forecast steps (0=disabled)",
    )
    env_group.add_argument(
        "--random_init", action="store_true",
        default=SIMULATION_CONFIG["random_init"],
        help="Randomize initial state and start position",
    )

    # ── Goal-based learning ─────────────────────────────────────────────
    goal_group = parser.add_argument_group("Goal-Based Learning")
    goal_group.add_argument(
        "--goal_based", action="store_true",
        default=SIMULATION_CONFIG["goal_based"],
        help="Enable goal-based temperature targeting",
    )
    goal_group.add_argument(
        "--goal_temp_min", type=float,
        default=SIMULATION_CONFIG["goal_temp_range"][0],
        help="Minimum goal temperature [°C]",
    )
    goal_group.add_argument(
        "--goal_temp_max", type=float,
        default=SIMULATION_CONFIG["goal_temp_range"][1],
        help="Maximum goal temperature [°C]",
    )
    goal_group.add_argument(
        "--temp_deviation_weight", type=float,
        default=SIMULATION_CONFIG["temp_deviation_weight"],
        help="Weight for temperature deviation in base reward",
    )

    # ── Training parameters ─────────────────────────────────────────────
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--total_timesteps", type=int,
        default=PPO_CONFIG["total_timesteps"],
        help="Total training timesteps",
    )
    train_group.add_argument(
        "--learning_rate", type=float,
        default=PPO_CONFIG["learning_rate"],
        help="PPO learning rate",
    )
    train_group.add_argument(
        "--n_steps", type=int,
        default=PPO_CONFIG["n_steps"],
        help="Steps per rollout collection",
    )
    train_group.add_argument(
        "--batch_size", type=int,
        default=PPO_CONFIG["batch_size"],
        help="Minibatch size for optimization",
    )
    train_group.add_argument(
        "--n_epochs", type=int,
        default=PPO_CONFIG["n_epochs"],
        help="Number of PPO optimization epochs",
    )
    train_group.add_argument(
        "--gamma", type=float,
        default=PPO_CONFIG["gamma"],
        help="Discount factor",
    )
    train_group.add_argument(
        "--seed", type=int,
        default=PPO_CONFIG["seed"],
        help="Random seed",
    )
    train_group.add_argument(
        "--device", type=str,
        default=PPO_CONFIG["device"],
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    train_group.add_argument(
        "--n_envs", type=int,
        default=1,
        help="Number of parallel environments",
    )

    # ── Output ──────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--logdir", type=str,
        default=None,
        help="Log directory (auto-generated if not specified)",
    )
    out_group.add_argument(
        "--run_name", type=str,
        default=None,
        help="Run name for TensorBoard (auto-generated if not specified)",
    )

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    # ── Setup run directory ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_{args.building}_{timestamp}"

    if args.logdir is None:
        args.logdir = str(RUNS_DIR / run_name)

    os.makedirs(args.logdir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.logdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Configuration saved to: {config_path}")

    # ── Set random seeds ────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ── Detect device ───────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n🖥️  Using device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Create environments ─────────────────────────────────────────────
    print(f"\n🏗️  Creating {args.n_envs} environment(s)...")

    if args.n_envs == 1:
        env = DummyVecEnv([make_env(args, rank=0)])
    else:
        env = SubprocVecEnv([make_env(args, rank=i) for i in range(args.n_envs)])

    # Create evaluation environment (separate instance)
    eval_env = DummyVecEnv([make_env(args, rank=99)])

    # ── Create PPO model ────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(
            pi=PPO_CONFIG["policy_kwargs"]["net_arch"]["pi"],
            vf=PPO_CONFIG["policy_kwargs"]["net_arch"]["vf"],
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        vf_coef=PPO_CONFIG["vf_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
    )

    print(f"\n📊 Model summary:")
    print(f"   Policy: MlpPolicy")
    print(f"   Network: pi={policy_kwargs['net_arch']['pi']}, vf={policy_kwargs['net_arch']['vf']}")
    print(f"   Total params: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.policy.parameters() if p.requires_grad):,}")

    # ── Setup callbacks ─────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best_model"),
        log_path=os.path.join(args.logdir, "eval_logs"),
        eval_freq=EVAL_CONFIG["eval_freq"],
        n_eval_episodes=EVAL_CONFIG["n_eval_episodes"],
        deterministic=EVAL_CONFIG["deterministic"],
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(args.logdir, "checkpoints"),
        name_prefix="ppo_heatpump",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── Train ───────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {args.total_timesteps:,} timesteps...")
    print(f"   Building: {args.building}")
    print(f"   Episode length: {args.days} days ({args.days * 24 * 3600 // args.delta_t:,} steps)")
    print(f"   Timestep: {args.delta_t}s ({args.delta_t / 60:.0f} min)")
    print(f"   TensorBoard: tensorboard --logdir {args.logdir}")
    print()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
    )

    # ── Save final model ────────────────────────────────────────────────
    final_path = os.path.join(args.logdir, "final_model")
    model.save(final_path)
    print(f"\n✅ Training complete!")
    print(f"   Final model: {final_path}")
    print(f"   Best model:  {os.path.join(args.logdir, 'best_model')}")
    print(f"   Logs:        {args.logdir}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
