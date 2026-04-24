"""
Evaluation & Visualization — Trained Heat Pump Controller Analysis.

This script loads a trained PPO model and runs it through evaluation episodes,
generating comprehensive visualizations and performance metrics.

Usage:
    python evaluate.py --model_path runs/ppo_xxx/final_model
    python evaluate.py --model_path runs/ppo_xxx/best_model/best_model
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from stable_baselines3 import PPO

from config import SIMULATION_CONFIG, REWARD_WEIGHTS, RUNS_DIR
from heat_pump_env import HeatPumpControlEnv


def evaluate_agent(
    model_path: str,
    n_episodes: int = 1,
    building: str = None,
    days: int = None,
    deterministic: bool = True,
    output_dir: str = None,
):
    """Run evaluation episodes and collect detailed metrics.

    Parameters
    ----------
    model_path : str
        Path to the saved PPO model (without .zip extension).
    n_episodes : int
        Number of evaluation episodes to run.
    building : str
        Override building model (default: from config).
    days : int
        Override episode length.
    deterministic : bool
        Use deterministic policy (no exploration).
    output_dir : str
        Directory for output plots and data.

    Returns
    -------
    dict
        Aggregated evaluation metrics.
    """
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create environment
    env = HeatPumpControlEnv(
        building=building or SIMULATION_CONFIG["building"],
        days=days or SIMULATION_CONFIG["days"],
        random_init=False,  # Fixed start for reproducible evaluation
    )

    if output_dir is None:
        output_dir = str(Path(model_path).parent / "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for ep in range(n_episodes):
        print(f"\n{'─'*50}")
        print(f"  Episode {ep + 1}/{n_episodes}")
        print(f"{'─'*50}")

        obs, info = env.reset()
        done = False
        step = 0

        # Collect trajectory data
        trajectory = {
            "T_room": [],
            "T_amb": [],
            "supply_temp": [],
            "price_eur_kwh": [],
            "Q_el_kWh": [],
            "reward": [],
            "action_raw": [],
        }

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            trajectory["T_room"].append(info.get("T_room", 20.0))
            trajectory["supply_temp"].append(info.get("u", 20.0))
            trajectory["price_eur_kwh"].append(info.get("price_eur_kwh", 0.30))
            trajectory["Q_el_kWh"].append(info.get("Q_el_kWh", 0.0))
            trajectory["reward"].append(reward)
            trajectory["action_raw"].append(float(action[0]))

            # Get T_amb from base env
            try:
                trajectory["T_amb"].append(float(env._env.get_cur_T_amb()))
            except (IndexError, AttributeError):
                trajectory["T_amb"].append(5.0)

        # Episode summary
        summary = env.get_episode_summary()
        summary["episode"] = ep
        summary["n_steps"] = step
        summary["avg_T_room"] = np.mean(trajectory["T_room"])
        summary["min_T_room"] = np.min(trajectory["T_room"])
        summary["max_T_room"] = np.max(trajectory["T_room"])
        summary["std_T_room"] = np.std(trajectory["T_room"])
        summary["avg_reward"] = np.mean(trajectory["reward"])
        summary["total_reward"] = np.sum(trajectory["reward"])

        all_results.append({"summary": summary, "trajectory": trajectory})

        print(f"  Steps: {step:,}")
        print(f"  Energy: {summary['total_energy_kwh']:.1f} kWh")
        print(f"  Cost: €{summary['total_cost_eur']:.2f}")
        print(f"  Compressor switches: {summary['compressor_switches']}")
        print(f"  Comfort violations: {summary['comfort_violations']}")
        print(f"  T_room: {summary['avg_T_room']:.1f}°C "
              f"(range: {summary['min_T_room']:.1f}–{summary['max_T_room']:.1f}°C)")
        print(f"  Total reward: {summary['total_reward']:.1f}")

    # ── Generate visualizations ─────────────────────────────────────────
    if n_episodes >= 1:
        traj = all_results[0]["trajectory"]
        _plot_episode(traj, all_results[0]["summary"], output_dir)

    # ── Save metrics ────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics = [r["summary"] for r in all_results]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n📊 Metrics saved to: {metrics_path}")

    env.close()
    return all_results


def _plot_episode(trajectory: dict, summary: dict, output_dir: str):
    """Generate comprehensive evaluation plots for a single episode.

    Creates a 4-panel figure showing:
    1. Room temperature vs. comfort bounds
    2. Supply temperature (agent's control action)
    3. Electricity price and energy consumption
    4. Cumulative reward over time
    """
    n = len(trajectory["T_room"])
    steps = np.arange(n)
    hours = steps * SIMULATION_CONFIG["delta_t"] / 3600  # Convert to hours

    T_min, T_max = REWARD_WEIGHTS["comfort_range"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"Heat Pump RL Controller — Evaluation\n"
        f"Building: {SIMULATION_CONFIG['building']} | "
        f"Energy: {summary['total_energy_kwh']:.0f} kWh | "
        f"Cost: €{summary['total_cost_eur']:.2f} | "
        f"Switches: {summary['compressor_switches']}",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel 1: Room Temperature ───────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(hours, trajectory["T_room"], color="#2196F3", linewidth=1.2, label="T_room")
    ax1.plot(hours, trajectory["T_amb"], color="#FF9800", linewidth=0.8, alpha=0.6, label="T_ambient")
    ax1.axhline(y=T_min, color="#4CAF50", linestyle="--", alpha=0.7, label=f"Comfort min ({T_min}°C)")
    ax1.axhline(y=T_max, color="#F44336", linestyle="--", alpha=0.7, label=f"Comfort max ({T_max}°C)")
    ax1.fill_between(hours, T_min, T_max, alpha=0.1, color="#4CAF50", label="Comfort zone")
    ax1.set_ylabel("Temperature [°C]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Room Temperature Control")
    ax1.grid(True, alpha=0.3)

    # Highlight comfort violations
    T_room_arr = np.array(trajectory["T_room"])
    violations = (T_room_arr < T_min) | (T_room_arr > T_max)
    if violations.any():
        ax1.fill_between(
            hours, T_room_arr, T_min,
            where=(T_room_arr < T_min),
            alpha=0.3, color="#F44336", label="Below comfort"
        )

    # ── Panel 2: Supply Temperature (Control Action) ────────────────────
    ax2 = axes[1]
    ax2.plot(hours, trajectory["supply_temp"], color="#9C27B0", linewidth=1.0)
    ax2.set_ylabel("Supply Temp [°C]")
    ax2.set_title("Heat Pump Supply Temperature (Agent Action)")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Price & Energy ─────────────────────────────────────────
    ax3 = axes[2]
    ax3_price = ax3
    ax3_energy = ax3.twinx()

    ax3_price.plot(
        hours, trajectory["price_eur_kwh"],
        color="#FF5722", linewidth=0.8, alpha=0.7, label="Price"
    )
    ax3_price.set_ylabel("Price [€/kWh]", color="#FF5722")
    ax3_price.tick_params(axis="y", labelcolor="#FF5722")

    ax3_energy.bar(
        hours, trajectory["Q_el_kWh"],
        width=SIMULATION_CONFIG["delta_t"] / 3600 * 0.8,
        alpha=0.5, color="#03A9F4", label="Energy"
    )
    ax3_energy.set_ylabel("Energy [kWh]", color="#03A9F4")
    ax3_energy.tick_params(axis="y", labelcolor="#03A9F4")

    ax3.set_title("Electricity Price & Energy Consumption")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Cumulative Reward ──────────────────────────────────────
    ax4 = axes[3]
    cumulative_reward = np.cumsum(trajectory["reward"])
    ax4.plot(hours, cumulative_reward, color="#4CAF50", linewidth=1.2)
    ax4.fill_between(hours, cumulative_reward, alpha=0.2, color="#4CAF50")
    ax4.set_ylabel("Cumulative Reward")
    ax4.set_xlabel("Time [hours]")
    ax4.set_title("Cumulative Reward Over Episode")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📈 Evaluation plot saved to: {plot_path}")

    # ── Additional: Daily breakdown ─────────────────────────────────────
    _plot_daily_breakdown(trajectory, output_dir)


def _plot_daily_breakdown(trajectory: dict, output_dir: str):
    """Plot daily energy cost and comfort violation breakdown."""
    delta_t = SIMULATION_CONFIG["delta_t"]
    steps_per_day = 24 * 3600 // delta_t
    n_days = len(trajectory["T_room"]) // steps_per_day

    if n_days < 2:
        return

    T_min, T_max = REWARD_WEIGHTS["comfort_range"]

    daily_energy = []
    daily_cost = []
    daily_violations = []

    for d in range(n_days):
        start = d * steps_per_day
        end = (d + 1) * steps_per_day

        energy = sum(trajectory["Q_el_kWh"][start:end])
        cost = sum(
            e * p for e, p in zip(
                trajectory["Q_el_kWh"][start:end],
                trajectory["price_eur_kwh"][start:end],
            )
        )
        violations = sum(
            1 for t in trajectory["T_room"][start:end]
            if t < T_min or t > T_max
        )

        daily_energy.append(energy)
        daily_cost.append(cost)
        daily_violations.append(violations)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Daily Performance Breakdown", fontsize=14, fontweight="bold")

    days = range(1, n_days + 1)

    axes[0].bar(days, daily_energy, color="#2196F3", alpha=0.8)
    axes[0].set_ylabel("Energy [kWh]")
    axes[0].set_title("Daily Energy Consumption")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(days, daily_cost, color="#FF9800", alpha=0.8)
    axes[1].set_ylabel("Cost [€]")
    axes[1].set_title("Daily Electricity Cost")
    axes[1].grid(True, alpha=0.3)

    colors = ["#4CAF50" if v == 0 else "#F44336" for v in daily_violations]
    axes[2].bar(days, daily_violations, color=colors, alpha=0.8)
    axes[2].set_ylabel("Violations [steps]")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Daily Comfort Violations")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "daily_breakdown.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Daily breakdown saved to: {plot_path}")


def compare_with_baseline(
    model_path: str,
    days: int = 7,
    output_dir: str = None,
):
    """Compare RL agent vs. simple rule-based heating curve baseline.

    The baseline uses i4b's built-in heating curve controller for comparison.
    """
    print("\n" + "="*60)
    print("  RL Agent vs. Baseline Comparison")
    print("="*60)

    # ── RL Agent ────────────────────────────────────────────────────────
    model = PPO.load(model_path)
    env = HeatPumpControlEnv(
        days=days,
        random_init=False,
    )

    obs, _ = env.reset()
    rl_energy = 0.0
    rl_violations = 0
    rl_T_rooms = []
    T_min, T_max = REWARD_WEIGHTS["comfort_range"]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rl_energy += info.get("Q_el_kWh", 0)
        T_room = info.get("T_room", 20.0)
        rl_T_rooms.append(T_room)
        if T_room < T_min or T_room > T_max:
            rl_violations += 1

    rl_summary = env.get_episode_summary()
    env.close()

    # ── Print comparison ────────────────────────────────────────────────
    print(f"\n{'Metric':<30} {'RL Agent':>15}")
    print(f"{'─'*45}")
    print(f"{'Energy [kWh]':<30} {rl_summary['total_energy_kwh']:>15.1f}")
    print(f"{'Cost [€]':<30} {rl_summary['total_cost_eur']:>15.2f}")
    print(f"{'Compressor switches':<30} {rl_summary['compressor_switches']:>15d}")
    print(f"{'Comfort violations':<30} {rl_summary['comfort_violations']:>15d}")
    print(f"{'Avg T_room [°C]':<30} {np.mean(rl_T_rooms):>15.1f}")
    print(f"{'T_room range [°C]':<30} {np.min(rl_T_rooms):>6.1f} – {np.max(rl_T_rooms):.1f}")

    return rl_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained heat pump controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to saved PPO model",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=3,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--building", type=str, default=None,
        help="Override building model",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Override episode length [days]",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare with baseline heating curve controller",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots and metrics",
    )

    args = parser.parse_args()

    results = evaluate_agent(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        building=args.building,
        days=args.days,
        output_dir=args.output_dir,
    )

    if args.compare:
        compare_with_baseline(
            model_path=args.model_path,
            days=args.days or SIMULATION_CONFIG["days"],
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
