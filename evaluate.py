"""
Evaluation & Visualisation — Hierarchical MPC Heat Pump Controller.

Generates comprehensive plots from MPC simulation results:
    1. Room & Tank Temperature vs. Comfort Bounds and References
    2. Heat Pump Power & Valve Position
    3. Electricity Price & Energy Consumption
    4. COP and Cumulative Cost

Usage:
    python evaluate.py --results_dir runs/mpc_xxx_20240101_120000
    python evaluate.py --results_dir runs/mpc_xxx_20240101_120000 --compare_baseline
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import SIMULATION_CONFIG, MPC_CONFIG, RUNS_DIR


def plot_mpc_results(log: dict, summary: dict, output_dir: str):
    """Generate the 5-panel MPC evaluation figure.

    Parameters
    ----------
    log : dict
        Trajectory log from HeatPumpSimulation.run().
    summary : dict
        Summary statistics.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(log["T_room"])
    dt = SIMULATION_CONFIG["delta_t"]
    hours = np.arange(n) * dt / 3600

    T_min = MPC_CONFIG["room_t_min"]
    T_max = MPC_CONFIG["room_t_max"]

    fig, axes = plt.subplots(5, 1, figsize=(18, 20), sharex=True)
    fig.suptitle(
        f"Hierarchical MPC Heat Pump Controller — Evaluation\n"
        f"Building: {SIMULATION_CONFIG['building']} | "
        f"Energy: {summary['total_energy_kwh']:.0f} kWh | "
        f"Cost: €{summary['total_cost_eur']:.2f} | "
        f"Comfort violations: {summary['comfort_violations']}",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: Temperatures ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(hours, log["T_room"], color="#2196F3", lw=1.5, label="T_room (actual)")
    ax1.plot(hours, log["T_room_ref"], color="#2196F3", lw=0.8, ls="--", alpha=0.6, label="T_room (MPC ref)")
    ax1.plot(hours, log["T_amb"], color="#FF9800", lw=0.8, alpha=0.5, label="T_ambient")
    ax1.axhline(T_min, color="#4CAF50", ls="--", alpha=0.7, label=f"Comfort min ({T_min}°C)")
    ax1.axhline(T_max, color="#F44336", ls="--", alpha=0.7, label=f"Comfort max ({T_max}°C)")
    ax1.fill_between(hours, T_min, T_max, alpha=0.08, color="#4CAF50")
    ax1.set_ylabel("Temperature [°C]")
    ax1.set_title("Room Temperature Control")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Tank Temperature ───────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(hours, log["T_tank"], color="#E91E63", lw=1.5, label="T_tank (actual)")
    ax2.plot(hours, log["T_tank_ref"], color="#E91E63", lw=0.8, ls="--", alpha=0.6, label="T_tank (MPC ref)")
    ax2.set_ylabel("Temperature [°C]")
    ax2.set_title("Buffer Tank Temperature")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Heat Pump Power & Valve ────────────────────────────────
    ax3 = axes[2]
    ax3_hp = ax3
    ax3_valve = ax3.twinx()

    ax3_hp.fill_between(hours, 0, np.array(log["Q_HP_w"]) / 1000,
                         alpha=0.4, color="#9C27B0", label="HP thermal [kW]")
    ax3_hp.set_ylabel("HP Power [kW]", color="#9C27B0")
    ax3_hp.tick_params(axis="y", labelcolor="#9C27B0")

    ax3_valve.plot(hours, log["valve_pos"], color="#00BCD4", lw=0.8, alpha=0.7, label="Valve position")
    ax3_valve.set_ylabel("Valve [0–1]", color="#00BCD4")
    ax3_valve.tick_params(axis="y", labelcolor="#00BCD4")
    ax3_valve.set_ylim(-0.05, 1.05)

    ax3.set_title("Heat Pump Power & Mixing Valve")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Price & Energy ─────────────────────────────────────────
    ax4 = axes[3]
    ax4_price = ax4
    ax4_energy = ax4.twinx()

    ax4_price.plot(hours, log["price_eur_kwh"], color="#FF5722", lw=0.8, alpha=0.7, label="Price")
    ax4_price.set_ylabel("Price [€/kWh]", color="#FF5722")
    ax4_price.tick_params(axis="y", labelcolor="#FF5722")

    q_el_kwh = np.array(log["Q_el_w"]) * dt / 3.6e6
    ax4_energy.bar(hours, q_el_kwh, width=dt / 3600 * 0.8,
                    alpha=0.4, color="#03A9F4", label="Electrical energy")
    ax4_energy.set_ylabel("Energy [kWh]", color="#03A9F4")
    ax4_energy.tick_params(axis="y", labelcolor="#03A9F4")

    ax4.set_title("Electricity Price & Consumption")
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: COP & Cumulative Cost ──────────────────────────────────
    ax5 = axes[4]
    ax5_cop = ax5
    ax5_cost = ax5.twinx()

    ax5_cop.plot(hours, log["cop"], color="#795548", lw=0.8, alpha=0.7, label="COP")
    ax5_cop.set_ylabel("COP", color="#795548")
    ax5_cop.tick_params(axis="y", labelcolor="#795548")
    ax5_cop.set_ylim(0, 8)

    cum_cost = np.cumsum(log["cost_eur"])
    ax5_cost.plot(hours, cum_cost, color="#4CAF50", lw=1.5, label="Cumulative cost")
    ax5_cost.fill_between(hours, cum_cost, alpha=0.15, color="#4CAF50")
    ax5_cost.set_ylabel("Cumulative Cost [€]", color="#4CAF50")
    ax5_cost.tick_params(axis="y", labelcolor="#4CAF50")

    ax5.set_xlabel("Time [hours]")
    ax5.set_title("COP & Cumulative Electricity Cost")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mpc_evaluation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📈 Evaluation plot saved to: {plot_path}")

    # ── Daily breakdown ─────────────────────────────────────────────────
    _plot_daily_breakdown(log, output_dir)


def _plot_daily_breakdown(log: dict, output_dir: str):
    """Plot per-day energy, cost, and tank utilisation."""
    dt = SIMULATION_CONFIG["delta_t"]
    steps_per_day = 24 * 3600 // dt
    n_days = len(log["T_room"]) // steps_per_day

    if n_days < 2:
        return

    daily_energy = []
    daily_cost = []
    daily_avg_tank = []
    daily_avg_cop = []

    for d in range(n_days):
        s, e = d * steps_per_day, (d + 1) * steps_per_day
        q_el = np.array(log["Q_el_w"][s:e])
        daily_energy.append(np.sum(q_el * dt / 3.6e6))
        daily_cost.append(sum(log["cost_eur"][s:e]))
        daily_avg_tank.append(np.mean(log["T_tank"][s:e]))
        daily_avg_cop.append(np.mean(log["cop"][s:e]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Daily Performance Breakdown", fontsize=14, fontweight="bold")
    days = range(1, n_days + 1)

    axes[0, 0].bar(days, daily_energy, color="#2196F3", alpha=0.8)
    axes[0, 0].set_ylabel("Energy [kWh]")
    axes[0, 0].set_title("Daily Electrical Consumption")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(days, daily_cost, color="#FF9800", alpha=0.8)
    axes[0, 1].set_ylabel("Cost [€]")
    axes[0, 1].set_title("Daily Electricity Cost")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(days, daily_avg_tank, color="#E91E63", alpha=0.8)
    axes[1, 0].set_ylabel("Avg T_tank [°C]")
    axes[1, 0].set_title("Average Daily Tank Temperature")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(days, daily_avg_cop, color="#795548", alpha=0.8)
    axes[1, 1].set_ylabel("Avg COP")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_title("Average Daily COP")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "daily_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Daily breakdown saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MPC heat pump controller results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to MPC run directory containing trajectory.csv and summary.json",
    )

    args = parser.parse_args()

    # Load trajectory
    traj_path = os.path.join(args.results_dir, "trajectory.csv")
    if not os.path.exists(traj_path):
        print(f"❌ trajectory.csv not found in {args.results_dir}")
        return

    df = pd.read_csv(traj_path)
    log = {col: df[col].tolist() for col in df.columns if col != "Unnamed: 0"}

    # Load summary
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    output_dir = os.path.join(args.results_dir, "plots")
    plot_mpc_results(log, summary, output_dir)
    print(f"\n✅ Evaluation complete. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
