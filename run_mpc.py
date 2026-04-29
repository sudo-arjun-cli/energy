"""
Run MPC — Main entry point for the Hierarchical MPC Heat Pump Controller.

Replaces the old RL training script.  This script:
    1. Initialises the simulation (i4b + Tank + MPC)
    2. Runs the closed-loop MPC simulation
    3. Saves logs and summary to the runs/ directory
    4. Optionally launches evaluation plots

Usage:
    python run_mpc.py                              # defaults
    python run_mpc.py --building sfh_2016_now_2_kfw --days 14
    python run_mpc.py --days 7 --plot
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from config import SIMULATION_CONFIG, RUNS_DIR
from simulation import HeatPumpSimulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hierarchical MPC Heat Pump Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--building", type=str,
        default=SIMULATION_CONFIG["building"],
        help="Building model from i4b TABULA database",
    )
    parser.add_argument(
        "--days", type=int,
        default=SIMULATION_CONFIG["days"],
        help="Simulation length [days]",
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use synthetic weather/price data (no API calls)",
    )
    parser.add_argument(
        "--live", action="store_true", default=False,
        help="Use live aWATTar + BrightSky data",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Generate evaluation plots after simulation",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (auto-generated if not specified)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mpc_{args.building}_{timestamp}"

    output_dir = args.output_dir or str(RUNS_DIR / run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"📁 Output directory: {output_dir}")

    # ── Run simulation ──────────────────────────────────────────────────
    use_synthetic = not args.live
    sim = HeatPumpSimulation(
        building=args.building,
        days=args.days,
        use_synthetic=use_synthetic,
    )

    results = sim.run()

    # ── Save results ────────────────────────────────────────────────────
    # Summary
    summary_path = os.path.join(output_dir, "summary.json")
    summary = results["summary"]
    # Convert numpy types for JSON serialisation
    summary_json = {
        k: (float(v) if isinstance(v, (np.floating, float)) else v)
        for k, v in summary.items()
        if k != "tank_summary"
    }
    summary_json["tank_summary"] = {
        k: float(v) if isinstance(v, (np.floating, float)) else v
        for k, v in summary.get("tank_summary", {}).items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"📊 Summary saved to: {summary_path}")

    # Trajectory log (as CSV for easy analysis)
    import pandas as pd
    log_df = pd.DataFrame(results["log"])
    log_path = os.path.join(output_dir, "trajectory.csv")
    log_df.to_csv(log_path, index=True)
    print(f"📈 Trajectory saved to: {log_path}")

    # ── Optionally plot ─────────────────────────────────────────────────
    if args.plot:
        from evaluate import plot_mpc_results
        plot_mpc_results(results["log"], results["summary"], output_dir)


if __name__ == "__main__":
    main()
