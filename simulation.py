"""
Simulation — Couples the Hierarchical MPC with i4b and the Buffer Tank.

This module ties together:
    1. The i4b RC thermal model    (ground-truth house simulation)
    2. The BufferTank model        (thermal energy storage)
    3. The EconomicMPC             (upper layer — 24 h optimiser)
    4. The TrackingController      (lower layer — 15 min PI tracker)
    5. The DataPipeline            (aWATTar prices + BrightSky weather)

It runs a closed-loop simulation where the MPC re-plans every hour
and the tracking controller executes every 15 minutes.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional

from config import (
    I4B_ROOT,
    SIMULATION_CONFIG,
    MPC_CONFIG,
    HEATPUMP_CONFIG,
    TANK_CONFIG,
)
from buffer_tank import BufferTank
from mpc_controller import EconomicMPC, TrackingController
from data_pipeline import AWattarClient, generate_synthetic_weather

# Add i4b to path
sys.path.insert(0, str(I4B_ROOT))


class HeatPumpSimulation:
    """End-to-end simulation of the hierarchical MPC heat pump system.

    Manages the interaction between the MPC optimiser, the tracking
    controller, the i4b building model, and the buffer tank.
    """

    def __init__(
        self,
        building: str = None,
        days: int = None,
        use_synthetic: bool = True,
    ):
        cfg = SIMULATION_CONFIG

        self.building = building or cfg["building"]
        self.days = days or cfg["days"]
        self.dt = cfg["delta_t"]               # 900 s (15 min)
        self.use_synthetic = use_synthetic

        # Derived
        self.steps_per_hour = 3600 // self.dt  # 4
        self.total_steps = self.days * 24 * self.steps_per_hour
        self.upper_replan_steps = MPC_CONFIG["upper_replan_seconds"] // self.dt

        # ── Create components ───────────────────────────────────────────
        self.tank = BufferTank()
        self.economic_mpc = EconomicMPC()
        self.tracker = TrackingController()

        # ── Prepare data ────────────────────────────────────────────────
        self._prepare_data()

        # ── Create i4b environment ──────────────────────────────────────
        self._create_i4b_env()

        # ── Logging ─────────────────────────────────────────────────────
        self.log = {
            "T_room": [], "T_tank": [], "T_amb": [],
            "supply_temp": [], "Q_HP_w": [], "Q_house_w": [],
            "Q_el_w": [], "price_eur_kwh": [], "valve_pos": [],
            "reward": [], "cost_eur": [], "cop": [],
            "T_tank_ref": [], "T_room_ref": [],
        }

    # ────────────────────────────────────────────────────────────────────
    def _prepare_data(self):
        """Load or generate weather and price data."""
        start = "2024-01-01"
        end_dt = pd.Timestamp(start) + pd.Timedelta(days=self.days + 2)
        end = end_dt.strftime("%Y-%m-%d")

        if self.use_synthetic:
            self.weather = generate_synthetic_weather(start, end, freq="h")
            self.prices = AWattarClient.generate_synthetic_prices(start, end)
        else:
            from data_pipeline import DataPipeline
            pipeline = DataPipeline()
            self.weather, self.prices = pipeline.get_training_data(
                start, end, use_synthetic=False,
            )

        # Resample to simulation timestep
        self.prices_15min = self.prices["price_eur_kwh"].resample(
            f"{self.dt}s"
        ).ffill()
        self.weather_hourly = self.weather.resample("h").mean().ffill()

    # ────────────────────────────────────────────────────────────────────
    def _create_i4b_env(self):
        """Initialise the i4b building simulation."""
        from src.gym_interface import make_room_heat_env

        original_cwd = os.getcwd()
        os.chdir(str(I4B_ROOT))

        try:
            fc_steps = list(range(1, SIMULATION_CONFIG["forecast_steps"] + 1))
            self._base_env = make_room_heat_env(
                building=self.building,
                hp_model=SIMULATION_CONFIG["hp_model"],
                method=SIMULATION_CONFIG["method"],
                mdot_HP=SIMULATION_CONFIG["mdot_hp"],
                internal_gain_profile=SIMULATION_CONFIG["internal_gain_profile"],
                weather_forecast_steps=fc_steps,
                delta_t=self.dt,
                days=self.days,
                random_init=SIMULATION_CONFIG["random_init"],
                goal_based=SIMULATION_CONFIG["goal_based"],
                goal_temp_range=SIMULATION_CONFIG["goal_temp_range"],
                temp_deviation_weight=SIMULATION_CONFIG["temp_deviation_weight"],
            )
        finally:
            os.chdir(original_cwd)

        self._env = self._base_env.unwrapped

    # ────────────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """Run the full closed-loop MPC simulation.

        Returns
        -------
        dict
            Complete simulation log and summary statistics.
        """
        print(f"\n{'='*60}")
        print(f"  Hierarchical MPC Simulation")
        print(f"  Building : {self.building}")
        print(f"  Duration : {self.days} days ({self.total_steps} steps)")
        print(f"  Tank     : {self.tank}")
        print(f"{'='*60}\n")

        # Reset all components
        obs, info = self._env.reset()
        self.tank.reset()
        self.tracker.reset()

        # Current MPC plan (filled on first step)
        plan = None
        plan_step = 0

        total_cost = 0.0
        total_energy_kwh = 0.0
        comfort_violations = 0

        for step in range(self.total_steps):
            # ── Get current state ───────────────────────────────────────
            try:
                T_room = float(info.get("T_room", 20.0))
            except (TypeError, KeyError):
                T_room = 20.0
            T_tank = self.tank.temperature

            try:
                T_amb = float(self._env.get_cur_T_amb())
            except (IndexError, AttributeError):
                T_amb = 5.0

            # Price at current step
            price_idx = min(step, len(self.prices_15min) - 1)
            current_price = float(self.prices_15min.iloc[price_idx])

            # ── Upper Layer: Re-plan every hour ─────────────────────────
            if step % self.upper_replan_steps == 0:
                plan = self._run_upper_layer(step, T_room, T_tank)
                plan_step = 0

            # ── Look up reference for this step ─────────────────────────
            # Upper layer runs at 1 h, lower at 15 min → interpolate
            upper_idx = min(plan_step // self.steps_per_hour, self.economic_mpc.N - 1)
            T_tank_ref = float(plan["T_tank_ref"][upper_idx + 1])
            T_room_ref = float(plan["T_room_ref"][upper_idx + 1])
            Q_HP_ref = float(plan["Q_HP_ref"][upper_idx])
            Q_house_ref = float(plan["Q_house_ref"][upper_idx])

            # ── Lower Layer: Tracking control ───────────────────────────
            cmd = self.tracker.compute(
                T_tank=T_tank,
                T_room=T_room,
                T_tank_ref=T_tank_ref,
                T_room_ref=T_room_ref,
                Q_HP_ref=Q_HP_ref,
                Q_house_ref=Q_house_ref,
                dt=self.dt,
            )

            q_hp_w = cmd["q_hp_w"]
            supply_temp = cmd["supply_temp"]
            valve_pos = cmd["valve_pos"]

            # ── Compute electrical consumption ──────────────────────────
            dT_lift = max(T_tank - T_amb, 5.0)
            cop = min(
                HEATPUMP_CONFIG["carnot_efficiency"] * (T_tank + 273.15) / (dT_lift + 5.0),
                7.0,
            )
            cop = max(cop, 1.0)
            q_el_w = q_hp_w / cop

            # ── Compute heat extracted from tank to house ───────────────
            # Approximate from i4b's energy consumption in the step
            q_house_w = valve_pos * max(T_tank - T_room, 0) * SIMULATION_CONFIG["mdot_hp"] * 4182
            q_house_w = min(q_house_w, q_hp_w + self.tank.C * 0.5 / self.dt)  # physical limit

            # ── Step the buffer tank ────────────────────────────────────
            tank_result = self.tank.step(q_hp_w, q_house_w, self.dt)

            # ── Step the i4b house model ────────────────────────────────
            # Map supply temp to i4b action range [-1, 1]
            t_min_hp = HEATPUMP_CONFIG["supply_temp_min"]
            t_max_hp = HEATPUMP_CONFIG["supply_temp_max"]
            action_normalized = 2.0 * (supply_temp - t_min_hp) / (t_max_hp - t_min_hp) - 1.0
            action_normalized = np.clip(action_normalized, -1.0, 1.0)

            obs, _, terminated, truncated, info = self._env.step(float(action_normalized))

            # ── Accounting ──────────────────────────────────────────────
            energy_kwh = q_el_w * self.dt / 3.6e6
            step_cost = energy_kwh * current_price
            total_cost += step_cost
            total_energy_kwh += energy_kwh

            T_room_actual = float(info.get("T_room", T_room))
            if T_room_actual < MPC_CONFIG["room_t_min"] or T_room_actual > MPC_CONFIG["room_t_max"]:
                comfort_violations += 1

            # ── Log ─────────────────────────────────────────────────────
            self.log["T_room"].append(T_room_actual)
            self.log["T_tank"].append(T_tank)
            self.log["T_amb"].append(T_amb)
            self.log["supply_temp"].append(supply_temp)
            self.log["Q_HP_w"].append(q_hp_w)
            self.log["Q_house_w"].append(q_house_w)
            self.log["Q_el_w"].append(q_el_w)
            self.log["price_eur_kwh"].append(current_price)
            self.log["valve_pos"].append(valve_pos)
            self.log["cost_eur"].append(step_cost)
            self.log["cop"].append(cop)
            self.log["T_tank_ref"].append(T_tank_ref)
            self.log["T_room_ref"].append(T_room_ref)

            plan_step += 1

            # Progress
            if step % (self.steps_per_hour * 24) == 0:
                day = step // (self.steps_per_hour * 24)
                print(f"  Day {day}/{self.days}  |  "
                      f"T_room={T_room_actual:.1f}°C  T_tank={T_tank:.1f}°C  "
                      f"Cost so far: €{total_cost:.2f}")

            if terminated or truncated:
                break

        # ── Summary ─────────────────────────────────────────────────────
        summary = {
            "total_energy_kwh": total_energy_kwh,
            "total_cost_eur": total_cost,
            "comfort_violations": comfort_violations,
            "comfort_violation_pct": 100 * comfort_violations / max(step + 1, 1),
            "avg_T_room": np.mean(self.log["T_room"]),
            "avg_T_tank": np.mean(self.log["T_tank"]),
            "avg_cop": np.mean(self.log["cop"]),
            "tank_summary": self.tank.get_summary(),
            "n_steps": step + 1,
        }

        print(f"\n{'='*60}")
        print(f"  Simulation Complete")
        print(f"  Total Energy : {summary['total_energy_kwh']:.1f} kWh")
        print(f"  Total Cost   : €{summary['total_cost_eur']:.2f}")
        print(f"  Comfort Viol.: {summary['comfort_violations']} "
              f"({summary['comfort_violation_pct']:.1f}%)")
        print(f"  Avg T_room   : {summary['avg_T_room']:.1f}°C")
        print(f"  Avg T_tank   : {summary['avg_T_tank']:.1f}°C")
        print(f"  Avg COP      : {summary['avg_cop']:.2f}")
        print(f"{'='*60}\n")

        return {"log": self.log, "summary": summary}

    # ────────────────────────────────────────────────────────────────────
    def _run_upper_layer(self, step: int, T_room: float, T_tank: float) -> Dict:
        """Invoke the Economic MPC for the next 24 hours."""
        N = self.economic_mpc.N

        # Build price forecast (next N hours)
        hour_idx = step // self.steps_per_hour
        prices = np.zeros(N)
        for k in range(N):
            idx = min(hour_idx + k, len(self.weather_hourly) - 1)
            p_idx = min((step + k * self.steps_per_hour), len(self.prices_15min) - 1)
            prices[k] = float(self.prices_15min.iloc[p_idx])

        # Build weather forecast (next N hours)
        T_amb = np.zeros(N)
        Q_solar = np.zeros(N)
        for k in range(N):
            idx = min(hour_idx + k, len(self.weather_hourly) - 1)
            T_amb[k] = float(self.weather_hourly["T_amb"].iloc[idx])
            Q_solar[k] = float(self.weather_hourly["solar"].iloc[idx]) * 0.3  # rough window gain

        plan = self.economic_mpc.solve(
            T_room_now=T_room,
            T_tank_now=T_tank,
            prices_eur_kwh=prices,
            T_amb_forecast=T_amb,
            Q_solar_forecast=Q_solar,
        )

        if plan["status"] == "optimal":
            print(f"    ✅ MPC solved (predicted cost: €{plan['cost_eur']:.3f})")
        else:
            print(f"    ⚠️  MPC fallback active")

        return plan
