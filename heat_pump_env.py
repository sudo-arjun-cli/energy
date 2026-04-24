"""
HeatPumpControlEnv — Enhanced Gymnasium Environment for RL-based Heat Pump Control.

This module wraps i4b's RoomHeatEnv and extends it with:
1. Electricity price awareness (from ENTSO-E or synthetic data)
2. Multi-objective reward function (comfort + cost + cycling)
3. Extended observation space with price signals and time features
4. Compressor cycling tracking and penalties

The environment follows the standard Gymnasium API (step, reset, etc.)
and is fully compatible with Stable-Baselines3.
"""

import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add i4b to Python path so its internal imports work
from config import I4B_ROOT, SIMULATION_CONFIG, REWARD_WEIGHTS, PRICE_CONFIG

sys.path.insert(0, str(I4B_ROOT))

from src.gym_interface import make_room_heat_env, BUILDING_NAMES2CLASS
from data_pipeline import ENTSOEClient, generate_synthetic_weather


class HeatPumpControlEnv(gym.Env):
    """Extended heat pump control environment with price-aware rewards.

    This environment wraps i4b's RoomHeatEnv and augments it with:
    - Electricity price signals in the observation space
    - Time-of-day and day-of-week cyclical features
    - Multi-objective reward combining comfort, cost, and cycling
    - Compressor cycling tracking

    Observation Space (extended beyond base i4b env):
    ┌─────────────────────┬──────────────┬──────────────────────────────┐
    │ Feature             │ Dimension    │ Source                       │
    ├─────────────────────┼──────────────┼──────────────────────────────┤
    │ Building states     │ 3-7 (varies) │ i4b RC model (T_room, etc.) │
    │ Disturbances        │ 2            │ T_amb, Qdot_gains           │
    │ Weather forecast    │ 0-N          │ T_amb forecast steps        │
    │ Electricity price   │ 2            │ current + next-hour price   │
    │ Price features      │ 2            │ price_rank, is_cheap_hour   │
    │ Time features       │ 4            │ hour_sin/cos, dow_sin/cos   │
    │ Compressor state    │ 2            │ prev_action, runtime_frac   │
    │ Goal temperature    │ 0-1          │ if goal_based=True          │
    └─────────────────────┴──────────────┴──────────────────────────────┘

    Action Space:
    - Continuous [-1, 1] → mapped to supply temperature [20°C, 65°C]
    - Same as i4b's base action space
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        building: str = None,
        hp_model: str = None,
        method: str = None,
        mdot_hp: float = None,
        internal_gain_profile: str = None,
        delta_t: int = None,
        days: int = None,
        forecast_steps: int = None,
        random_init: bool = None,
        goal_based: bool = None,
        goal_temp_range: Tuple[float, float] = None,
        temp_deviation_weight: float = None,
        price_data: Optional[pd.DataFrame] = None,
        reward_weights: Optional[Dict] = None,
    ):
        """Initialize the HeatPumpControlEnv.

        Parameters
        ----------
        building : str
            Building model name from i4b's TABULA database.
        hp_model : str
            Heat pump model class name (e.g., 'Heatpump_AW').
        method : str
            RC-network model method (e.g., '4R3C', '7R5C').
        mdot_hp : float
            Mass flow rate of heat pump [kg/s].
        delta_t : int
            Simulation timestep in seconds.
        days : int
            Episode length in days.
        forecast_steps : int
            Number of weather forecast steps to include.
        random_init : bool
            Randomize initial conditions and start position.
        goal_based : bool
            Enable goal-based temperature targeting.
        goal_temp_range : tuple
            (min, max) goal temperature range [°C].
        temp_deviation_weight : float
            Weight for temperature deviation penalty.
        price_data : pd.DataFrame, optional
            Pre-loaded electricity price data. If None, generates synthetic.
        reward_weights : dict, optional
            Override default reward weights from config.
        """
        super().__init__()

        # Apply defaults from config
        cfg = SIMULATION_CONFIG
        self._building = building or cfg["building"]
        self._hp_model = hp_model or cfg["hp_model"]
        self._method = method or cfg["method"]
        self._mdot_hp = mdot_hp or cfg["mdot_hp"]
        self._internal_gain_profile = internal_gain_profile or cfg["internal_gain_profile"]
        self._delta_t = delta_t or cfg["delta_t"]
        self._days = days or cfg["days"]
        self._forecast_steps = forecast_steps if forecast_steps is not None else cfg["forecast_steps"]
        self._random_init = random_init if random_init is not None else cfg["random_init"]
        self._goal_based = goal_based if goal_based is not None else cfg["goal_based"]
        self._goal_temp_range = goal_temp_range or cfg["goal_temp_range"]
        self._temp_dev_weight = temp_deviation_weight if temp_deviation_weight is not None else cfg["temp_deviation_weight"]

        # Reward configuration
        self._rw = reward_weights or REWARD_WEIGHTS

        # Build forecast steps list
        fc_steps = list(range(1, self._forecast_steps + 1)) if self._forecast_steps > 0 else []

        # Create the base i4b environment
        # We need to change cwd to i4b root for its data loading to work
        import os
        original_cwd = os.getcwd()
        os.chdir(str(I4B_ROOT))

        try:
            self._base_env = make_room_heat_env(
                building=self._building,
                hp_model=self._hp_model,
                method=self._method,
                mdot_HP=self._mdot_hp,
                internal_gain_profile=self._internal_gain_profile,
                weather_forecast_steps=fc_steps,
                delta_t=self._delta_t,
                days=self._days,
                random_init=self._random_init,
                goal_based=self._goal_based,
                goal_temp_range=self._goal_temp_range,
                temp_deviation_weight=self._temp_dev_weight,
            )
        finally:
            os.chdir(original_cwd)

        # Get the unwrapped env to access internal state
        self._env = self._base_env.unwrapped

        # ── Electricity price data ──────────────────────────────────────
        if price_data is not None:
            self._price_data = price_data
        else:
            # Generate synthetic prices matching the simulation length
            total_hours = len(self._env.p)
            start_date = "2024-01-01"
            end_date = pd.Timestamp(start_date) + pd.Timedelta(hours=total_hours)
            self._price_data = ENTSOEClient.generate_synthetic_prices(
                start_date, end_date.strftime("%Y-%m-%d")
            )

        # Resample prices to match simulation timestep
        self._prices_resampled = self._price_data["price_eur_kwh"].resample(
            f"{self._delta_t}s"
        ).ffill()

        # ── Compressor tracking ─────────────────────────────────────────
        self._prev_compressor_on = False
        self._compressor_switches = 0
        self._steps_since_switch = 0
        self._episode_energy_kwh = 0.0
        self._episode_cost_eur = 0.0
        self._episode_comfort_violations = 0

        # ── Extended observation & action spaces ────────────────────────
        base_obs_dim = self._env.observation_space.shape[0]

        # Additional dimensions: price(2) + price_features(2) + time(4) + compressor_state(2)
        extra_dims = 10
        total_obs_dim = base_obs_dim + extra_dims

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

        # Action space: same as base (continuous [-1, 1])
        self.action_space = self._env.action_space

        print(f"\n{'='*60}")
        print(f"  HeatPumpControlEnv initialized")
        print(f"  Building: {self._building}")
        print(f"  Heat Pump: {self._hp_model}")
        print(f"  RC Model: {self._method}")
        print(f"  Timestep: {self._delta_t}s ({self._delta_t/60:.0f} min)")
        print(f"  Episode: {self._days} days")
        print(f"  Base obs dim: {base_obs_dim}")
        print(f"  Extended obs dim: {total_obs_dim}")
        print(f"  Goal-based: {self._goal_based}")
        print(f"  Forecast steps: {self._forecast_steps}")
        print(f"{'='*60}\n")

    # ════════════════════════════════════════════════════════════════════
    # Gymnasium API
    # ════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        """Reset the environment and return extended initial observation."""
        base_obs, base_info = self._env.reset(seed=seed)

        # Reset tracking
        self._prev_compressor_on = False
        self._compressor_switches = 0
        self._steps_since_switch = 0
        self._episode_energy_kwh = 0.0
        self._episode_cost_eur = 0.0
        self._episode_comfort_violations = 0

        obs = self._extend_observation(base_obs)
        return obs, base_info

    def step(self, action):
        """Execute one timestep and return extended observation + shaped reward."""
        # i4b's RoomHeatEnv expects a scalar action, but SB3 passes np.array([x])
        if hasattr(action, '__len__') and len(action) == 1:
            action_scalar = action[0]
        else:
            action_scalar = action

        # Step the base i4b environment
        base_obs, base_reward, terminated, truncated, info = self._env.step(action_scalar)

        # ── Get current price ───────────────────────────────────────────
        price_idx = min(self._env.t, len(self._prices_resampled) - 1)
        current_price = float(self._prices_resampled.iloc[price_idx])
        next_price = float(self._prices_resampled.iloc[min(price_idx + 1, len(self._prices_resampled) - 1)])

        # ── Track compressor cycling ────────────────────────────────────
        current_compressor_on = info.get("Q_el_kWh", 0) > 0.001
        switched = current_compressor_on != self._prev_compressor_on
        if switched:
            self._compressor_switches += 1
            self._steps_since_switch = 0
        else:
            self._steps_since_switch += 1
        self._prev_compressor_on = current_compressor_on

        # ── Track energy & cost ─────────────────────────────────────────
        energy_kwh = info.get("Q_el_kWh", 0.0)
        step_cost = energy_kwh * current_price
        self._episode_energy_kwh += energy_kwh
        self._episode_cost_eur += step_cost

        # ── Compute shaped reward ───────────────────────────────────────
        T_room = info.get("T_room", 20.0)
        reward = self._compute_reward(
            T_room=T_room,
            energy_kwh=energy_kwh,
            current_price=current_price,
            switched=switched,
        )

        # ── Track comfort violations ────────────────────────────────────
        T_min, T_max = self._rw["comfort_range"]
        if T_room < T_min or T_room > T_max:
            self._episode_comfort_violations += 1

        # ── Extend info dict ────────────────────────────────────────────
        info.update({
            "price_eur_kwh": current_price,
            "step_cost_eur": step_cost,
            "compressor_switches": self._compressor_switches,
            "episode_energy_kwh": self._episode_energy_kwh,
            "episode_cost_eur": self._episode_cost_eur,
            "comfort_violations": self._episode_comfort_violations,
            "shaped_reward": reward,
        })

        # ── Extend observation ──────────────────────────────────────────
        obs = self._extend_observation(base_obs)

        return obs, reward, terminated, truncated, info

    # ════════════════════════════════════════════════════════════════════
    # Internal Methods
    # ════════════════════════════════════════════════════════════════════

    def _extend_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Augment the base i4b observation with price and time features.

        Additional features appended:
        [0] current_price (normalized to ~[0,1])
        [1] next_hour_price (normalized)
        [2] price_rank (normalized to [0,1])
        [3] is_cheap_hour (0.0 or 1.0)
        [4] hour_sin  (cyclical hour encoding)
        [5] hour_cos
        [6] dow_sin   (cyclical day-of-week encoding)
        [7] dow_cos
        [8] prev_action_normalized (last compressor state)
        [9] runtime_fraction (fraction of minimum ON/OFF time)
        """
        # Price features
        t = self._env.t
        price_idx = min(t, len(self._prices_resampled) - 1)
        current_price = float(self._prices_resampled.iloc[price_idx])
        next_price = float(self._prices_resampled.iloc[min(price_idx + 1, len(self._prices_resampled) - 1)])

        # Normalize prices (typical range 0.05-0.50 €/kWh → ~[0,1])
        price_norm = current_price / 0.50
        next_price_norm = next_price / 0.50

        # Price rank within current day (approximate)
        steps_per_day = 24 * 3600 // self._delta_t
        day_start = (t // steps_per_day) * steps_per_day
        day_end = min(day_start + steps_per_day, len(self._prices_resampled))
        if day_end > day_start:
            day_prices = self._prices_resampled.iloc[day_start:day_end].values
            rank = np.searchsorted(np.sort(day_prices), current_price) / max(len(day_prices), 1)
        else:
            rank = 0.5
        is_cheap = 1.0 if current_price < np.median(self._prices_resampled.values[:min(steps_per_day, len(self._prices_resampled))]) else 0.0

        # Time features (cyclical encoding)
        try:
            cur_time = self._env.get_cur_time()
            hour = cur_time.hour + cur_time.minute / 60.0
            dow = cur_time.dayofweek
        except (IndexError, AttributeError):
            hour = 12.0
            dow = 0

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Compressor state features
        prev_action_norm = 1.0 if self._prev_compressor_on else 0.0
        # Min ON/OFF time = 10 min = 10*60/delta_t steps
        min_steps = max(1, 10 * 60 // self._delta_t)
        runtime_frac = min(1.0, self._steps_since_switch / min_steps)

        extra = np.array([
            price_norm,
            next_price_norm,
            rank,
            is_cheap,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            prev_action_norm,
            runtime_frac,
        ], dtype=np.float32)

        return np.concatenate([base_obs, extra])

    def _compute_reward(
        self,
        T_room: float,
        energy_kwh: float,
        current_price: float,
        switched: bool,
    ) -> float:
        """Multi-objective reward function.

        Balances four competing objectives:
        1. Thermal comfort (highest priority)
        2. Electricity cost
        3. Compressor cycling
        4. Energy efficiency

        Parameters
        ----------
        T_room : float
            Current room temperature [°C].
        energy_kwh : float
            Electricity consumed this step [kWh].
        current_price : float
            Current electricity price [€/kWh].
        switched : bool
            Whether the compressor switched state this step.

        Returns
        -------
        float
            Scalar reward value.
        """
        rw = self._rw

        # 1. Temperature comfort (quadratic penalty outside comfort range)
        T_min, T_max = rw["comfort_range"]
        if T_min <= T_room <= T_max:
            r_comfort = 1.0  # Bonus for being in range
        else:
            deviation = max(T_min - T_room, T_room - T_max)
            r_comfort = -rw["comfort_penalty_scale"] * deviation ** 2

        # 2. Electricity cost (price-weighted energy penalty)
        step_cost = energy_kwh * current_price
        r_cost = -step_cost * rw["cost_scale"]

        # 3. Compressor cycling penalty
        r_cycling = -rw["cycling_penalty"] if switched else 0.0

        # 4. Efficiency: small bonus for low energy use while in comfort
        r_efficiency = 0.0
        if T_min <= T_room <= T_max and energy_kwh < 0.5:
            r_efficiency = 0.1  # Small bonus for efficient comfort maintenance

        # Weighted sum
        reward = (
            rw["comfort_weight"] * r_comfort
            + rw["cost_weight"] * r_cost
            + rw["cycling_weight"] * r_cycling
            + rw["efficiency_weight"] * r_efficiency
        )

        return float(reward)

    # ════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ════════════════════════════════════════════════════════════════════

    def get_episode_summary(self) -> Dict:
        """Return summary statistics for the current/completed episode."""
        return {
            "total_energy_kwh": self._episode_energy_kwh,
            "total_cost_eur": self._episode_cost_eur,
            "compressor_switches": self._compressor_switches,
            "comfort_violations": self._episode_comfort_violations,
            "avg_cost_per_kwh": (
                self._episode_cost_eur / max(self._episode_energy_kwh, 0.001)
            ),
        }

    def get_building_info(self) -> Dict:
        """Return building model information."""
        return {
            "building": self._building,
            "hp_model": self._hp_model,
            "method": self._method,
            "mdot_hp": self._mdot_hp,
            "delta_t": self._delta_t,
        }

    def render(self, mode="human"):
        """Not implemented — use evaluation scripts for visualization."""
        pass

    def close(self):
        """Clean up."""
        if hasattr(self, '_base_env'):
            self._base_env.close()
