"""
Centralized configuration for the Heat Pump ML Controller.

This module provides all tunable parameters for:
- Building selection (from i4b Tabula data)
- Heat pump model selection
- Simulation parameters
- RL training hyperparameters
- Reward function weights
- Data source configuration

All parameters are organized as dataclass-style dicts for easy serialization
and command-line override.
"""

import os
from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the cloned i4b repository
I4B_ROOT = PROJECT_ROOT / "i4b"

# Output directories
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure output dirs exist
for d in [RUNS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Building Configuration (from i4b TABULA data for Germany)
# ============================================================================

# Available building epochs in i4b (Single Family Houses — SFH)
AVAILABLE_BUILDINGS = [
    "sfh_1919_1948_0_soc",  # Original state
    "sfh_1919_1948_1_enev", # Renovated to EnEV standard
    "sfh_1919_1948_2_kfw",  # Renovated to KfW standard
    "sfh_1949_1957_0_soc",
    "sfh_1949_1957_1_enev",
    "sfh_1949_1957_2_kfw",
    "sfh_1958_1968_0_soc",
    "sfh_1958_1968_1_enev",
    "sfh_1958_1968_2_kfw",
    "sfh_1969_1978_0_soc",
    "sfh_1969_1978_1_enev",
    "sfh_1969_1978_2_kfw",
    "sfh_1979_1983_0_soc",
    "sfh_1979_1983_1_enev",
    "sfh_1979_1983_2_kfw",
    "sfh_1984_1994_0_soc",
    "sfh_1984_1994_1_enev",
    "sfh_1984_1994_2_kfw",
    "sfh_1995_2001_0_soc",
    "sfh_1995_2001_1_enev",
    "sfh_1995_2001_2_kfw",
    "sfh_2002_2009_0_soc",
    "sfh_2002_2009_1_enev",
    "sfh_2002_2009_2_kfw",
    "sfh_2010_2015_0_soc",
    "sfh_2010_2015_1_enev",
    "sfh_2010_2015_2_kfw",
    "sfh_2016_now_0_soc",
    "sfh_2016_now_1_enev",
    "sfh_2016_now_2_kfw",
    "i4c",  # i4b test building
]

# Renovation standard descriptions
RENOVATION_STANDARDS = {
    "0_soc": "State of Construction (original, unrenovated)",
    "1_enev": "Renovated to EnEV / GEG standard",
    "2_kfw": "Renovated to KfW Efficiency House standard (best)",
}

# ============================================================================
# Default Simulation Configuration
# ============================================================================

SIMULATION_CONFIG = {
    # Building model
    "building": "sfh_1984_1994_1_enev",  # Common German house, EnEV renovated
    "method": "4R3C",                     # RC-network model complexity
    "hp_model": "Heatpump_AW",           # Air-Water heat pump from i4b

    # Heat pump parameters
    "mdot_hp": 0.25,                      # Mass flow rate [kg/s]

    # Simulation timing
    "delta_t": 900,                       # Timestep: 15 minutes (900 seconds)
    "days": 30,                           # Episode length: 30 days

    # Internal gains profile (relative to i4b root)
    "internal_gain_profile": "data/profiles/InternalGains/ResidentialDetached.csv",

    # Weather forecast
    "forecast_steps": 4,                  # 4 steps × 15 min = 1 hour look-ahead

    # Initialization
    "random_init": True,                  # Randomize start position in data

    # Goal-based learning
    "goal_based": True,                   # Enable goal temperature targeting
    "goal_temp_range": (19.0, 23.0),      # Comfortable temperature range [°C]
    "temp_deviation_weight": 5.0,         # Weight for temp deviation in reward
}

# ============================================================================
# Electricity Price Configuration
# ============================================================================

PRICE_CONFIG = {
    # BrightSky API (weather — free, no API key needed)
    "brightsky_base_url": "https://api.brightsky.dev",
    "default_lat": 49.87,    # Darmstadt, Germany
    "default_lon": 8.65,

    # ENTSO-E Transparency Platform (electricity prices)
    "entsoe_api_key": os.environ.get("ENTSOE_API_KEY", ""),
    "bidding_zone": "DE_LU",  # Germany-Luxembourg

    # Synthetic price profile (fallback when no API key)
    "use_synthetic_prices": True,
    "synthetic_base_price": 0.30,         # €/kWh base
    "synthetic_peak_multiplier": 1.8,     # Peak hours multiplier
    "synthetic_off_peak_multiplier": 0.6, # Off-peak multiplier
    "peak_hours": list(range(6, 10)) + list(range(17, 21)),  # 6-10, 17-21
}

# ============================================================================
# Reward Function Weights
# ============================================================================

REWARD_WEIGHTS = {
    # Temperature comfort (HIGHEST PRIORITY)
    "comfort_weight": 4.0,
    "comfort_range": (20.0, 23.0),         # Acceptable room temp [°C]
    "comfort_penalty_scale": 5.0,          # Quadratic penalty scale

    # Electricity cost minimization
    "cost_weight": 2.0,
    "cost_scale": 10.0,                    # Normalization factor

    # Compressor cycling penalty
    "cycling_weight": 1.0,
    "cycling_penalty": 2.0,                # Per switch event

    # Energy efficiency bonus
    "efficiency_weight": 0.5,
    "efficiency_bonus_threshold": 3.0,     # COP above this gets bonus
}

# ============================================================================
# RL Training Hyperparameters (PPO)
# ============================================================================

PPO_CONFIG = {
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,                       # Steps per rollout
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,                         # Discount factor
    "gae_lambda": 0.95,                    # GAE lambda
    "clip_range": 0.2,                     # PPO clip range
    "ent_coef": 0.01,                      # Entropy coefficient
    "vf_coef": 0.5,                        # Value function coefficient
    "max_grad_norm": 0.5,                  # Gradient clipping
    "seed": 42,

    # Network architecture
    "policy_kwargs": {
        "net_arch": {
            "pi": [128, 128, 64],          # Policy network
            "vf": [128, 128, 64],          # Value network
        }
    },

    # Device
    "device": "auto",                      # 'auto', 'cpu', or 'cuda'
}

# ============================================================================
# Evaluation Configuration
# ============================================================================

EVAL_CONFIG = {
    "n_eval_episodes": 10,
    "eval_freq": 10_000,                   # Evaluate every N steps
    "deterministic": True,                 # Use deterministic policy for eval
    "render": False,
}
