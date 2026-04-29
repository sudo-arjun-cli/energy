"""
Centralized configuration for the Hierarchical MPC Heat Pump Controller.

This module provides all tunable parameters for:
- Building selection (from i4b Tabula data)
- Heat pump specifications
- Buffer tank (Thermal Energy Storage)
- MPC controller (Upper + Lower layers)
- Data source configuration
"""

import os
from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
I4B_ROOT = PROJECT_ROOT / "i4b"

RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [RUNS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Building Configuration (from i4b TABULA data for Germany)
# ============================================================================

AVAILABLE_BUILDINGS = [
    "sfh_1919_1948_0_soc",
    "sfh_1919_1948_1_enev",
    "sfh_1919_1948_2_kfw",
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
    "i4c",
]

RENOVATION_STANDARDS = {
    "0_soc": "State of Construction (original, unrenovated)",
    "1_enev": "Renovated to EnEV / GEG standard",
    "2_kfw": "Renovated to KfW Efficiency House standard (best)",
}

# ============================================================================
# Simulation Configuration (i4b backend)
# ============================================================================

SIMULATION_CONFIG = {
    "building": "sfh_1984_1994_1_enev",
    "method": "4R3C",
    "hp_model": "Heatpump_AW",
    "mdot_hp": 0.25,                      # Mass flow rate [kg/s]
    "delta_t": 900,                       # Timestep: 15 minutes (900 seconds)
    "days": 7,                            # Simulation length
    "internal_gain_profile": "data/profiles/InternalGains/ResidentialDetached.csv",
    "forecast_steps": 4,
    "random_init": False,
    "goal_based": True,
    "goal_temp_range": (19.0, 23.0),
    "temp_deviation_weight": 5.0,
}

# ============================================================================
# Data Source Configuration
# ============================================================================

PRICE_CONFIG = {
    # BrightSky API (weather — free, no API key needed)
    "brightsky_base_url": "https://api.brightsky.dev",
    "default_lat": 49.87,    # Darmstadt, Germany
    "default_lon": 8.65,

    # aWATTar API (electricity prices — free, no API key needed)
    "awattar_base_url": "https://api.awattar.at/v1/marketdata",

    # Synthetic price profile (fallback)
    "use_synthetic_prices": True,
    "synthetic_base_price": 0.30,         # €/kWh base
    "synthetic_peak_multiplier": 1.8,
    "synthetic_off_peak_multiplier": 0.6,
    "peak_hours": list(range(6, 10)) + list(range(17, 21)),
}

# ============================================================================
# Buffer Tank (Thermal Energy Storage) Configuration
# ============================================================================

TANK_CONFIG = {
    "volume_liters": 500,                 # Tank volume [L]
    "ua_value": 3.0,                      # Standby heat loss coefficient [W/K]
    "t_ambient": 15.0,                    # Basement / surrounding temp [°C]
    "t_min": 30.0,                        # Minimum allowed tank temp [°C]
    "t_max": 80.0,                        # Maximum allowed tank temp [°C]
    "t_init": 45.0,                       # Initial tank temperature [°C]
}

# ============================================================================
# Heat Pump Configuration
# ============================================================================

HEATPUMP_CONFIG = {
    "q_max_w": 8000,                      # Max thermal power output [W]
    "q_min_w": 0,                         # Min thermal power (0 = on/off) [W]
    "carnot_efficiency": 0.45,            # Fraction of ideal Carnot COP
    "min_runtime_s": 600,                 # Minimum compressor ON time [s]
    "min_offtime_s": 600,                 # Minimum compressor OFF time [s]
    "supply_temp_max": 65.0,              # Max supply temperature [°C]
    "supply_temp_min": 20.0,              # Min supply temperature [°C]
}

# ============================================================================
# Simplified Building Model (for Upper Layer MPC prediction)
# ============================================================================

BUILDING_MODEL_CONFIG = {
    # Simplified 1R1C lumped model for the Economic MPC.
    # These values approximate a renovated 1984-1994 SFH (EnEV).
    "ua_building": 150.0,                 # Overall UA-value [W/K]
    "c_room": 15e6,                       # Effective thermal capacity [J/K]
    "radiator_ua": 500.0,                 # Radiator heat transfer coeff [W/K]
    "floor_area": 150.0,                  # Heated floor area [m²]
}

# ============================================================================
# MPC Controller Configuration
# ============================================================================

MPC_CONFIG = {
    # ── Upper Layer (Economic MPC) ──
    "upper_horizon_hours": 24,            # Prediction horizon [h]
    "upper_step_seconds": 3600,           # Discretization step [s] (1 hour)
    "upper_replan_seconds": 3600,         # Re-solve frequency [s]
    "upper_solver": "ipopt",

    # ── Lower Layer (Tracking Controller) ──
    "lower_horizon_hours": 2,             # Tracking horizon [h]
    "lower_step_seconds": 900,            # Control step [s] (15 min)

    # ── Comfort Bounds ──
    "room_t_min": 20.0,                   # Minimum room temperature [°C]
    "room_t_max": 23.0,                   # Maximum room temperature [°C]

    # ── Objective Weights ──
    "w_electricity": 1.0,                 # Weight for electricity cost
    "w_comfort": 100.0,                   # Weight for comfort slack penalty
    "w_cycling": 1e-6,                    # Weight for HP cycling penalty
    "w_tank_loss": 0.0,                   # Weight for explicit tank loss penalty
}

# ============================================================================
# Evaluation Configuration
# ============================================================================

EVAL_CONFIG = {
    "deterministic": True,
    "render": False,
}
