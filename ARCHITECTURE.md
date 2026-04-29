# Hierarchical MPC Heat Pump Controller — Architecture & Technical Reference

> **Version:** 2.0 — Hierarchical MPC with Buffer Tank  
> **Date:** April 2026  
> **Building Data:** i4b TABULA dataset (German single-family homes)  
> **Price Data:** aWATTar AT spot market API  
> **Weather Data:** BrightSky (DWD) API

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Why Hierarchical MPC](#2-why-hierarchical-mpc)
3. [Physical System](#3-physical-system)
4. [Control Architecture](#4-control-architecture)
5. [Upper Layer — Economic MPC](#5-upper-layer--economic-mpc)
6. [Lower Layer — Tracking Controller](#6-lower-layer--tracking-controller)
7. [Buffer Tank Model](#7-buffer-tank-model)
8. [Data Pipeline](#8-data-pipeline)
9. [Simulation Loop](#9-simulation-loop)
10. [File Structure & Module Map](#10-file-structure--module-map)
11. [Configuration Reference](#11-configuration-reference)
12. [How to Run](#12-how-to-run)
13. [Mathematical Formulation](#13-mathematical-formulation)

---

## 1. System Overview

This project implements an **intelligent, price-aware heat pump controller** for residential buildings in Germany. The controller uses a **two-layer Model Predictive Control (MPC)** architecture to minimise electricity costs while maintaining indoor thermal comfort.

The key idea: a **hot water buffer tank** acts as a thermal battery, decoupling *when* electricity is consumed from *when* heat is delivered. The MPC exploits volatile spot-market prices (via aWATTar) by charging the tank during cheap hours and discharging it when prices are high.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  aWATTar API  │  │  BrightSky   │  │  Synthetic Fallback   │  │
│  │  (€/kWh, 24h) │  │  (Weather)   │  │  (Offline mode)       │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
└─────────┼──────────────────┼─────────────────────┼──────────────┘
          │                  │                     │
          ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                               │
│         (data_pipeline.py — fetches, cleans, resamples)         │
└────────────────────────────┬────────────────────────────────────┘
                             │  prices[24h], T_amb[24h], Q_solar[24h]
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              UPPER LAYER — Economic MPC                         │
│     (mpc_controller.py::EconomicMPC — CasADi + IPOPT)          │
│                                                                 │
│  Horizon: 24 hours  |  Step: 1 hour  |  Re-plan: every hour    │
│                                                                 │
│  Outputs: T_tank_ref[24], T_room_ref[24], Q_HP_ref[24]         │
└────────────────────────────┬────────────────────────────────────┘
                             │  reference trajectories
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              LOWER LAYER — Tracking Controller                  │
│     (mpc_controller.py::TrackingController — PI control)        │
│                                                                 │
│  Step: 15 minutes  |  Tracks upper-layer references             │
│                                                                 │
│  Outputs: q_hp_w (HP power), supply_temp, valve_pos             │
└────────┬───────────────────────────────┬────────────────────────┘
         │                               │
         ▼                               ▼
┌──────────────────┐           ┌──────────────────────┐
│   BUFFER TANK    │           │   i4b BUILDING SIM   │
│ (buffer_tank.py) │──heat────▶│  (4R3C RC-model)     │
│  500 L, 30-80°C  │           │  TABULA SFH data     │
└──────────────────┘           └──────────────────────┘
```

---

## 2. Why Hierarchical MPC

### Why not a single MPC?

A single monolithic MPC that plans 24 hours ahead at 15-minute resolution would need **96 steps × multiple states and controls** — computationally expensive and difficult to tune. It also conflates two fundamentally different goals:

| Concern | Timescale | Nature |
|---|---|---|
| "When should I buy electricity?" | Hours | Economic, strategic |
| "How do I track the setpoint without overshooting?" | Minutes | Physical, reactive |

### Why not two independent systems?

Two completely separate systems (one for tank charging, one for house heating) cannot coordinate. The tank charger doesn't know how much heat the house will need, and the house controller doesn't know the price forecast. This leaves **money and efficiency on the table**.

### The hierarchical solution

The **upper layer** solves the economic question with a coarse model. The **lower layer** handles the physical reality with fast feedback. They communicate through **reference trajectories** — the upper layer says "the tank should be at 60°C by hour 14", and the lower layer figures out exactly how to get there.

---

## 3. Physical System

The system consists of four physical components:

### 3.1 Air-Source Heat Pump
- **Max thermal power:** 8 kW
- **COP model:** Carnot-based — `COP = η × (T_tank + 273.15) / (T_tank − T_amb + 5)`
- **Carnot efficiency:** η = 0.45 (typical for residential units)
- **COP range:** ~2.5 (cold day, hot tank) to ~5.5 (mild day, warm tank)
- **Key insight:** COP drops as tank temperature rises. Heating the tank to 80°C is thermodynamically expensive. The MPC balances this against price savings.

### 3.2 Buffer Tank (Thermal Energy Storage)
- **Volume:** 500 litres (≈ 2.07 MJ/K thermal capacity)
- **Temperature range:** 30°C to 80°C
- **Standby losses:** UA = 3.0 W/K (well-insulated, ~90 W loss at 60°C in 15°C basement)
- **Role:** Thermal battery. Stores cheap energy for later use.

### 3.3 Building (Heated Space)
- **Default:** Renovated 1984–1994 SFH (EnEV standard)
- **Thermal model:** i4b 4R3C lumped-parameter (detailed simulation)
- **Simplified model for MPC:** 1R1C with UA = 150 W/K, C = 15 MJ/K
- **Comfort band:** 20°C – 23°C

### 3.4 Mixing Valve & Radiators
- **UA_radiator:** 500 W/K
- The valve mixes hot tank water with cooler return water to control supply temperature
- `T_supply = valve × T_tank + (1 − valve) × T_return`

---

## 4. Control Architecture

### Information Flow (per simulation step)

```
Every 1 hour:
  ┌───────────────────────────────────────────────────────┐
  │ 1. Fetch 24h price forecast from aWATTar              │
  │ 2. Fetch 24h weather forecast from BrightSky          │
  │ 3. Read current T_room and T_tank                     │
  │ 4. Solve Economic MPC (IPOPT) → 24h reference plan    │
  └───────────────────────────────────────────────────────┘
                            │
              T_tank_ref[], T_room_ref[], Q_HP_ref[]
                            │
Every 15 minutes:           ▼
  ┌───────────────────────────────────────────────────────┐
  │ 5. Look up reference for current timestep             │
  │ 6. PI controller computes:                            │
  │    - q_hp_w:      heat pump thermal power [W]         │
  │    - supply_temp: water temperature to house [°C]     │
  │    - valve_pos:   mixing valve opening [0–1]          │
  │ 7. Step the buffer tank model (energy balance)        │
  │ 8. Step the i4b building simulation                   │
  │ 9. Log all variables                                  │
  └───────────────────────────────────────────────────────┘
```

---

## 5. Upper Layer — Economic MPC

**File:** `mpc_controller.py` → class `EconomicMPC`  
**Solver:** CasADi `Opti` interface + IPOPT (bundled)  
**Horizon:** 24 hours, 1-hour steps (N = 24)  
**Re-plan frequency:** Every 1 hour (receding horizon)

### What it optimises

The upper layer solves a **Non-Linear Programme (NLP)** at each re-plan:

**Minimise:**
```
J = Σ [ w_elec × price[k] × P_el[k] × Δt      ← electricity cost
      + w_comfort × slack[k]²                    ← comfort violation penalty
      + w_cycling × (Q_HP[k] − Q_HP[k-1])² ]    ← compressor cycling penalty
```

**Subject to:**
- Room dynamics: `T_room[k+1] = T_room[k] + Δt/C_room × (Q_wall + Q_house + Q_solar + Q_int)`
- Tank dynamics: `T_tank[k+1] = T_tank[k] + Δt/C_tank × (Q_HP − Q_house − Q_loss)`
- COP physics: `P_el = Q_HP / COP(T_tank, T_amb)`
- Comfort bounds: `T_room_min − slack ≤ T_room ≤ T_room_max + slack`
- Tank bounds: `30°C ≤ T_tank ≤ 80°C`
- HP capacity: `0 ≤ Q_HP ≤ 8000 W`
- Radiator limit: `Q_house ≤ UA_rad × max(T_tank − T_room, 0)`

### Decision variables (per step k)

| Variable | Description | Unit |
|---|---|---|
| `Q_HP[k]` | Heat pump thermal output | W |
| `Q_house[k]` | Heat from tank to house | W |
| `T_room[k]` | Room temperature state | °C |
| `T_tank[k]` | Tank temperature state | °C |
| `slack[k]` | Comfort relaxation | °C |

### Parameters (filled at runtime)

| Parameter | Source | Unit |
|---|---|---|
| `T_room_0` | i4b measurement | °C |
| `T_tank_0` | Tank model state | °C |
| `price[k]` | aWATTar API | €/kWh |
| `T_amb[k]` | BrightSky forecast | °C |
| `Q_solar[k]` | BrightSky × 0.3 window factor | W |
| `Q_int[k]` | Default 200 W (occupancy) | W |

### Outputs

The solver returns **reference trajectories** for the next 24 hours:
- `T_tank_ref[0..24]` — optimal tank temperature path
- `T_room_ref[0..24]` — optimal room temperature path  
- `Q_HP_ref[0..23]` — optimal HP power schedule

### Fallback strategy

If IPOPT fails to converge, the system switches to a **rule-based fallback**:
- Heat during cheap hours (price < median)
- Supply heat when room < T_min
- This ensures the system never stops operating

---

## 6. Lower Layer — Tracking Controller

**File:** `mpc_controller.py` → class `TrackingController`  
**Type:** Feedforward + PI feedback  
**Step:** 15 minutes (matches i4b simulation timestep)

### How it works

The tracking controller receives the upper layer's reference plan and must convert it into physical actuator commands. It uses two PI loops:

#### Tank Temperature Loop (controls heat pump)
```
e_tank = T_tank_ref − T_tank_actual
q_hp = Q_HP_ref + Kp_tank × e_tank + Ki_tank × ∫e_tank·dt
q_hp = clip(q_hp, 0, 8000)
```
- **Kp = 800 W/°C** — if the tank is 1°C below reference, add 800 W
- **Ki = 50 W/(°C·s)** — slowly correct persistent errors

#### Room Temperature Loop (controls mixing valve)
```
e_room = T_room_ref − T_room_actual
valve_pos = 0.5 + Kp_room × e_room + Ki_room × ∫e_room·dt
valve_pos = clip(valve_pos, 0, 1)
```
- **Kp = 0.15 /°C** — opens valve 15% per degree of error
- **Ki = 0.005** — slow integration for steady-state accuracy

#### Supply temperature calculation
```
T_return ≈ T_room + 5°C
T_supply = valve × T_tank + (1 − valve) × T_return
T_supply = min(T_supply, T_tank)    ← can't be hotter than tank
```

---

## 7. Buffer Tank Model

**File:** `buffer_tank.py` → class `BufferTank`

### Physics

A **single-node (well-mixed)** tank model. One state variable: `T_tank`.

```
C_tank × dT/dt = Q_HP − Q_house − Q_loss

where:
  C_tank = ρ × cp × V = 988 × 4182 × 0.5 = 2,065,908 J/K  (≈ 2.07 MJ/K)
  Q_loss = UA × (T_tank − T_ambient) = 3.0 × (T − 15)      [W]
```

### Euler discretisation (per step)
```python
dT = (q_in - q_out - q_loss) * dt / C_tank
T_tank_new = clip(T_tank + dT, 30°C, 80°C)
```

### Energy accounting

The tank tracks cumulative energy flows in kWh:
- `total_q_in_kwh` — total energy from heat pump
- `total_q_out_kwh` — total energy delivered to house
- `total_q_loss_kwh` — total standby losses

### Why single-node (not stratified)?

A stratified (multi-node) model is more accurate but adds complexity and computational cost to the MPC. For control purposes, the well-mixed assumption is standard and sufficient.

---

## 8. Data Pipeline

**File:** `data_pipeline.py`

### 8.1 aWATTar API — Electricity Prices

```
GET https://api.awattar.at/v1/marketdata?start={ms}&end={ms}
```

- **No API key required**
- Returns hourly day-ahead prices in €/MWh
- Converted to €/kWh for the MPC
- Enriched with `price_rank` (0–23) and `is_cheap_hour` flag
- **Fallback:** Synthetic price generator mimicking German market patterns (low overnight, morning ramp, midday solar dip, evening peak)

### 8.2 BrightSky API — Weather

```
GET https://api.brightsky.dev/weather?lat=49.87&lon=8.65&date={date}
```

- **No API key required** (uses German DWD open data)
- Returns hourly: temperature, solar irradiance, wind, cloud cover, humidity
- Default location: Darmstadt, Germany (49.87°N, 8.65°E)
- **Fallback:** Synthetic weather with seasonal German climate patterns

### 8.3 Synthetic Data Generators

Both price and weather data have synthetic generators for offline testing:
- **Weather:** Seasonal temperature (−2°C Jan to 20°C Jul), diurnal cycle, solar geometry
- **Prices:** German duck-curve pattern with configurable base price (default 0.30 €/kWh)

---

## 9. Simulation Loop

**File:** `simulation.py` → class `HeatPumpSimulation`

The simulation orchestrates the entire system in a closed loop. Here is one complete cycle:

```
INITIALISE:
  1. Create BufferTank(500L, T_init=45°C)
  2. Create EconomicMPC(N=24, dt=3600s)
  3. Create TrackingController()
  4. Create i4b env (4R3C model, selected building)
  5. Load/generate weather + price data

FOR each 15-minute step:

  ── READ STATE ──────────────────────────────────────
  T_room  ← i4b env (info["T_room"])
  T_tank  ← BufferTank.temperature
  T_amb   ← i4b env (get_cur_T_amb)
  price   ← prices_15min[step]

  ── UPPER LAYER (every 4th step = 1 hour) ──────────
  IF step % 4 == 0:
    Gather 24h price + weather forecasts
    plan = EconomicMPC.solve(T_room, T_tank, prices, T_amb, Q_solar)
    → produces T_tank_ref[25], T_room_ref[25], Q_HP_ref[24]

  ── LOWER LAYER (every step) ───────────────────────
  Look up reference for this step from the plan
  cmd = TrackingController.compute(
      T_tank, T_room, T_tank_ref, T_room_ref, Q_HP_ref, Q_house_ref
  )
  → produces q_hp_w, supply_temp, valve_pos

  ── COMPUTE ELECTRICAL CONSUMPTION ─────────────────
  COP = η_carnot × (T_tank + 273.15) / (T_tank − T_amb + 5)
  q_el_w = q_hp_w / COP

  ── STEP BUFFER TANK ───────────────────────────────
  tank.step(q_in=q_hp_w, q_out=q_house_w, dt=900)

  ── STEP i4b BUILDING ──────────────────────────────
  action = normalise supply_temp to [-1, 1] for i4b
  obs, _, terminated, truncated, info = env.step(action)

  ── LOG & ACCOUNT ──────────────────────────────────
  energy_kwh = q_el_w × 900 / 3.6e6
  step_cost  = energy_kwh × price
  Log: T_room, T_tank, T_amb, supply_temp, Q_HP, Q_el, price, COP, cost
```

---

## 10. File Structure & Module Map

```
dev/
├── ARCHITECTURE.md        ← This document
├── config.py              ← All tunable parameters (centralised)
├── data_pipeline.py       ← aWATTar + BrightSky + synthetic generators
├── buffer_tank.py         ← Thermal energy storage model
├── mpc_controller.py      ← EconomicMPC + TrackingController
├── simulation.py          ← Closed-loop simulation harness
├── run_mpc.py             ← CLI entry point (replaces old train.py)
├── evaluate.py            ← Plotting & analysis
├── requirements.txt       ← Python dependencies
├── .gitignore
├── i4b/                   ← Building physics library (git submodule)
│   └── src/
│       ├── simulator.py          ← RC thermal model solver
│       ├── gym_interface/        ← Gymnasium env wrapper
│       ├── models/               ← Building RC parameters
│       └── disturbances.py       ← Weather & occupancy
├── runs/                  ← Simulation output (auto-created)
├── logs/                  ← Log files (auto-created)
└── models/                ← Saved models (auto-created)
```

### Module dependency graph

```
config.py ──────────────────────────────────────────────┐
    │                                                    │
    ├──► buffer_tank.py                                  │
    │                                                    │
    ├──► mpc_controller.py ◄── casadi                    │
    │        ├── EconomicMPC                             │
    │        └── TrackingController                      │
    │                                                    │
    ├──► data_pipeline.py                                │
    │        ├── AWattarClient ──► aWATTar API            │
    │        ├── BrightSkyClient ──► BrightSky API       │
    │        └── generate_synthetic_weather()             │
    │                                                    │
    └──► simulation.py ◄── all above + i4b               │
              │                                          │
              ▼                                          │
         run_mpc.py ──► evaluate.py ◄────────────────────┘
```

---

## 11. Configuration Reference

All parameters live in `config.py`. Key sections:

### TANK_CONFIG
| Parameter | Value | Description |
|---|---|---|
| `volume_liters` | 500 | Tank volume |
| `ua_value` | 3.0 W/K | Heat loss coefficient |
| `t_ambient` | 15.0°C | Surrounding temperature |
| `t_min` / `t_max` | 30 / 80°C | Operating range |
| `t_init` | 45°C | Starting temperature |

### HEATPUMP_CONFIG
| Parameter | Value | Description |
|---|---|---|
| `q_max_w` | 8000 W | Max thermal output |
| `carnot_efficiency` | 0.45 | Fraction of ideal Carnot |
| `supply_temp_max` | 65°C | Max supply temperature |

### MPC_CONFIG
| Parameter | Value | Description |
|---|---|---|
| `upper_horizon_hours` | 24 | Economic planning horizon |
| `upper_step_seconds` | 3600 | 1-hour discretisation |
| `lower_step_seconds` | 900 | 15-min control step |
| `room_t_min` / `room_t_max` | 20 / 23°C | Comfort band |
| `w_electricity` | 1.0 | Cost weight |
| `w_comfort` | 100.0 | Comfort penalty weight |
| `w_cycling` | 1e-6 | Anti-cycling weight |

### BUILDING_MODEL_CONFIG (simplified, for upper-layer MPC)
| Parameter | Value | Description |
|---|---|---|
| `ua_building` | 150 W/K | Overall heat loss coefficient |
| `c_room` | 15 MJ/K | Effective thermal mass |
| `radiator_ua` | 500 W/K | Radiator heat transfer |

---

## 12. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
# Requires: casadi, numpy, pandas, scipy, matplotlib, requests, gymnasium, pvlib
```

### Basic run (7 days, synthetic data)
```bash
python run_mpc.py
```

### Custom building, 14 days, with plots
```bash
python run_mpc.py --building sfh_2016_now_2_kfw --days 14 --plot
```

### Live data (requires internet)
```bash
python run_mpc.py --live --days 3 --plot
```

### Evaluate a previous run
```bash
python evaluate.py --results_dir runs/mpc_sfh_1984_1994_1_enev_20240429_120000
```

### Output files
Each run creates a directory in `runs/` containing:
- `config.json` — run parameters
- `summary.json` — total cost, energy, comfort violations, avg COP
- `trajectory.csv` — timestep-by-timestep log of all variables
- `mpc_evaluation.png` — 5-panel evaluation plot (if `--plot`)
- `daily_breakdown.png` — per-day energy, cost, tank temp, COP

---

## 13. Mathematical Formulation

### COP Model (Carnot-based)

```
COP(T_tank, T_amb) = η_carnot × (T_tank + 273.15) / max(T_tank − T_amb + 5, 5)
COP = min(COP, 7.0)     ← physical upper limit
COP = max(COP, 1.0)     ← numerical safety floor
```

The `+5` offset accounts for the temperature difference between refrigerant and working fluid (evaporator/condenser pinch).

### Tank Energy Balance

```
C_tank × dT_tank/dt = Q_HP − Q_house − UA_tank × (T_tank − T_basement)
```

Discretised (Forward Euler, Δt = 900s):
```
T_tank[k+1] = T_tank[k] + Δt/C_tank × (Q_HP[k] − Q_house[k] − Q_loss[k])
```

### Room Energy Balance (simplified 1R1C for MPC)

```
C_room × dT_room/dt = UA_bld × (T_amb − T_room) + Q_house + Q_solar + Q_internal
```

Discretised:
```
T_room[k+1] = T_room[k] + Δt/C_room × (Q_wall[k] + Q_house[k] + Q_solar[k] + Q_int[k])
```

### Electrical Power

```
P_el[k] = Q_HP[k] / COP[k]
```

### Electricity Cost (objective term 1)

```
Cost = Σ_k  price[k] × P_el[k] × Δt / 3.6×10⁶    [€]
```

(Division by 3.6×10⁶ converts W·s to kWh)

### Comfort Penalty (objective term 2)

Soft constraint via slack variable:
```
T_room_min − slack ≤ T_room ≤ T_room_max + slack
Penalty = w_comfort × slack²
```

The quadratic penalty ensures small violations are tolerated but large ones are heavily penalised.

### Cycling Penalty (objective term 3)

```
Cycling = w_cycling × (Q_HP[k] − Q_HP[k−1])²
```

Penalises rapid changes in heat pump output to reduce compressor wear.
