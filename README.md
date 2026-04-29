# Heat Pump Hierarchical MPC — Quickstart Guide

This guide explains how to set up, configure, and run the Hierarchical Model Predictive Control (MPC) simulation for the residential heat pump system.

## 1. Prerequisites & Installation

The system requires Python 3.10+ and several scientific computing libraries (including CasADi for optimization). It is highly recommended to use a virtual environment.

### Step 1: Clone the repository
(If you haven't already, clone the main repo and ensure the `i4b` submodule is pulled).

### Step 2: Create a virtual environment
Open your terminal in the project directory (`dev/`) and run:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Running a Simulation

Unlike Reinforcement Learning, **Model Predictive Control does not require a "training" phase**. The mathematical optimizer solves the problem live at every simulation step. 

To run a simulation and generate evaluation plots, use the `run_mpc.py` script.

### Basic 3-Day Run (Synthetic Data)
```bash
python run_mpc.py --days 3 --plot
```
*This uses synthetic German duck-curve electricity prices and typical seasonal weather.*

### Live Data Run (Requires Internet)
```bash
python run_mpc.py --live --days 3 --plot
```
*This fetches real day-ahead prices from the aWATTar API and live weather forecasts from the BrightSky API.*

### Customizing the Building
You can specify the `i4b` TABULA building string. For example, to run a modern KfW 2016 standard house:
```bash
python run_mpc.py --building sfh_2016_now_2_kfw --days 3 --plot
```

---

## 3. Where to Find the Results

Every time you run the simulation, a new timestamped folder is created inside the `runs/` directory (e.g., `runs/mpc_sfh_1984_1994_1_enev_20260429_144105/`).

Inside this folder you will find:
1.  **`summary.json`**: A text file containing the total cost (€), total energy (kWh), and average temperatures.
2.  **`trajectory.csv`**: A spreadsheet of every single 15-minute timestep (temperatures, prices, heat pump power).
3.  **`mpc_evaluation.png`**: A highly detailed 5-panel graph showing how the MPC behaved (temperatures, actuator commands, electricity prices, and cumulative cost).
4.  **`daily_breakdown.png`**: A bar chart summarizing cost and energy usage per day.

---

## 4. How to "Tune" the MPC (Improving Comfort or Cost)

If the simulation results show that the house is too cold (Comfort Violations) or that it is spending too much money, you need to "tune" the mathematical weights. 

Open `config.py` and find the `MPC_CONFIG` section:

```python
MPC_CONFIG = {
    # ... other settings ...
    "w_electricity": 1.0,     # Weight for minimizing € cost
    "w_comfort": 100.0,       # Penalty for dropping below 20°C
    "w_cycling": 1e-6,        # Penalty for turning the compressor on/off
}
```

### Scenario A: The house is too cold (e.g., 19.3°C)
The optimizer thinks saving €1.00 is worth letting the house get cold.
*   **Fix:** Increase `w_comfort` to `1000.0` or `5000.0`. The MPC will now aggressively heat the house to stay above 20°C, even if electricity is expensive.

### Scenario B: The system is ignoring electricity prices
The optimizer is keeping the house perfectly at 21°C but buying power during the expensive evening peak.
*   **Fix:** Increase `w_electricity` to `10.0` or `50.0`. The MPC will now try to pre-heat the buffer tank overnight when prices are cheap, and let the tank drain during the evening peak.

### Scenario C: The Heat Pump turns on and off too rapidly
*   **Fix:** Increase `w_cycling` to `0.01` or `0.1`. This heavily penalizes changing the heat pump output, forcing it to run smoothly and constantly at a lower power.
