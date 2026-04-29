"""
Hierarchical MPC Controller — Economic Optimiser + Tracking Controller.

Upper Layer (EconomicMPC):
    Solves a 24-hour non-linear programme every hour to find the
    cost-optimal heating strategy given aWATTar electricity prices,
    weather forecasts, and the building + tank thermal model.

Lower Layer (TrackingController):
    Fast PI-based controller running every 15 minutes.  Tracks the
    reference trajectories from the Upper Layer and converts them
    into physical actuator commands (heat pump power, mixing valve).

Dependencies:
    CasADi >= 3.6  (for the NLP formulation)
    IPOPT solver   (bundled with CasADi)
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple

import casadi as ca

from config import (
    MPC_CONFIG,
    TANK_CONFIG,
    HEATPUMP_CONFIG,
    BUILDING_MODEL_CONFIG,
)


# ============================================================================
# Upper Layer — Economic MPC (24 h horizon, 1 h step)
# ============================================================================

class EconomicMPC:
    """Cost-minimising MPC over a 24-hour receding horizon.

    Uses a simplified 1R1C building model and a well-mixed tank model
    to formulate a non-linear programme (NLP) solved by IPOPT.

    The optimiser decides:
        - Q_HP[k]:    thermal power from heat pump into tank  [W]
        - Q_house[k]: thermal power from tank into house      [W]

    And outputs reference trajectories:
        - T_tank_ref[k]:  optimal tank temperature trajectory
        - T_room_ref[k]:  optimal room temperature trajectory
    """

    def __init__(self, config: Dict = None):
        cfg = config or {}
        mpc = MPC_CONFIG
        tank = TANK_CONFIG
        hp = HEATPUMP_CONFIG
        bld = BUILDING_MODEL_CONFIG

        # Horizon
        self.N = cfg.get("horizon", mpc["upper_horizon_hours"])
        self.dt = cfg.get("dt", mpc["upper_step_seconds"])

        # Building (simplified 1R1C)
        self.C_room = bld["c_room"]             # J/K
        self.UA_bld = bld["ua_building"]         # W/K
        self.UA_rad = bld["radiator_ua"]         # W/K

        # Tank
        rho, cp = 988.0, 4182.0
        V = tank["volume_liters"] / 1000.0
        self.C_tank = rho * cp * V               # J/K
        self.UA_tank = tank["ua_value"]           # W/K
        self.T_basement = tank["t_ambient"]       # °C
        self.T_tank_min = tank["t_min"]
        self.T_tank_max = tank["t_max"]

        # Heat pump
        self.Q_HP_max = hp["q_max_w"]             # W
        self.eta_carnot = hp["carnot_efficiency"]

        # Comfort
        self.T_room_min = mpc["room_t_min"]
        self.T_room_max = mpc["room_t_max"]

        # Weights
        self.w_elec = mpc["w_electricity"]
        self.w_comfort = mpc["w_comfort"]
        self.w_cycling = mpc["w_cycling"]

        # Build the symbolic optimisation problem once
        self._build_nlp()

    # ────────────────────────────────────────────────────────────────────
    def _build_nlp(self):
        """Construct the CasADi NLP (called once at init)."""
        N = self.N
        opti = ca.Opti()

        # ── Decision variables ──────────────────────────────────────────
        T_room = opti.variable(N + 1)
        T_tank = opti.variable(N + 1)
        Q_HP   = opti.variable(N)
        Q_house = opti.variable(N)
        slack  = opti.variable(N)   # comfort relaxation

        # ── Parameters (filled at each solve) ───────────────────────────
        T_room_0  = opti.parameter()
        T_tank_0  = opti.parameter()
        p_T_amb   = opti.parameter(N)
        p_price   = opti.parameter(N)   # €/kWh
        p_Q_solar = opti.parameter(N)   # W
        p_Q_int   = opti.parameter(N)   # W

        # ── Dynamics ────────────────────────────────────────────────────
        cost = 0.0
        for k in range(N):
            # COP model (Carnot-based, clamped)
            dT_lift = T_tank[k] - p_T_amb[k]
            COP = self.eta_carnot * (T_tank[k] + 273.15) / ca.fmax(dT_lift + 5.0, 5.0)
            COP = ca.fmin(COP, 7.0)
            P_el = Q_HP[k] / ca.fmax(COP, 1.0)  # W electrical

            # Room energy balance (1R1C)
            Q_wall = self.UA_bld * (p_T_amb[k] - T_room[k])
            T_room_next = T_room[k] + self.dt / self.C_room * (
                Q_wall + Q_house[k] + p_Q_solar[k] + p_Q_int[k]
            )
            opti.subject_to(T_room[k + 1] == T_room_next)

            # Tank energy balance
            Q_loss = self.UA_tank * (T_tank[k] - self.T_basement)
            T_tank_next = T_tank[k] + self.dt / self.C_tank * (
                Q_HP[k] - Q_house[k] - Q_loss
            )
            opti.subject_to(T_tank[k + 1] == T_tank_next)

            # ── Constraints ─────────────────────────────────────────────
            opti.subject_to(Q_HP[k] >= 0)
            opti.subject_to(Q_HP[k] <= self.Q_HP_max)
            opti.subject_to(Q_house[k] >= 0)
            # Can only extract heat if tank is hotter than room
            opti.subject_to(Q_house[k] <= self.UA_rad * ca.fmax(T_tank[k] - T_room[k], 0))
            opti.subject_to(T_tank[k + 1] >= self.T_tank_min)
            opti.subject_to(T_tank[k + 1] <= self.T_tank_max)
            # Comfort with soft constraint (slack)
            opti.subject_to(T_room[k + 1] >= self.T_room_min - slack[k])
            opti.subject_to(T_room[k + 1] <= self.T_room_max + slack[k])
            opti.subject_to(slack[k] >= 0)

            # ── Cost terms ──────────────────────────────────────────────
            # 1. Electricity cost  [€]
            cost += self.w_elec * p_price[k] * P_el * self.dt / 3.6e6

            # 2. Comfort violation penalty
            cost += self.w_comfort * slack[k] ** 2

            # 3. HP cycling penalty (rate of change)
            if k > 0:
                cost += self.w_cycling * (Q_HP[k] - Q_HP[k - 1]) ** 2

        # ── Initial conditions ──────────────────────────────────────────
        opti.subject_to(T_room[0] == T_room_0)
        opti.subject_to(T_tank[0] == T_tank_0)

        # ── Solver ──────────────────────────────────────────────────────
        opti.minimize(cost)
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-4,
            "print_time": False,
        }
        opti.solver("ipopt", opts)

        # ── Store handles ───────────────────────────────────────────────
        self._opti = opti
        self._var = {
            "T_room": T_room, "T_tank": T_tank,
            "Q_HP": Q_HP, "Q_house": Q_house, "slack": slack,
        }
        self._par = {
            "T_room_0": T_room_0, "T_tank_0": T_tank_0,
            "T_amb": p_T_amb, "price": p_price,
            "Q_solar": p_Q_solar, "Q_int": p_Q_int,
        }

    # ────────────────────────────────────────────────────────────────────
    def solve(
        self,
        T_room_now: float,
        T_tank_now: float,
        prices_eur_kwh: np.ndarray,
        T_amb_forecast: np.ndarray,
        Q_solar_forecast: np.ndarray = None,
        Q_internal_forecast: np.ndarray = None,
    ) -> Dict:
        """Solve the economic MPC for the next horizon.

        Parameters
        ----------
        T_room_now : float
            Current room temperature [°C].
        T_tank_now : float
            Current tank temperature [°C].
        prices_eur_kwh : array (N,)
            Electricity price forecast [€/kWh].
        T_amb_forecast : array (N,)
            Ambient temperature forecast [°C].
        Q_solar_forecast : array (N,), optional
            Solar gains forecast [W].  Defaults to zeros.
        Q_internal_forecast : array (N,), optional
            Internal gains forecast [W].  Defaults to 200 W.

        Returns
        -------
        dict with keys:
            T_tank_ref : array (N+1,)
            T_room_ref : array (N+1,)
            Q_HP_ref   : array (N,)
            Q_house_ref: array (N,)
            P_el_ref   : array (N,)   estimated electrical power [W]
            cost_eur   : float        predicted total electricity cost
            status     : str          'optimal' or 'fallback'
        """
        N = self.N

        # Pad / trim forecasts to exactly N steps
        prices = self._pad(prices_eur_kwh, N, default=0.30)
        T_amb  = self._pad(T_amb_forecast, N, default=5.0)
        Q_sol  = self._pad(Q_solar_forecast, N, default=0.0) if Q_solar_forecast is not None else np.zeros(N)
        Q_int  = self._pad(Q_internal_forecast, N, default=200.0) if Q_internal_forecast is not None else np.full(N, 200.0)

        # Set parameters
        self._opti.set_value(self._par["T_room_0"], T_room_now)
        self._opti.set_value(self._par["T_tank_0"], T_tank_now)
        self._opti.set_value(self._par["T_amb"], T_amb)
        self._opti.set_value(self._par["price"], prices)
        self._opti.set_value(self._par["Q_solar"], Q_sol)
        self._opti.set_value(self._par["Q_int"], Q_int)

        # Warm-start with a simple initial guess
        self._opti.set_initial(self._var["T_room"], T_room_now)
        self._opti.set_initial(self._var["T_tank"], T_tank_now)
        self._opti.set_initial(self._var["Q_HP"], self.Q_HP_max * 0.3)
        self._opti.set_initial(self._var["Q_house"], 2000.0)
        self._opti.set_initial(self._var["slack"], 0.0)

        try:
            sol = self._opti.solve()
            T_tank_ref = np.array(sol.value(self._var["T_tank"])).flatten()
            T_room_ref = np.array(sol.value(self._var["T_room"])).flatten()
            Q_HP_ref   = np.array(sol.value(self._var["Q_HP"])).flatten()
            Q_house_ref = np.array(sol.value(self._var["Q_house"])).flatten()

            # Estimate electrical consumption
            P_el_ref = np.zeros(N)
            for k in range(N):
                dT = max(T_tank_ref[k] - T_amb[k], 5.0)
                cop = min(self.eta_carnot * (T_tank_ref[k] + 273.15) / (dT + 5.0), 7.0)
                cop = max(cop, 1.0)
                P_el_ref[k] = Q_HP_ref[k] / cop

            cost_eur = float(np.sum(prices * P_el_ref * self.dt / 3.6e6))
            status = "optimal"

        except RuntimeError:
            warnings.warn("IPOPT failed — using fallback strategy.")
            T_tank_ref, T_room_ref, Q_HP_ref, Q_house_ref, P_el_ref = (
                self._fallback(T_room_now, T_tank_now, prices, T_amb)
            )
            cost_eur = float(np.sum(prices * P_el_ref * self.dt / 3.6e6))
            status = "fallback"

        return {
            "T_tank_ref": T_tank_ref,
            "T_room_ref": T_room_ref,
            "Q_HP_ref": Q_HP_ref,
            "Q_house_ref": Q_house_ref,
            "P_el_ref": P_el_ref,
            "cost_eur": cost_eur,
            "status": status,
        }

    # ────────────────────────────────────────────────────────────────────
    def _fallback(self, T_room, T_tank, prices, T_amb):
        """Simple rule-based fallback if IPOPT fails."""
        N = self.N
        T_tank_ref = np.full(N + 1, T_tank)
        T_room_ref = np.full(N + 1, T_room)
        Q_HP_ref = np.zeros(N)
        Q_house_ref = np.zeros(N)

        median_price = np.median(prices)
        for k in range(N):
            # Heat during cheap hours
            if prices[k] < median_price and T_tank < self.T_tank_max - 5:
                Q_HP_ref[k] = self.Q_HP_max * 0.7
            # Always supply some heat if room is cold
            if T_room < self.T_room_min:
                Q_house_ref[k] = 3000.0

        dT = np.maximum(T_tank_ref[:-1] - T_amb, 5.0)
        cop = np.minimum(self.eta_carnot * (T_tank_ref[:-1] + 273.15) / (dT + 5.0), 7.0)
        cop = np.maximum(cop, 1.0)
        P_el_ref = Q_HP_ref / cop

        return T_tank_ref, T_room_ref, Q_HP_ref, Q_house_ref, P_el_ref

    @staticmethod
    def _pad(arr, length, default=0.0):
        """Pad or trim an array to exactly `length`."""
        if arr is None:
            return np.full(length, default)
        arr = np.asarray(arr, dtype=float).flatten()
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)), constant_values=default)


# ============================================================================
# Lower Layer — Tracking Controller (PI-based, 15-min step)
# ============================================================================

class TrackingController:
    """Fast tracking controller that converts MPC reference trajectories
    into physical actuator commands.

    Outputs
    -------
    q_hp_w : float
        Thermal power command to the heat pump [W].
    supply_temp : float
        Supply temperature to the house heating loop [°C].
        Determined by mixing valve position on the buffer tank.
    """

    def __init__(self, config: Dict = None):
        cfg = config or {}
        hp = HEATPUMP_CONFIG

        # PI gains for tank temperature tracking
        self.Kp_tank = cfg.get("Kp_tank", 800.0)   # W per °C error
        self.Ki_tank = cfg.get("Ki_tank", 50.0)     # W per °C·s error

        # PI gains for room temperature tracking
        self.Kp_room = cfg.get("Kp_room", 0.15)     # valve fraction per °C
        self.Ki_room = cfg.get("Ki_room", 0.005)

        # Limits
        self.Q_HP_max = hp["q_max_w"]
        self.T_supply_min = hp["supply_temp_min"]
        self.T_supply_max = hp["supply_temp_max"]

        # Integrator states
        self._e_tank_int = 0.0
        self._e_room_int = 0.0

    def compute(
        self,
        T_tank: float,
        T_room: float,
        T_tank_ref: float,
        T_room_ref: float,
        Q_HP_ref: float,
        Q_house_ref: float,
        dt: float,
    ) -> Dict:
        """Compute actuator commands for one lower-layer timestep.

        Parameters
        ----------
        T_tank, T_room : float
            Current measured temperatures [°C].
        T_tank_ref, T_room_ref : float
            Reference temperatures from the Upper Layer [°C].
        Q_HP_ref, Q_house_ref : float
            Reference powers from the Upper Layer [W].
        dt : float
            Timestep [s].

        Returns
        -------
        dict
            q_hp_w:      heat pump thermal power command [W]
            supply_temp: supply temperature to house [°C]
            valve_pos:   mixing valve position [0–1]
        """
        # ── Heat pump control (track T_tank_ref) ────────────────────────
        e_tank = T_tank_ref - T_tank
        self._e_tank_int += e_tank * dt
        self._e_tank_int = np.clip(self._e_tank_int, -1e5, 1e5)

        q_hp_fb = self.Kp_tank * e_tank + self.Ki_tank * self._e_tank_int
        q_hp_w = np.clip(Q_HP_ref + q_hp_fb, 0, self.Q_HP_max)

        # ── Valve control (track T_room_ref) ────────────────────────────
        e_room = T_room_ref - T_room
        self._e_room_int += e_room * dt
        self._e_room_int = np.clip(self._e_room_int, -100, 100)

        valve_pos = np.clip(
            0.5 + self.Kp_room * e_room + self.Ki_room * self._e_room_int,
            0.0, 1.0,
        )

        # Supply temperature from mixing valve
        T_return = T_room + 5.0  # approximate return temp
        supply_temp = valve_pos * T_tank + (1 - valve_pos) * T_return
        supply_temp = np.clip(supply_temp, self.T_supply_min, self.T_supply_max)

        # Can't supply hotter than tank
        supply_temp = min(supply_temp, T_tank)

        return {
            "q_hp_w": float(q_hp_w),
            "supply_temp": float(supply_temp),
            "valve_pos": float(valve_pos),
        }

    def reset(self):
        """Reset integrator states."""
        self._e_tank_int = 0.0
        self._e_room_int = 0.0
