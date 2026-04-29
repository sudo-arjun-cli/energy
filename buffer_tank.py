"""
Buffer Tank — Thermal Energy Storage (TES) Model.

A single-node (well-mixed) hot water buffer tank that acts as the
thermal battery between the heat pump and the building heating system.

The tank decouples heat generation from heat distribution, enabling
the MPC to run the heat pump during cheap-price hours and store
the energy for later use.

Physics:
    C_tank · dT/dt = Q_HP − Q_house − Q_loss

    where:
        C_tank  = ρ · cp · V        [J/K]     thermal capacity
        Q_HP    = heat added by HP   [W]       controlled by Upper Layer
        Q_house = heat to building   [W]       controlled by Lower Layer
        Q_loss  = UA · (T − T_amb)   [W]       standby losses
"""

import numpy as np
from typing import Dict, Optional

from config import TANK_CONFIG


class BufferTank:
    """Single-node (well-mixed) hot water buffer tank model.

    This simplified model assumes uniform temperature throughout the
    tank (no stratification). Suitable for control-oriented MPC
    applications where computational speed is prioritised.

    Parameters
    ----------
    volume_liters : float
        Tank volume in litres.
    ua_value : float
        Standby heat loss coefficient [W/K].
    t_ambient : float
        Temperature of the tank's surroundings (e.g. basement) [°C].
    t_min : float
        Minimum allowed tank temperature [°C].
    t_max : float
        Maximum allowed tank temperature [°C].
    t_init : float
        Initial tank temperature [°C].
    """

    # Water properties (at ~50 °C)
    RHO = 988.0     # kg/m³  density
    CP = 4182.0     # J/(kg·K)  specific heat capacity

    def __init__(
        self,
        volume_liters: float = None,
        ua_value: float = None,
        t_ambient: float = None,
        t_min: float = None,
        t_max: float = None,
        t_init: float = None,
    ):
        cfg = TANK_CONFIG
        self.volume = (volume_liters or cfg["volume_liters"]) / 1000.0  # → m³
        self.ua = ua_value if ua_value is not None else cfg["ua_value"]
        self.t_ambient = t_ambient if t_ambient is not None else cfg["t_ambient"]
        self.t_min = t_min if t_min is not None else cfg["t_min"]
        self.t_max = t_max if t_max is not None else cfg["t_max"]

        # Derived
        self.mass = self.RHO * self.volume          # kg
        self.C = self.mass * self.CP                # J/K  (thermal capacity)

        # State
        self.temperature = t_init if t_init is not None else cfg["t_init"]

        # Logging accumulators (reset each episode)
        self._total_q_in_j = 0.0
        self._total_q_out_j = 0.0
        self._total_q_loss_j = 0.0

    # ────────────────────────────────────────────────────────────────────
    # Core API
    # ────────────────────────────────────────────────────────────────────

    def step(self, q_in_w: float, q_out_w: float, dt: float) -> Dict:
        """Advance the tank state by one timestep.

        Parameters
        ----------
        q_in_w : float
            Thermal power added by the heat pump [W].
        q_out_w : float
            Thermal power extracted for house heating [W].
        dt : float
            Timestep duration [s].

        Returns
        -------
        dict
            Tank state after the step:
            - T_tank:  new temperature [°C]
            - Q_loss:  heat lost to surroundings this step [W]
            - Q_in:    heat added [W]
            - Q_out:   heat extracted [W]
        """
        q_loss_w = self.heat_loss_rate()

        # Energy balance: C · dT = (Q_in − Q_out − Q_loss) · dt
        dT = (q_in_w - q_out_w - q_loss_w) * dt / self.C
        self.temperature += dT

        # Enforce physical bounds
        self.temperature = np.clip(self.temperature, self.t_min, self.t_max)

        # Accumulate for logging
        self._total_q_in_j += q_in_w * dt
        self._total_q_out_j += q_out_w * dt
        self._total_q_loss_j += q_loss_w * dt

        return {
            "T_tank": self.temperature,
            "Q_loss_w": q_loss_w,
            "Q_in_w": q_in_w,
            "Q_out_w": q_out_w,
        }

    def heat_loss_rate(self) -> float:
        """Current standby heat loss rate [W]."""
        return self.ua * max(self.temperature - self.t_ambient, 0.0)

    def stored_energy_kwh(self, t_ref: float = None) -> float:
        """Usable energy stored above a reference temperature [kWh].

        Parameters
        ----------
        t_ref : float, optional
            Reference temperature [°C].  Defaults to t_min.
        """
        t_ref = t_ref if t_ref is not None else self.t_min
        dT = max(self.temperature - t_ref, 0.0)
        return self.C * dT / 3.6e6  # J → kWh

    def reset(self, t_init: float = None):
        """Reset the tank to initial conditions."""
        self.temperature = t_init if t_init is not None else TANK_CONFIG["t_init"]
        self._total_q_in_j = 0.0
        self._total_q_out_j = 0.0
        self._total_q_loss_j = 0.0

    def get_summary(self) -> Dict:
        """Return cumulative energy accounting [kWh]."""
        return {
            "T_tank": self.temperature,
            "total_q_in_kwh": self._total_q_in_j / 3.6e6,
            "total_q_out_kwh": self._total_q_out_j / 3.6e6,
            "total_q_loss_kwh": self._total_q_loss_j / 3.6e6,
            "stored_energy_kwh": self.stored_energy_kwh(),
        }

    def __repr__(self):
        return (
            f"BufferTank(V={self.volume*1000:.0f}L, "
            f"T={self.temperature:.1f}°C, "
            f"C={self.C/1e6:.2f} MJ/K, "
            f"UA={self.ua:.1f} W/K)"
        )
