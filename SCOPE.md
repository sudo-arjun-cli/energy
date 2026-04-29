# Project Scope Document

> **Project Name:** Intelligent Heat Pump Control via Hierarchical MPC
> **Version:** 2.0 (Transitioned from RL to MPC)
> **Date:** April 2026

---

## 1. Executive Summary

This project aims to develop, simulate, and evaluate an intelligent, price-aware control system for residential air-source heat pumps in Germany. By utilizing a Hierarchical Model Predictive Control (MPC) architecture combined with a hot water buffer tank, the system acts as a flexible thermal battery. It proactively shifts electricity consumption to periods of low spot-market prices or high renewable energy availability, thereby minimizing operating costs while strictly maintaining occupant thermal comfort.

## 2. Core Objectives

*   **Cost Minimization:** Reduce the overall electricity cost of operating a residential heat pump by exploiting day-ahead spot market price volatility.
*   **Comfort Guarantee:** Maintain indoor temperatures within a predefined comfort band (e.g., 20°C - 23°C) regardless of external weather conditions.
*   **System Longevity:** Prevent compressor short-cycling by incorporating wear-penalties into the optimization algorithm.
*   **Decoupled Operation:** Effectively decouple heat generation from heat distribution using a thermal buffer tank model.
*   **Open-Box Optimization:** Transition from black-box Machine Learning (RL) to an interpretable, physics-based optimization framework using CasADi and IPOPT.

## 3. Scope of Work

### 3.1 In Scope

*   **Control Architecture:**
    *   Implementation of an Upper Layer **Economic MPC** (24-hour horizon, 1-hour steps) to solve the economic scheduling problem via Non-Linear Programming (NLP).
    *   Implementation of a Lower Layer **Tracking Controller** (PI loops, 15-minute steps) to translate reference trajectories into physical actuator commands (heat pump power, mixing valve position).
*   **Physical Modeling:**
    *   Development of a single-node, well-mixed thermodynamic model for a 500L buffer tank (`buffer_tank.py`).
    *   Integration with the `i4b` (Intelligence for Buildings) framework for high-fidelity 4R3C RC-network simulation of German residential buildings (TABULA dataset).
    *   Implementation of a dynamic, temperature-dependent Coefficient of Performance (COP) model for the air-source heat pump.
*   **Data Pipelines:**
    *   Live integration with the **aWATTar API** for day-ahead electricity spot prices.
    *   Live integration with the **BrightSky API** (DWD) for weather forecasts (temperature, solar irradiance).
    *   Robust fallback mechanisms featuring synthetic data generators for offline development and testing.
*   **Simulation & Evaluation:**
    *   A closed-loop simulation harness (`simulation.py`) that steps through time, solving the MPC and updating physical states.
    *   Automated evaluation scripts (`evaluate.py`) that generate comprehensive 5-panel performance dashboards (Temperature, Power, Cost, COP).

### 3.2 Out of Scope

*   **Hardware Deployment:** The project is strictly a software simulation and research framework. Deployment to physical PLCs, microcontrollers, or real-world heat pump hardware is not included.
*   **Advanced Stratification Models:** The buffer tank is modeled as a single well-mixed volume. Multi-node stratified tank models are excluded to maintain computational efficiency in the NLP solver.
*   **Multi-Zone Control:** The building is modeled as a single aggregated thermal zone. Individual room control (e.g., smart thermostatic radiator valves per room) is out of scope.
*   **Additional Energy Assets:** Integration of Photovoltaic (PV) generation models or stationary battery storage systems is not currently supported (though the architecture could be extended).
*   **Reinforcement Learning:** The previous RL/PPO implementation (v1) is considered legacy and is no longer actively maintained or improved within the core scope.

## 4. Key Deliverables

1.  **Simulation Engine:** The core Python codebase (`mpc_controller.py`, `buffer_tank.py`, `simulation.py`).
2.  **Data Ingestion Module:** `data_pipeline.py` for API connections and synthetic data generation.
3.  **Configuration System:** Centralized `config.py` defining all physical and economic parameters.
4.  **Documentation:** Detailed markdown documentation including `ARCHITECTURE.md`, `DIAGRAMS.md`, and this `SCOPE.md`.
5.  **Visualization Tools:** `evaluate.py` for generating performance plots and cost summaries.

## 5. Technology Stack

*   **Language:** Python 3.10+
*   **Optimization:** CasADi (with bundled IPOPT solver)
*   **Simulation Framework:** `i4b` (Intelligence for Buildings) / `gymnasium`
*   **Data Manipulation:** `pandas`, `numpy`, `scipy`
*   **Visualization:** `matplotlib`
*   **External APIs:** `requests` (aWATTar, BrightSky)

## 6. Target Audience

*   Researchers and academics studying Demand Response and Model Predictive Control in HVAC systems.
*   Energy engineers developing smart-grid ready building automation systems.
*   Developers looking for a reference architecture for price-aware heating control.
