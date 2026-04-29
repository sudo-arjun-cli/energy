# Technical Diagrams — Hierarchical MPC Heat Pump Controller

> Visual reference for the system architecture, control flows, physics models, and data paths.
> Companion to [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table of Contents

1. [Full System Architecture](#1-full-system-architecture)
2. [Control Hierarchy](#2-control-hierarchy)
3. [Physical System Schematic](#3-physical-system-schematic)
4. [Simulation Loop Flowchart](#4-simulation-loop-flowchart)
5. [Upper Layer — Economic MPC Detail](#5-upper-layer--economic-mpc-detail)
6. [Lower Layer — Tracking Controller Detail](#6-lower-layer--tracking-controller-detail)
7. [Data Pipeline](#7-data-pipeline)
8. [Buffer Tank Thermal Model](#8-buffer-tank-thermal-model)
9. [COP Behaviour](#9-cop-behaviour)
10. [Module Dependency Graph](#10-module-dependency-graph)
11. [Timing Diagram](#11-timing-diagram)

---

## 1. Full System Architecture

```mermaid
graph TB
    subgraph External["☁️ External Data Sources"]
        AW["💰 aWATTar API<br/>Electricity Spot Prices<br/>24h ahead, hourly"]
        BS["☀️ BrightSky API<br/>DWD Weather Data<br/>Temperature, Solar, Wind"]
        SY["🔄 Synthetic Fallback<br/>Offline Mode<br/>No API needed"]
    end

    subgraph Pipeline["📊 Data Pipeline — data_pipeline.py"]
        AWC["AWattarClient<br/>fetch_day_ahead_prices()"]
        BSC["BrightSkyClient<br/>fetch_weather()"]
        SYN["generate_synthetic_weather()<br/>generate_synthetic_prices()"]
        DP["DataPipeline<br/>get_training_data()"]
    end

    subgraph Upper["🧠 Upper Layer — Economic MPC"]
        NLP["CasADi NLP Formulation<br/>24 variables × 24 steps"]
        IPOPT["IPOPT Solver<br/>Interior Point Optimiser"]
        FB["Rule-Based Fallback<br/>if solver fails"]
        REF["Reference Trajectories<br/>T_tank_ref, T_room_ref<br/>Q_HP_ref, Q_house_ref"]
    end

    subgraph Lower["⚡ Lower Layer — Tracking Controller"]
        PI_T["PI Controller<br/>Tank Temperature Loop<br/>Kp=800, Ki=50"]
        PI_R["PI Controller<br/>Room Temperature Loop<br/>Kp=0.15, Ki=0.005"]
        CMD["Actuator Commands<br/>q_hp_w, supply_temp<br/>valve_pos"]
    end

    subgraph Plant["🏠 Physical Plant"]
        HP["Heat Pump<br/>8 kW thermal<br/>Air-Source"]
        TANK["Buffer Tank<br/>500L, 30-80°C<br/>UA = 3.0 W/K"]
        VALVE["3-Way Mixing Valve<br/>Controls T_supply"]
        HOUSE["Building<br/>i4b 4R3C Model<br/>TABULA SFH"]
    end

    AW --> AWC
    BS --> BSC
    SY --> SYN
    AWC & BSC & SYN --> DP

    DP -->|"prices[24h]<br/>T_amb[24h]<br/>Q_solar[24h]"| NLP
    NLP --> IPOPT
    IPOPT -->|"optimal"| REF
    IPOPT -->|"failed"| FB
    FB --> REF

    REF -->|"T_tank_ref<br/>T_room_ref"| PI_T & PI_R
    PI_T -->|"q_hp_w"| CMD
    PI_R -->|"valve_pos"| CMD

    CMD -->|"Power ON/OFF"| HP
    CMD -->|"Position 0-1"| VALVE
    HP -->|"Q_HP [W]"| TANK
    TANK -->|"Hot Water"| VALVE
    VALVE -->|"T_supply"| HOUSE

    HOUSE -->|"T_room"| PI_R
    TANK -->|"T_tank"| PI_T
    HOUSE -->|"T_room"| NLP
    TANK -->|"T_tank"| NLP

    style External fill:#1a1a2e,stroke:#16213e,color:#e8e8e8
    style Pipeline fill:#0f3460,stroke:#16213e,color:#e8e8e8
    style Upper fill:#533483,stroke:#16213e,color:#e8e8e8
    style Lower fill:#e94560,stroke:#16213e,color:#e8e8e8
    style Plant fill:#1b4332,stroke:#16213e,color:#e8e8e8
```

---

## 2. Control Hierarchy

```mermaid
graph LR
    subgraph L1["Layer 1 — Economic MPC<br/>SLOW (1 hour)"]
        direction TB
        OBJ["Objective: Minimise Cost<br/>min Σ price × P_el"]
        HOR["Horizon: 24 hours<br/>Step: 1 hour<br/>Solver: IPOPT"]
        MOD["Model: Simplified 1R1C<br/>+ Well-mixed Tank"]
        OUT1["Output: Reference<br/>Trajectories"]
        OBJ --> HOR --> MOD --> OUT1
    end

    subgraph L2["Layer 2 — Tracking Controller<br/>FAST (15 min)"]
        direction TB
        OBJ2["Objective: Track References<br/>min |T - T_ref|"]
        HOR2["Step: 15 minutes<br/>Controller: PI"]
        MOD2["Model: Direct Feedback<br/>from Sensors"]
        OUT2["Output: Actuator<br/>Commands"]
        OBJ2 --> HOR2 --> MOD2 --> OUT2
    end

    OUT1 -->|"T_tank_ref<br/>T_room_ref<br/>Q_HP_ref"| OBJ2
    OUT2 -->|"q_hp_w<br/>supply_temp<br/>valve_pos"| PLANT["🏠 Physical<br/>Plant"]
    PLANT -->|"T_room, T_tank<br/>T_amb"| OBJ
    PLANT -->|"T_room, T_tank"| OBJ2

    style L1 fill:#4a148c,stroke:#7b1fa2,color:#e8e8e8
    style L2 fill:#b71c1c,stroke:#c62828,color:#e8e8e8
    style PLANT fill:#1b5e20,stroke:#2e7d32,color:#e8e8e8
```

---

## 3. Physical System Schematic

```mermaid
graph LR
    subgraph Outdoor["🌡️ Outdoor"]
        AMB["Ambient Air<br/>T_amb"]
    end

    subgraph HeatPump["⚙️ Heat Pump"]
        EVAP["Evaporator<br/>extracts heat<br/>from outdoor air"]
        COMP["Compressor<br/>8 kW thermal max"]
        COND["Condenser<br/>heats water"]
    end

    subgraph Storage["🔋 Buffer Tank"]
        TANK["500L Tank<br/>30°C — 80°C<br/>C = 2.07 MJ/K"]
        LOSS["Standby Loss<br/>Q_loss = 3.0 × ΔT"]
    end

    subgraph Distribution["🏠 Heat Distribution"]
        VALVE["3-Way<br/>Mixing Valve"]
        RAD["Radiators<br/>UA = 500 W/K"]
        ROOM["Heated Space<br/>20°C — 23°C"]
    end

    AMB -->|"Low-grade<br/>heat"| EVAP
    EVAP --> COMP
    COMP -->|"P_el = Q_HP / COP"| COND
    COND -->|"Q_HP [W]<br/>hot water"| TANK
    TANK --> LOSS
    LOSS -->|"to basement"| AMB

    TANK -->|"T_tank"| VALVE
    ROOM -->|"T_return ≈ T_room+5"| VALVE
    VALVE -->|"T_supply =<br/>valve×T_tank +<br/>(1-valve)×T_return"| RAD
    RAD -->|"Q_house [W]"| ROOM

    style Outdoor fill:#e3f2fd,stroke:#1565c0,color:#000
    style HeatPump fill:#fce4ec,stroke:#c62828,color:#000
    style Storage fill:#fff3e0,stroke:#e65100,color:#000
    style Distribution fill:#e8f5e9,stroke:#2e7d32,color:#000
```

---

## 4. Simulation Loop Flowchart

```mermaid
flowchart TD
    START(["▶ Start Simulation"]) --> INIT

    INIT["Initialise:<br/>• BufferTank (500L, 45°C)<br/>• EconomicMPC (N=24)<br/>• TrackingController<br/>• i4b env (4R3C)<br/>• Load data"]

    INIT --> LOOP{"step < total_steps?"}

    LOOP -->|Yes| READ["Read State:<br/>T_room ← i4b info<br/>T_tank ← tank.temperature<br/>T_amb ← env.get_cur_T_amb()<br/>price ← prices_15min"]

    READ --> CHECK{"step % 4 == 0?<br/>(every hour)"}

    CHECK -->|"Yes — Re-plan"| UPPER["Upper Layer:<br/>1. Build 24h price forecast<br/>2. Build 24h weather forecast<br/>3. Solve NLP via IPOPT<br/>4. Get reference trajectories"]

    CHECK -->|"No — Use existing plan"| LOOKUP

    UPPER --> LOOKUP["Look Up Reference:<br/>T_tank_ref, T_room_ref<br/>Q_HP_ref for current step"]

    LOOKUP --> LOWER["Lower Layer (PI):<br/>e_tank = T_tank_ref − T_tank<br/>e_room = T_room_ref − T_room<br/>→ q_hp_w, supply_temp, valve_pos"]

    LOWER --> COP["Compute COP:<br/>COP = 0.45 × (T_tank+273)<br/>      / (T_tank−T_amb+5)<br/>q_el_w = q_hp_w / COP"]

    COP --> TANK_STEP["Step Buffer Tank:<br/>tank.step(q_hp, q_house, 900s)<br/>→ update T_tank"]

    TANK_STEP --> I4B["Step i4b Building:<br/>action = normalise(supply_temp)<br/>env.step(action)<br/>→ update T_room"]

    I4B --> LOG["Log & Account:<br/>energy_kwh, step_cost<br/>comfort violations"]

    LOG --> LOOP
    LOOP -->|No| SUMMARY["Print Summary:<br/>Total kWh, Total €<br/>Comfort violations<br/>Average COP"]

    SUMMARY --> SAVE["Save to runs/:<br/>summary.json<br/>trajectory.csv"]

    SAVE --> PLOT{"--plot flag?"}
    PLOT -->|Yes| EVAL["Generate 5-panel<br/>evaluation plot"]
    PLOT -->|No| END(["■ End"])
    EVAL --> END

    style START fill:#4caf50,color:#fff
    style END fill:#f44336,color:#fff
    style UPPER fill:#7b1fa2,color:#fff
    style LOWER fill:#c62828,color:#fff
    style TANK_STEP fill:#e65100,color:#fff
    style I4B fill:#1b5e20,color:#fff
```

---

## 5. Upper Layer — Economic MPC Detail

```mermaid
graph TB
    subgraph Inputs["📥 Inputs (Parameters)"]
        T0["T_room_0, T_tank_0<br/>Current state measurements"]
        PR["price[0..23]<br/>€/kWh from aWATTar"]
        WX["T_amb[0..23]<br/>Weather forecast"]
        SG["Q_solar[0..23], Q_int[0..23]<br/>Solar & internal gains"]
    end

    subgraph NLP["🔧 NLP Formulation (CasADi Opti)"]
        DV["Decision Variables:<br/>T_room[0..24], T_tank[0..24]<br/>Q_HP[0..23], Q_house[0..23]<br/>slack[0..23]"]
        DYN["Dynamics Constraints:<br/>Room: T[k+1] = T[k] + Δt/C × (Q_wall + Q_house + Q_sol + Q_int)<br/>Tank: T[k+1] = T[k] + Δt/C_tank × (Q_HP − Q_house − Q_loss)"]
        CON["Box Constraints:<br/>0 ≤ Q_HP ≤ 8000 W<br/>30°C ≤ T_tank ≤ 80°C<br/>20°C − slack ≤ T_room ≤ 23°C + slack<br/>Q_house ≤ UA_rad × (T_tank − T_room)"]
        OBJ["Objective:<br/>min Σ [ w_elec × price × P_el × Δt<br/>      + w_comfort × slack²<br/>      + w_cycling × (ΔQ_HP)² ]"]
    end

    subgraph Solver["⚡ IPOPT Solver"]
        SOLVE["Interior Point Method<br/>max 500 iterations<br/>tolerance 1e-4"]
        CHECK{"Converged?"}
        OPT["✅ Optimal Solution"]
        FALL["⚠️ Fallback:<br/>Heat during cheap hours<br/>Supply heat if T_room low"]
    end

    subgraph Outputs["📤 Outputs (References)"]
        REF["T_tank_ref[0..24]<br/>T_room_ref[0..24]<br/>Q_HP_ref[0..23]<br/>Q_house_ref[0..23]<br/>P_el_ref[0..23]<br/>cost_eur"]
    end

    T0 & PR & WX & SG --> DV
    DV --> DYN --> CON --> OBJ
    OBJ --> SOLVE
    SOLVE --> CHECK
    CHECK -->|Yes| OPT --> REF
    CHECK -->|No| FALL --> REF

    style Inputs fill:#0d47a1,color:#fff
    style NLP fill:#4a148c,color:#fff
    style Solver fill:#e65100,color:#fff
    style Outputs fill:#1b5e20,color:#fff
```

---

## 6. Lower Layer — Tracking Controller Detail

```mermaid
graph LR
    subgraph Refs["From Upper Layer"]
        TR["T_tank_ref"]
        RR["T_room_ref"]
        QR["Q_HP_ref"]
    end

    subgraph Measurements["From Sensors"]
        TT["T_tank (actual)"]
        TRM["T_room (actual)"]
    end

    subgraph TankLoop["Tank Temperature PI Loop"]
        ET["e_tank = T_tank_ref − T_tank"]
        PI_TANK["q_hp_fb = Kp×e + Ki×∫e·dt<br/>Kp=800 W/°C, Ki=50"]
        QHP["q_hp_w = clip(Q_HP_ref + q_hp_fb, 0, 8000)"]
    end

    subgraph RoomLoop["Room Temperature PI Loop"]
        ER["e_room = T_room_ref − T_room"]
        PI_ROOM["valve_fb = Kp×e + Ki×∫e·dt<br/>Kp=0.15/°C, Ki=0.005"]
        VP["valve_pos = clip(0.5 + valve_fb, 0, 1)"]
    end

    subgraph MixingCalc["Supply Temperature"]
        MIX["T_supply = valve × T_tank<br/>+ (1−valve) × (T_room+5)<br/>clip to 20-65°C<br/>cap at T_tank"]
    end

    TR --> ET
    TT --> ET
    ET --> PI_TANK --> QHP

    RR --> ER
    TRM --> ER
    ER --> PI_ROOM --> VP

    QR --> QHP
    VP --> MIX
    TT --> MIX

    QHP -->|"Heat Pump<br/>Power"| OUT(["To Plant"])
    MIX -->|"Supply<br/>Temperature"| OUT

    style TankLoop fill:#4a148c,color:#fff
    style RoomLoop fill:#c62828,color:#fff
    style MixingCalc fill:#e65100,color:#fff
```

---

## 7. Data Pipeline

```mermaid
graph TB
    subgraph APIs["External APIs"]
        AW_API["aWATTar API<br/>api.awattar.at/v1/marketdata<br/>No API key needed"]
        BS_API["BrightSky API<br/>api.brightsky.dev/weather<br/>No API key needed"]
    end

    subgraph Clients["API Clients"]
        AWC["AWattarClient"]
        BSC["BrightSkyClient"]
    end

    subgraph Processing["Data Processing"]
        TS["Timestamp Conversion<br/>ms-UNIX → Europe/Berlin"]
        RS["Resample to hourly<br/>.resample('h').mean().ffill()"]
        EN["Enrichment:<br/>price_rank (0-23)<br/>is_cheap_hour (bool)<br/>price_eur_kwh"]
    end

    subgraph Fallback["Synthetic Generators"]
        SP["generate_synthetic_prices()<br/>German duck-curve pattern<br/>base: 0.30 €/kWh"]
        SW["generate_synthetic_weather()<br/>Seasonal German climate<br/>Lat: 49.87°N (Darmstadt)"]
    end

    subgraph Output["Output DataFrames"]
        PDF["Price DataFrame<br/>index: datetime (Europe/Berlin)<br/>columns: price_eur_kwh,<br/>price_rank, is_cheap_hour"]
        WDF["Weather DataFrame<br/>index: datetime (Europe/Berlin)<br/>columns: T_amb, solar,<br/>wind_speed, cloud_cover"]
    end

    AW_API -->|"GET ?start=ms&end=ms"| AWC
    BS_API -->|"GET ?lat=&lon=&date="| BSC
    AWC -->|"success"| TS --> RS --> EN --> PDF
    AWC -->|"failure"| SP --> PDF
    BSC -->|"success"| WDF
    BSC -->|"failure"| SW --> WDF

    style APIs fill:#263238,color:#e8e8e8
    style Fallback fill:#bf360c,color:#e8e8e8
    style Output fill:#1b5e20,color:#e8e8e8
```

---

## 8. Buffer Tank Thermal Model

```mermaid
graph TB
    subgraph EnergyBalance["Energy Balance: C_tank × dT/dt = Q_HP − Q_house − Q_loss"]
        direction LR
        QIN["Q_HP<br/>Heat Pump Input<br/>0 — 8000 W"]
        TANK_NODE["Buffer Tank<br/>T_tank<br/>C = 2.07 MJ/K"]
        QOUT["Q_house<br/>Heat to Building<br/>via mixing valve"]
        QLOSS["Q_loss<br/>Standby Losses<br/>UA × (T − T_amb)"]

        QIN -->|"Energy In"| TANK_NODE
        TANK_NODE -->|"Energy Out"| QOUT
        TANK_NODE -->|"Losses"| QLOSS
    end

    subgraph Parameters["Tank Parameters"]
        VOL["Volume: 500 L<br/>Mass: 494 kg"]
        CAP["Capacity: 2,065,908 J/K<br/>≈ 2.07 MJ/K"]
        UA["UA: 3.0 W/K<br/>Loss at 60°C: ~135 W"]
        RNG["Range: 30°C — 80°C<br/>Init: 45°C"]
    end

    subgraph Discretisation["Forward Euler (Δt = 900s)"]
        EQ["dT = (Q_in − Q_out − Q_loss) × Δt / C_tank<br/>T_new = clip(T + dT, 30, 80)"]
    end

    EnergyBalance --> Discretisation

    style EnergyBalance fill:#e65100,color:#fff
    style Parameters fill:#4e342e,color:#fff
    style Discretisation fill:#1a237e,color:#fff
```

---

## 9. COP Behaviour

The heat pump COP depends on the temperature lift from ambient air to the tank. Higher tank temperatures and colder ambient air reduce COP, making electricity-to-heat conversion less efficient.

```mermaid
graph LR
    subgraph Formula["COP Formula"]
        F["COP = η × (T_tank + 273.15) / max(T_tank − T_amb + 5, 5)<br/>η = 0.45 (Carnot efficiency)<br/>COP clamped to [1.0, 7.0]"]
    end

    subgraph Examples["Example COP Values"]
        E1["Mild day + warm tank<br/>T_amb = 10°C, T_tank = 35°C<br/>COP ≈ 4.6"]
        E2["Average conditions<br/>T_amb = 5°C, T_tank = 50°C<br/>COP ≈ 2.9"]
        E3["Cold day + hot tank<br/>T_amb = −5°C, T_tank = 70°C<br/>COP ≈ 2.1"]
    end

    subgraph Tradeoff["The Core Tradeoff"]
        T1["Higher T_tank → More stored energy ✅<br/>Higher T_tank → Lower COP ❌<br/>Higher T_tank → More standby loss ❌"]
        T2["The MPC finds the sweet spot:<br/>Heat tank just enough to cover<br/>demand at minimum cost"]
    end

    Formula --> Examples --> Tradeoff

    style Formula fill:#1a237e,color:#fff
    style Examples fill:#004d40,color:#fff
    style Tradeoff fill:#b71c1c,color:#fff
```

---

## 10. Module Dependency Graph

```mermaid
graph BT
    CONFIG["config.py<br/>All parameters"]
    TANK["buffer_tank.py<br/>BufferTank"]
    MPC["mpc_controller.py<br/>EconomicMPC<br/>TrackingController"]
    DATA["data_pipeline.py<br/>AWattarClient<br/>BrightSkyClient"]
    SIM["simulation.py<br/>HeatPumpSimulation"]
    RUN["run_mpc.py<br/>CLI entry point"]
    EVAL["evaluate.py<br/>Plotting"]
    I4B["i4b/<br/>Building physics<br/>(git submodule)"]

    CASADI["casadi<br/>(external)"]
    IPOPT_EXT["IPOPT<br/>(bundled)"]

    CONFIG --> TANK
    CONFIG --> MPC
    CONFIG --> DATA
    CONFIG --> SIM
    CONFIG --> EVAL

    CASADI --> MPC
    IPOPT_EXT --> MPC

    TANK --> SIM
    MPC --> SIM
    DATA --> SIM
    I4B --> SIM

    SIM --> RUN
    SIM --> EVAL

    style CONFIG fill:#ff8f00,color:#000
    style MPC fill:#7b1fa2,color:#fff
    style TANK fill:#e65100,color:#fff
    style SIM fill:#1565c0,color:#fff
    style RUN fill:#2e7d32,color:#fff
    style DATA fill:#0d47a1,color:#fff
    style EVAL fill:#2e7d32,color:#fff
    style I4B fill:#37474f,color:#fff
```

---

## 11. Timing Diagram

Shows how the two control layers interleave over 2 hours of operation:

```
Time    Upper Layer              Lower Layer              Plant State
─────   ─────────────────────    ─────────────────────    ────────────────────
00:00   ┌─ SOLVE (IPOPT) ──┐
        │  24h NLP          │
        │  N=24 vars        │
        └── plan ready ─────┤
                            ├── Step 1: PI compute ──►   T_room=20.3, T_tank=45
                            │   q_hp=4200W, valve=0.52
00:15                       ├── Step 2: PI compute ──►   T_room=20.5, T_tank=46
                            │   q_hp=3800W, valve=0.48
00:30                       ├── Step 3: PI compute ──►   T_room=20.8, T_tank=47
                            │   q_hp=3100W, valve=0.45
00:45                       ├── Step 4: PI compute ──►   T_room=21.0, T_tank=48
                            │   q_hp=2500W, valve=0.40

01:00   ┌─ RE-SOLVE ───────┐
        │  New prices/weather│
        │  Updated T_room,  │
        │  T_tank from plant│
        └── new plan ───────┤
                            ├── Step 5: PI compute ──►   T_room=21.1, T_tank=49
                            │   q_hp=1800W, valve=0.38
01:15                       ├── Step 6: PI compute ──►   T_room=21.0, T_tank=48
                            │   q_hp=0W, valve=0.35       (cheap hour ended)
01:30                       ├── Step 7: PI compute ──►   T_room=20.8, T_tank=47
                            │   q_hp=0W, valve=0.42       (using stored heat)
01:45                       ├── Step 8: PI compute ──►   T_room=20.6, T_tank=46
                            │   q_hp=0W, valve=0.50       (discharging tank)

02:00   ┌─ RE-SOLVE ───────┐    ...continues...
```

### Key Insight

The upper layer decides *strategy* ("charge now, coast later"), while the lower layer handles *execution* ("maintain 21°C using the stored energy"). The 4:1 step ratio (4 lower steps per upper step) keeps the system responsive without over-solving the economic problem.

---

## Evaluation Output Panels

After a simulation run with `--plot`, the system generates a 5-panel figure:

```
┌──────────────────────────────────────────────────────────────────┐
│  Panel 1: Room Temperature                                       │
│  • Blue solid: T_room actual    • Blue dashed: T_room MPC ref    │
│  • Orange: T_ambient            • Green band: comfort zone       │
├──────────────────────────────────────────────────────────────────┤
│  Panel 2: Tank Temperature                                       │
│  • Pink solid: T_tank actual    • Pink dashed: T_tank MPC ref    │
├──────────────────────────────────────────────────────────────────┤
│  Panel 3: HP Power & Valve                                       │
│  • Purple fill: HP thermal kW   • Cyan line: valve position      │
├──────────────────────────────────────────────────────────────────┤
│  Panel 4: Price & Energy                                         │
│  • Red line: €/kWh price        • Blue bars: electrical kWh      │
├──────────────────────────────────────────────────────────────────┤
│  Panel 5: COP & Cost                                             │
│  • Brown line: COP              • Green fill: cumulative €       │
└──────────────────────────────────────────────────────────────────┘
```
