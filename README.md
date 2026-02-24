# Multi-Connectivity 5G – Strategy Evaluation Framework

This repository contains the analysis and simulation framework used to evaluate **multi-connectivity strategies** in 5G networks using real experimental measurements.

The project compares:

- Baseline single-link operation  
- Full Duplication (FD)  
- Partial Duplication (PD)  
- Intelligent Switching  
- Link Aggregation  

using latency, packet loss, radio conditions (RSRP), and uplink transmit power metrics.

---

# 📁 Project Structure
.
├── data_loader.py
├── MC_strategies.py
├── link_aggregation.py
├── metrics.py
├── plot_figures.py
├── route.py
├── run_analysis.py
├── run_plot_cost_latency_analysis.py
├── run_plot_sensitivity_analysis.py
└── data/


---

# 📦 Main Components

## 1️⃣ Data Loading

**File:** `data_loader.py`

`load_experiment_data()` loads experiment CSV files and:

- Filters by scenario and Mbps
- Computes best interface per packet
- Fills missing packets
- Computes best latency
- Returns:
  - `df_full`
  - `df_radio`
  - `scenarios`
  - `mbps_list`

### Experiments

| Experiment | Scenarios | Mbps |
|------------|-----------|------|
| 1 | URBAN, HYBRID, RURAL | 4, 2, 1 |
| 2 | RURAL | 4, 2, 1, 0.5, 0.25 |

---

## 2️⃣ Multi-Connectivity Strategies

Implemented in:

- `MC_strategies.py`
- `link_aggregation.py`

### ✔ Full Duplication (FD)

Selects the interface with minimum latency for each packet.

### ✔ Link Aggregation

Combines both interfaces into a logical high-throughput link.

### ✔ Partial Duplication (PD)

Duplicates packets only under bad radio/latency conditions:

- RSRP threshold
- Tx power threshold
- Latency threshold
- Combined score mode

### ✔ Switching Strategy

Implements threshold-based and score-based interface switching with:

- Anti ping-pong protection
- Radio sampling period
- Latency sampling period

---

## 3️⃣ Metrics

**File:** `metrics.py`

`compute_statistics()` calculates:

- Latency percentiles (P90, P95, P99, P99.9)
- Packet loss percentage
- Interface usage percentage
- RSRP percentiles
- Tx power percentiles

---

## 4️⃣ Plotting

**File:** `plot_figures.py`

Provides:

- RSRP vs Latency distributions
- Broken-axis latency plots
- Packet loss per radio bin
- UL Tx power vs RSRP correlations
- Strategy comparison figures

---

## 5️⃣ Route Visualization

**File:** `route.py`

Generates interactive Folium maps:

- Route GPS trace
- Base station locations
- Operator markers

---

# 🚀 Running the Analysis

Main entry point:

**File:** `run_analysis.py`

## Basic Usage

```bash
python run_analysis.py --experiment 1 --scenario URBAN --mbps 4
Run Only One Strategy
python run_analysis.py \
    --experiment 2 \
    --scenario RURAL \
    --mbps 4 \
    --strategy FD
Run Partial Duplication Mode
python run_analysis.py \
    --experiment 2 \
    --scenario RURAL \
    --mbps 4 \
    --strategy PD \
    --mode full
Show Route
python run_analysis.py \
    --experiment 1 \
    --scenario URBAN \
    --mbps 4 \
    --show_route True
📊 Cost & Latency Analysis

Additional scripts:

run_plot_cost_latency_analysis.py → Cost vs Latency comparison

run_plot_sensitivity_analysis.py → Sensitivity analysis (Tx power threshold)

These scripts use manually aggregated experiment results for visualization.

📈 Strategies Overview
Strategy	Description	Duplication	Switching
Baseline	Single interface	❌	❌
FD	Always duplicate	✅	❌
PD	Conditional duplication	✅	❌
Switching	Smart interface change	❌	✅
Link Aggregation	Logical merging	⚡	❌
📚 Requirements

Recommended environment:

pip install pandas numpy matplotlib seaborn scipy statsmodels folium

Python ≥ 3.9 recommended.

🧠 Research Context

This framework evaluates:

Reliability vs Cost trade-offs

Radio-aware decision making

Latency-aware packet duplication

Intelligent multi-operator utilization

The system enables reproducible evaluation of:

Radio thresholds

Duplication policies

Switching policies

Cost models

🏗 Example Workflow

Load experiment data

Apply strategy

Compute statistics

Plot distributions

Compare cost vs latency