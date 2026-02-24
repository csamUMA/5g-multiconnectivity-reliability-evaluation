import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import argparse
from pathlib import Path

def main():
    
    

    # ============================================================
    # Note: You have to fill this data manually from the experiments, as shown in the example data below. If not, your PC can just explode of munch of data.
    # ============================================================

    DATA = {
        "Baseline": {
            "telenor": (100, 0),
            "tdc":     (100, 0),
        },
        "Latency\nPAAF Switching": {
            "telenor": (95.8, 4.2),
            "tdc":     (91.2, 8.8),
        },
        "Latency\nPAAF PD": {
            "telenor": (100, (8.4+15.8)/2),
            "tdc":     (100, (8.4+15.8)/2),
        },
        "(RSRP AND UL Tx\nPwr) OR Latency\nPAAF PD": {
            "telenor": (100, (22.8+29.8)/2),
            "tdc":     (100, (22.8+29.8)/2),
        },
        "Full\nDuplication": {
            "telenor": (100, 100),
            "tdc":     (100, 100),
        },
    }

    LATENCY = {
        "Baseline": ((283+1941)/2, (1952+10000)/2),   # (P95, P99) in ms
        "Full\nDuplication": (53, 185),
        "Latency\nPAAF Switching": ((152+952)/2, (1099+10000)/2),
        "Latency\nPAAF PD": ((80+133)/2, (307+218)/2),
        "(RSRP AND UL Tx\nPwr) OR Latency\nPAAF PD": ((62+127)/2, (196+229)/2),
    }

    # ============================================================
    

    MODES = [
        "Baseline",
        "Full\nDuplication",
        "Latency\nPAAF Switching",
        "Latency\nPAAF PD",
        "(RSRP AND UL Tx\nPwr) OR Latency\nPAAF PD",
    ]

    lat_p95 = [LATENCY[m][0] for m in MODES]
    lat_p99 = [LATENCY[m][1] for m in MODES]

    def compute_cost(if1_pct, if2_pct, alpha):
        if1 = if1_pct / 100.0
        if2 = if2_pct / 100.0
        principal = max(if1, if2)
        secundaria = min(if1, if2)
        return principal + secundaria * alpha

    ALPHAS = [1.2, 1.5, 2.0, 3.0]
    costs = {a: [] for a in ALPHAS}

    for mode in MODES:
        tel_if1, tel_if2 = DATA[mode]["telenor"]
        tdc_if1, tdc_if2 = DATA[mode]["tdc"]

        for alpha in ALPHAS:
            cost_tel = compute_cost(tel_if1, tel_if2, alpha)
            cost_tdc = compute_cost(tdc_if1, tdc_if2, alpha)
            cost_mean = (cost_tel + cost_tdc) / 2.0
            costs[alpha].append(cost_mean * 100)

    plt.rcParams.update({
            "axes.labelsize": 14,      # axis label font size
            "xtick.labelsize": 12,     # x tick labels
            "ytick.labelsize": 12,     # y tick labels
            "legend.fontsize": 11,     # legend
        })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()

    x = np.arange(len(MODES))

    cost_width = 0.14
    lat_width = 0.14

    # ---- Cost bars (LEFT axis, blue cluster) ----
    cost_colors = [
        "#b3d9ff",
        "#66b2ff",
        "#1f77b4",
        "#08306b",
    ]

    for i, alpha in enumerate(ALPHAS):
        ax.bar(
            x + (i - (len(ALPHAS)-1)/2) * cost_width,
            costs[alpha],
            cost_width,
            color=cost_colors[i],
            label=f"Cost x{alpha}",
            zorder=2,
        )

    # ---- Latency bars (RIGHT axis, SEPARATE cluster) ----
    # shift latency bars to the right of all cost bars
    lat_offset = (len(ALPHAS) / 2 + 0.5) * cost_width

    ax2.bar(
        x + lat_offset,
        lat_p95,
        lat_width,
        facecolor="#ffcc80",     
        edgecolor="#ff7f0e",     
        hatch="//",              
        linewidth=1.0,
        label="Latency P95",
        zorder=3,
    )
    # ---- Labels on top of latency bars (P95) ----
    for xi, yi in zip(x + lat_offset, lat_p95):
        ax2.text(
            xi,
            yi + 0.02 * ax2.get_ylim()[1], 
            f"{yi:.0f}",                   
            ha="center",
            va="bottom",
            fontsize=8,
            color="#ff7f0e",
            rotation=90,                
            zorder=4,
        )



        
    ax2.bar(
        x + lat_offset + lat_width,
        lat_p99,
        lat_width,
        facecolor="#ffb74d",     
        edgecolor="#ff7f0e",
        hatch="---",             
        linewidth=1.0,
        label="Latency P99",
        zorder=3,
    )

    # ---- Labels on top of latency bars (P95) ----
    for xi, yi in zip(x + lat_offset + lat_width, lat_p99):
        if yi > 1300:
            print(f"Skipping label for {xi} because {yi} is too high")
            continue  
        
        ax2.text(
            xi,
            yi + 0.005 * ax2.get_ylim()[1], 
            f"{yi:.0f}",                    
            ha="center",
            va="bottom",
            fontsize=8,
            color="#ff7f0e",
            rotation=90,                    
            zorder=4,
        )

    blue = "#1f77b4"

    ax.set_xticks(x)
    ax.set_xticklabels(MODES)

    ax.set_ylabel("Normalized cost [%]", color=blue)
    ax.tick_params(axis="y", colors=blue)
    ax.spines["left"].set_color(blue)
    ax.set_ylim(80, 420)

    ax2.set_ylabel("Latency [ms]", color="#ff7f0e")
    ax2.tick_params(axis="y", colors="#ff7f0e")
    ax2.spines["right"].set_color("#ff7f0e")
    ax2.set_ylim(0, max(lat_p95) * 1.3)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()
    BASE_DIR = Path(__file__).resolve().parent.parent
    FIGURES_DIR = BASE_DIR / "figures" / "cost_latency"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = FIGURES_DIR / f"paff_cost_latency_comparison.pdf"
    plt.savefig(out_name, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    main()
