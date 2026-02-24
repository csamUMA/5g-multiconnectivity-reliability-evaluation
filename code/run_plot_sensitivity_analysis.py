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

    telenor_tx_pwr_data = [
        {"tx_power_dbm": 10.0, "IF1": 100.0, "IF2": 60.0, "latency_p95_ms": 57,  "latency_p99_ms": 192,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 11.0, "IF1": 100.0, "IF2": 58.7, "latency_p95_ms": 57,  "latency_p99_ms": 192,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 12.0, "IF1": 100.0, "IF2": 57.1, "latency_p95_ms": 57,  "latency_p99_ms": 192,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 13.0, "IF1": 100.0, "IF2": 56.4, "latency_p95_ms": 58,  "latency_p99_ms": 192,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 14.0, "IF1": 100.0, "IF2": 54.8, "latency_p95_ms": 58,  "latency_p99_ms": 193,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 15.0, "IF1": 100.0, "IF2": 53.0, "latency_p95_ms": 58,  "latency_p99_ms": 193,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 16.0, "IF1": 100.0, "IF2": 50.7, "latency_p95_ms": 58,  "latency_p99_ms": 193,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 17.0, "IF1": 100.0, "IF2": 48.9, "latency_p95_ms": 58,  "latency_p99_ms": 193,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 18.0, "IF1": 100.0, "IF2": 46.9, "latency_p95_ms": 59,  "latency_p99_ms": 193,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 19.0, "IF1": 100.0, "IF2": 39.3, "latency_p95_ms": 60,  "latency_p99_ms": 193,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 20.0, "IF1": 100.0, "IF2": 38.2, "latency_p95_ms": 62,  "latency_p99_ms": 209,  "latency_p999_ms": 482,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 20.5, "IF1": 100.0, "IF2": 34.1, "latency_p95_ms": 63,  "latency_p99_ms": 214,  "latency_p999_ms": 489,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 21.0, "IF1": 100.0, "IF2": 33.5, "latency_p95_ms": 63,  "latency_p99_ms": 214,  "latency_p999_ms": 489,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 21.5, "IF1": 100.0, "IF2": 11.9, "latency_p95_ms": 128, "latency_p99_ms": 629,  "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 22.0, "IF1": 100.0, "IF2": 11.7, "latency_p95_ms": 128, "latency_p99_ms": 629,  "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"tx_power_dbm": 22.5, "IF1": 100.0, "IF2": 2.4,  "latency_p95_ms": 277, "latency_p99_ms": 1948, "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
    ]


    tdc_tx_pwr_data = [
        {"tx_power_dbm": 10.0, "IF1": 65.4, "IF2": 100.0, "latency_p95_ms": 89,  "latency_p99_ms": 192,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 11.0, "IF1": 64.3, "IF2": 100.0, "latency_p95_ms": 96,  "latency_p99_ms": 192,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 12.0, "IF1": 63.3, "IF2": 100.0, "latency_p95_ms": 96,  "latency_p99_ms": 192,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 13.0, "IF1": 62.4, "IF2": 100.0, "latency_p95_ms": 102, "latency_p99_ms": 192,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 14.0, "IF1": 60.7, "IF2": 100.0, "latency_p95_ms": 110, "latency_p99_ms": 192,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 15.0, "IF1": 58.4, "IF2": 100.0, "latency_p95_ms": 113, "latency_p99_ms": 192,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 16.0, "IF1": 56.7, "IF2": 100.0, "latency_p95_ms": 115, "latency_p99_ms": 194,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 17.0, "IF1": 54.0, "IF2": 100.0, "latency_p95_ms": 118, "latency_p99_ms": 196,  "latency_p999_ms": 482,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 18.0, "IF1": 51.6, "IF2": 100.0, "latency_p95_ms": 120, "latency_p99_ms": 198,  "latency_p999_ms": 483,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 19.0, "IF1": 43.6, "IF2": 100.0, "latency_p95_ms": 122, "latency_p99_ms": 200,  "latency_p999_ms": 483,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 20.0, "IF1": 41.9, "IF2": 100.0, "latency_p95_ms": 125, "latency_p99_ms": 208,  "latency_p999_ms": 484,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 20.5, "IF1": 39.6, "IF2": 100.0, "latency_p95_ms": 127, "latency_p99_ms": 209,  "latency_p999_ms": 484,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 21.0, "IF1": 38.2, "IF2": 100.0, "latency_p95_ms": 128, "latency_p99_ms": 211,  "latency_p999_ms": 484,    "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 21.5, "IF1": 25.2, "IF2": 100.0, "latency_p95_ms": 136, "latency_p99_ms": 253,  "latency_p999_ms": 1696,   "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 22.0, "IF1": 25.2, "IF2": 100.0, "latency_p95_ms": 136, "latency_p99_ms": 253,  "latency_p999_ms": 1696,   "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
        {"tx_power_dbm": 22.5, "IF1": 2.3,  "IF2": 100.0, "latency_p95_ms": 1671,"latency_p99_ms": 10000,"latency_p999_ms": 10000,  "cost_20":0, "cost_50":0, "cost_100":0, "cost_200":0},
    ]


    telenor_rsrp_data = [
        {"rsrp_threshold": -85,  "IF1": 100.0, "IF2": 62.2, "latency_p95_ms": 57,  "latency_p99_ms": 188,  "latency_p999_ms": 481,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -90,  "IF1": 100.0, "IF2": 55.3, "latency_p95_ms": 58,  "latency_p99_ms": 188,  "latency_p999_ms": 481,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -95,  "IF1": 100.0, "IF2": 44.2, "latency_p95_ms": 60,  "latency_p99_ms": 191,  "latency_p999_ms": 487,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -100, "IF1": 100.0, "IF2": 27.1, "latency_p95_ms": 64,  "latency_p99_ms": 198,  "latency_p999_ms": 489,   "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -105, "IF1": 100.0, "IF2": 11.7, "latency_p95_ms": 121, "latency_p99_ms": 1124, "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -110, "IF1": 100.0, "IF2": 5.8,  "latency_p95_ms": 198, "latency_p99_ms": 1336, "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -115, "IF1": 100.0, "IF2": 2.5,  "latency_p95_ms": 230, "latency_p99_ms": 1336, "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -120, "IF1": 100.0, "IF2": 0.0,  "latency_p95_ms": 283, "latency_p99_ms": 1953, "latency_p999_ms": 10000, "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
    ]


    tdc_rsrp_data = [
        {"rsrp_threshold": -85,  "IF1": 61.4, "IF2": 100.0, "latency_p95_ms": 116,  "latency_p99_ms": 199,   "latency_p999_ms": 483,    "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -90,  "IF1": 53.7, "IF2": 100.0, "latency_p95_ms": 125,  "latency_p99_ms": 203,   "latency_p999_ms": 483,    "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -95,  "IF1": 43.1, "IF2": 100.0, "latency_p95_ms": 127,  "latency_p99_ms": 205,   "latency_p999_ms": 483,    "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -100, "IF1": 32.1, "IF2": 100.0, "latency_p95_ms": 130,  "latency_p99_ms": 208,   "latency_p999_ms": 484,    "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -105, "IF1": 18.8, "IF2": 100.0, "latency_p95_ms": 142,  "latency_p99_ms": 266,   "latency_p999_ms": 690,    "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -110, "IF1": 7.3,  "IF2": 100.0, "latency_p95_ms": 245,  "latency_p99_ms": 1928,  "latency_p999_ms": 10000,  "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -115, "IF1": 0.5,  "IF2": 100.0, "latency_p95_ms": 1850, "latency_p99_ms": 10000, "latency_p999_ms": 10000,  "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
        {"rsrp_threshold": -120, "IF1": 0.0,  "IF2": 100.0, "latency_p95_ms": 1941, "latency_p99_ms": 10000, "latency_p999_ms": 10000,  "cost_20": 0, "cost_50": 0, "cost_100": 0, "cost_200": 0},
    ]

    parser = argparse.ArgumentParser(description="Run Plot for the Sensitivity Analisis. You must include the data from the experiments manually in the code, as shown in the example data above. If not, your PC can just explode of munch of data.")

    parser.add_argument("--kpi", type=str, required=True,
                        help="[tx_power_dbm, rsrp_threshold]")



    DATASETS = {
        "tx_power_dbm": (telenor_tx_pwr_data, tdc_tx_pwr_data),
        "rsrp_threshold": (telenor_rsrp_data, tdc_rsrp_data),
    }

    kpi = parser.parse_args().kpi  # "rsrp_threshold", "tx_power_dbm"

    telenor_data, tdc_data = DATASETS[kpi]

    Incremental_cost = 20  # Percentage points (se mantiene por si lo usas después)

    color_overhead = "#ee732bff"
    color_latency = "#276faaff"

    mobile_data = []

    for tel, tdc in zip(telenor_data, tdc_data):
        assert tel[kpi] == tdc[kpi]

        mobile_data.append({
            kpi: tel[kpi],
            "overhead": (tel["IF2"] + tdc["IF1"]) / 2,
            "latency_p95_ms": (tel["latency_p95_ms"] + tdc["latency_p95_ms"]) / 2,
            "latency_p99_ms": (tel["latency_p99_ms"] + tdc["latency_p99_ms"]) / 2,
            # costes se dejan calculados por si luego los quieres recuperar, pero ya no se plotean
            "cost_20": ((tel["IF2"] + tdc["IF1"]) / 2) * 1.2,
            "cost_50": ((tel["IF2"] + tdc["IF1"]) / 2) * 1.5,
            "cost_100": ((tel["IF2"] + tdc["IF1"]) / 2) * 2.0,
            "cost_200": ((tel["IF2"] + tdc["IF1"]) / 2) * 3.0,
        })


    # =========================
    # Transformaciones eje Y
    # =========================
    multiplyr = 5
    offset = 0

    def overhead_to_latency(y):
        return multiplyr * y + offset

    def latency_to_overhead(y):
        return (y - offset) / multiplyr


    # =========================
    # Plot (SIN subplot superior)
    # =========================
    def plot_operator_no_cost(data, operator_name):
        kpi_value = [d[kpi] for d in data]
        overhead = [d["overhead"] for d in data]
        latency_p95 = [d["latency_p95_ms"] for d in data]
        latency_p99 = [d["latency_p99_ms"] for d in data]

        # Convertir latencias a escala del eje izquierdo
        latency_p95_ov = [(l - offset) / multiplyr for l in latency_p95]
        latency_p99_ov = [(l - offset) / multiplyr for l in latency_p99]

        plt.rcParams.update({
            "axes.labelsize": 14,      # axis label font size
            "xtick.labelsize": 12,     # x tick labels
            "ytick.labelsize": 12,     # y tick labels
            "legend.fontsize": 11,     # legend
        })

        fig, ax = plt.subplots(figsize=(8,6))

        # Curvas
        ax.plot(
            kpi_value,
            overhead,
            color=color_overhead,
            linewidth=2,
            label="Overhead"
        )

        ax.plot(
            kpi_value,
            latency_p95_ov,
            color=color_latency,
            linestyle="-.",
            linewidth=2,
            label="Latency P95"
        )

        ax.plot(
            kpi_value,
            latency_p99_ov,
            color=color_latency,
            linestyle=":",
            linewidth=2,
            label="Latency P99"
        )

        # Labels / ejes
        if kpi == "rsrp_threshold":
            ax.set_xlabel("RSRP Threshold (dBm)")
            ticks = [-85, -90, -95, -100, -105, -110, -115, -120]
            labels = [-85, -90, -95, -100, -105, -110, -115, -120]
        else:
            ax.set_xlabel("Tx Power Threshold (dBm)")
            ticks = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20.5, 21, 21.5, 22]
            labels = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, "20.5", 21, "21.5", 22]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=30, ha="center", rotation_mode="anchor")
        ax.set_xlim(min(ticks) - 1, max(ticks) + 1)

        ax.set_ylabel("Overhead (%)", color=color_overhead)
        ax.tick_params(axis="y", colors=color_overhead)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        ax.grid(True, linestyle="--", alpha=0.5)

        # Eje secundario (latencia)
        secax = ax.secondary_yaxis(
            "right",
            functions=(overhead_to_latency, latency_to_overhead)
        )
        secax.set_ylabel("Uplink Latency [ms]", color=color_latency)
        secax.tick_params(axis="y", colors=color_latency)

        # Nota: en tu código original estabas seteando ticks dos veces (se sobrescribía).
        # Aquí dejo solo una lista coherente.
        secax.set_yticks([0, 100, 200, 300, 400, 500, 600])

        ax.spines["left"].set_color(color_overhead)
        secax.spines["right"].set_color(color_latency)

        ax.legend(loc="upper left", frameon=True)

        plt.tight_layout()

        BASE_DIR = Path(__file__).resolve().parent.parent
        FIGURES_DIR = BASE_DIR / "figures" / "sensitivity_analysis"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_name = FIGURES_DIR / f"sensitivity_analysis_{kpi}.pdf"

        plt.savefig(out_name, bbox_inches="tight")
        plt.show()


    # =========================
    # Ejecutar
    # =========================
    plot_operator_no_cost(mobile_data, "BOTH")


if __name__ == "__main__":
    main()
