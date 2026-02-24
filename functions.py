import cv2
import socket
import gi
import struct
import random
import argparse
from datetime import datetime, timezone
import time
import threading
import pandas as pd
import numpy as np
from pathlib import Path
import os, glob, pickle, re
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import folium
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display
import base64
from io import BytesIO
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import ast
import math
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator
from math import cos, radians, sqrt
from tabulate import tabulate
import matplotlib as mpl
from matplotlib.lines import Line2D

def prepare_radio_dataframe(df_radio_all, interface_filter=None):
    """
    Limpia, transforma y codifica las columnas del dataframe de radio,
    devolviendo un dataframe numérico listo para modelado (por ejemplo, LSTM).

    Parámetros
    ----------
    df_radio_all : pd.DataFrame
        DataFrame original con todas las mediciones sin procesar.
    interface_filter : str, opcional
        Nombre de la interfaz a filtrar (por defecto None: no filtra).

    Devuelve
    --------
    df_radio : pd.DataFrame
        DataFrame limpio, numérico y ordenado temporalmente.
    """

    # --- 1️⃣ Seleccionar columnas relevantes ---
    cols2keep = [
        "Now_ms", 
        "Seq",
        "Mbps_x100",
        "Packet_Sent",
        "lat",
        "lon",
        "alt_m",
        "Uplink_Latency",
        "Downlink_latency",
        "RTT",  
        "Interface",
        "pci_lte", 
        "earfcn", 
        "rsrp_lte", 
        "rsrq_lte",
        "rssi_lte",
        "sinr_lte",
        "pci_nr", 
        "tx_power_dbm",
        "rsrp_nr", 
        "rsrq_nr", 
        "sinr_nr",
        "nr_band",
        "enb_id", 
        "cell_local_id", 
        "pcc_band",
        "handover_type", 
        "neighbor_best_rsrp", 
        "neighbor_best_rsrq",
        "mbps",
        "scenario",
        "best_interface",
        "best_latency",
        "cainfo_json"
    ]
    existing_cols = [c for c in cols2keep if c in df_radio_all.columns]

    df_radio = df_radio_all[existing_cols].copy()
    df_radio['Uplink_Latency']        = df_radio['Uplink_Latency'].fillna(10000)
    df_radio['Downlink_latency']        = df_radio['Downlink_latency'].fillna(10000)
    df_radio['RTT']        = df_radio['RTT'].fillna(10000)

    # --- 2️⃣ Extraer número de banda (por ejemplo LTE20 → 20) ---
    df_radio['pcc_band'] = df_radio['pcc_band'].astype(str).str.extract(r'(\d+)').astype(float)

    # --- 3️⃣ Mapas de handovers ---
    handover_map_lte = {
        np.nan: 0,
        'Intrafrequency-LTE': 1,
        'Interfreq-LTE': 2,
        'Intersite-LTE': 3
    }

    handover_map_nr = {
        np.nan: 0,
        'Intrafreq-NR': 1,
        'Interfreq-NR': 2,
        'Intersite-NR': 3,
        'Connect-NR': 4,
        'Disconnect-NR': 5
    }

    scenario_map = {
        'RURAL': 1,
        'URBAN': 2,
        'HYBRID': 3
    }

    # --- 4️⃣ Función auxiliar para codificar el handover ---
    def encode_handover_type(value):
        if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).lower() in ['nan', 'none', '', '[]']:
            return 0
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                value = parsed if isinstance(parsed, list) else [value]
            except Exception:
                value = [value]
        elif not isinstance(value, list):
            value = [value]

        lte_code, nr_code = 0, 0
        for item in value:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if 'LTE' in item:
                lte_code = handover_map_lte.get(item, 0)
            elif 'NR' in item:
                # NOTE: Si solo quiero ver el handover en LTE, comento la siguiente línea! (pongo pass)
                nr_code = handover_map_nr.get(item, 0)
        return int(lte_code * 10 + nr_code)

    # --- 5️⃣ Aplicar codificación de handover ---
    df_radio['handover_type_final'] = df_radio['handover_type'].apply(encode_handover_type)

    df_radio['scenario'] = df_radio['scenario'].map(scenario_map)

    # --- 6️⃣ Crear feature binaria de vecinos + imputar valores ---
    df_radio['has_neighbor_info']  = df_radio['neighbor_best_rsrp'].notna().astype(int)
    df_radio['neighbor_best_rsrp'] = df_radio['neighbor_best_rsrp'].ffill().bfill().fillna(-140)
    df_radio['neighbor_best_rsrq'] = df_radio['neighbor_best_rsrq'].ffill().bfill().fillna(-20)
    
    # --- 8️⃣ Nueva feature: disponibilidad de parámetros de radio ---
    df_radio['has_radio_parameters'] = df_radio['earfcn'].notna().astype(int)

    # --- 9️⃣ Imputación de columnas LTE y NR ---
    df_radio['pci_lte']        = df_radio['pci_lte'].ffill().bfill().fillna(0)
    df_radio['earfcn']         = df_radio['earfcn'].ffill().bfill().fillna(0)
    df_radio['rsrp_lte']       = df_radio['rsrp_lte'].ffill().bfill().fillna(-140)
    df_radio['rsrq_lte']       = df_radio['rsrq_lte'].ffill().bfill().fillna(-20)
    df_radio['rssi_lte']       = df_radio['rssi_lte'].ffill().bfill().fillna(-140)
    df_radio['sinr_lte']       = df_radio['sinr_lte'].ffill().bfill().fillna(-20)
    df_radio['tx_power_dbm']   = df_radio['tx_power_dbm'].ffill().bfill().fillna(25)
    df_radio['pci_nr']         = df_radio['pci_nr'].ffill().bfill().fillna(0)
    df_radio['rsrp_nr']        = df_radio['rsrp_nr'].ffill().bfill().fillna(-140)
    df_radio['rsrq_nr']        = df_radio['rsrq_nr'].ffill().bfill().fillna(-20)
    df_radio['sinr_nr']        = df_radio['sinr_nr'].ffill().bfill().fillna(-20)
    df_radio['nr_band']        = df_radio['nr_band'].ffill().bfill().fillna(0)
    df_radio['enb_id']         = df_radio['enb_id'].ffill().bfill().fillna(0)
    df_radio['cell_local_id']  = df_radio['cell_local_id'].ffill().bfill().fillna(0)
    df_radio['pcc_band']       = df_radio['pcc_band'].ffill().bfill().fillna(0)

    df_radio['cainfo_json']       = df_radio['cainfo_json'].ffill().bfill()
    
    # --- 7️⃣ Flag de conexión 5G ---
    #df_radio['has_5G'] = df_radio['rsrp_nr'].notna().astype(int)
    df_radio['has_5G'] = 0

    # Set 'has_5G' to 1 where 'cainfo_json' contains the NR band
    df_radio.loc[
    # This matches: "band": "NR5G1" or "band": "NR5G78" etc.
    df_radio['cainfo_json'].str.contains('"band": "NR', na=False),
        'has_5G'
    ] = 1
    
    try:
        df_radio['Mbps_x100']        = df_radio['Mbps_x100'].ffill().bfill()
    except:
        pass

    # Fill GPS information
    df_radio['lat']        = df_radio['lat'].ffill().bfill()
    df_radio['lon']        = df_radio['lon'].ffill().bfill()
    df_radio['alt_m']        = df_radio['alt_m'].ffill().bfill()

    # --- 🔟 Filtrar por interfaz (si aplica) ---
    if interface_filter is not None:
        df_radio = df_radio[df_radio["Interface"] == interface_filter].copy()

    df_radio['Interface'] = df_radio['Interface'].map({'5G_1': 1, '5G_2': 2}).ffill().bfill().fillna(0)

    # --- 11️⃣ Limpieza final ---
    #df_radio = df_radio.drop(columns=['handover_type'])
    df_radio = df_radio.sort_values("Now_ms").reset_index(drop=True)

    # --- 12️⃣ Verificación ---
    #print(f"[INFO] DataFrame limpio para {interface_filter}: {df_radio.shape[0]} filas, {df_radio.shape[1]} columnas")
    #print(f"[INFO] NaN totales restantes: {df_radio.isna().sum().sum()}")

    return df_radio

def fill_missing_packets(df, interfaces):
    """
    For each (scenario, mbps), fill missing (Seq, Interface) combinations
    with Uplink_Latency = 10000.
    """
    result = []

    for (scenario, mbps), group in df.groupby(["scenario", "mbps"]):
        print(f"Filling: {scenario} {mbps}")
        group = group.copy()

        # Generate all expected combinations of (Seq, Interface)
        all_seqs = np.arange(group["Seq"].min(), group["Seq"].max() + 1)
        expected = pd.MultiIndex.from_product(
            [all_seqs, interfaces],
            names=["Seq", "Interface"]
        ).to_frame(index=False)

        # Merge with actual data
        filled = expected.merge(group, on=["Seq", "Interface"], how="left")

        # Fill missing latency with 10000
        filled["Uplink_Latency"] = filled["Uplink_Latency"].fillna(10000)

        if "Mbps_x100" not in filled.columns:
            filled["Mbps_x100"] = 0

        # Interpolate Now_ms by group (per interface)
        filled["Now_ms"] = filled.groupby("Interface")["Now_ms"].transform(
            lambda x: x.interpolate(method="linear")
        )

        # If interpolation leaves NaNs at the ends, forward/back fill
        filled["Now_ms"] = filled.groupby("Interface")["Now_ms"].transform(
            lambda x: x.ffill().bfill()
        )

        try:
            # Interpolate Now_ms by group (per interface)
            filled["Packet_Sent"] = filled.groupby("Interface")["Packet_Sent"].transform(
                lambda x: x.interpolate(method="linear")
            )

            # If interpolation leaves NaNs at the ends, forward/back fill
            filled["Packet_Sent"] = filled.groupby("Interface")["Packet_Sent"].transform(
                lambda x: x.ffill().bfill()
            )      
        except:
            pass

        # Add context columns
        filled["scenario"] = scenario
        filled["mbps"] = mbps

        #filled["tx_power_dbm"] = filled["tx_power_dbm"].ffill().bfill()
        #filled["rsrp_lte"]     = filled["rsrp_lte"].ffill().bfill()

        #filled["tx_power_dbm"] = filled["tx_power_dbm"].ffill().bfill()

        result.append(filled)

    filled_df = pd.concat(result, ignore_index=True)
    return filled_df


def compute_best_latency(df):
    """
    Compute the best (minimum) Uplink_Latency per Seq
    and map it back to all rows of that Seq.
    """
    df = df.copy()
    result = []

    for (scenario, mbps), group in df.groupby(["scenario", "mbps"]):
        print(f"Computing best latency: {scenario} {mbps}")
        group = group.copy()

        idx_min_latency = group.groupby("Seq")["Uplink_Latency"].idxmin()
        best_interface_map = group.loc[idx_min_latency, ["Seq", "Interface"]].set_index("Seq")["Interface"]
        best_latency_map = group.loc[idx_min_latency, ["Seq", "Uplink_Latency"]].set_index("Seq")["Uplink_Latency"]

        group["best_interface"] = group["Seq"].map(best_interface_map)
        group["best_latency"] = group["Seq"].map(best_latency_map)

        result.append(group)

    final_df = pd.concat(result, ignore_index=True)
    return final_df

def coalesce(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        s = str(v)
        if s.strip() == "" or s.strip().lower() == "none":
            continue
        return v
    return None

def pick_cell(row):
    """
    Build a cell identifier from whatever is available, preferring enb_id,
    then NR identifiers, then LTE identifiers.
    """
    cid = coalesce(row.get("enb_id"))
    if cid is not None:
        return str(cid)

    pci_nr, arfcn_nr = row.get("pci_nr"), row.get("nr_arfcn")
    if pd.notna(pci_nr) and pd.notna(arfcn_nr):
        return f"NR-PCI{int(pci_nr)}-ARFCN{int(float(arfcn_nr))}"

    pci_lte, earfcn = row.get("pci_lte"), row.get("earfcn")
    if pd.notna(pci_lte) and pd.notna(earfcn):
        return f"LTE-PCI{int(pci_lte)}-EARFCN{int(float(earfcn))}"

    return None

def pick_rsrp(row):
    return coalesce(row.get("rsrp_lte"), row.get("rsrp_nr"), row.get("pcc_RSRP"))

def pick_rsrq(row):
    return coalesce(row.get("rsrq_lte"), row.get("rsrq_nr"), row.get("pcc_RSRQ"))

def subset_by_switch(df, scenario, interface, mbps, use_only_radio: bool):
    sub = df[
        (df["scenario"] == scenario) &
        (df["Interface"] == interface) &
        (df["mbps"] == mbps)
    ].copy()

    if use_only_radio:
        sub = sub[sub["has_radio_parameters"] == 1]
    else:
        sub = sub[sub["has_radio_parameters"] == 0]
    return sub

# --- Loss attribution: assign missing next-seq packets to the current row's KPI value
def assign_losses_to_kpi(
    df: pd.DataFrame,
    seq_col: str = "Seq",
    kpi_col: str = "rsrp_lte",
    group_cols: list[str] = ("scenario","Interface","mbps"),
    order_col: str = "Now_ms",
    count_tail_loss: bool = True,      # NEW
    expected_next_seq_col: str | None = None  # if you know the expected last seq, pass a column
) -> pd.DataFrame:
    if kpi_col not in df.columns:
        raise KeyError(f"Missing KPI column '{kpi_col}'")
    out = df.copy()

    def _per_group(g):
        g = g.sort_values([seq_col, order_col], kind="stable").copy()
        next_seq = g[seq_col].shift(-1)
        gap = (next_seq - g[seq_col] - 1).fillna(0)
        g["loss_assigned"] = gap.clip(lower=0).astype(int)

        if count_tail_loss and len(g):
            # If we know the expected next seq, use it; else assume no tail loss.
            if expected_next_seq_col and expected_next_seq_col in g.columns:
                expected_last = g[expected_next_seq_col].iloc[-1]
                # tail gap = expected_last - last_observed_seq
                tail_gap = int(max(0, (expected_last - int(g[seq_col].iloc[-1]))))
                g.loc[g.index[-1], "loss_assigned"] += tail_gap
        return g

    return out.groupby(list(group_cols), group_keys=False).apply(_per_group, include_groups=False)


# --- Update your bin/stats to use loss_assigned if present
def bin_stats(df: pd.DataFrame, metric_col: str, bins) -> pd.DataFrame:
    """
    Build KPI bins once, then:
      - loss stats from all rows with KPI (no latency requirement)
      - latency stats from rows that also have latency
    """
    # 1) KPI-only view (for loss)
    kpi_only = df[[metric_col]].copy().dropna(subset=[metric_col])
    cat_all = pd.cut(kpi_only[metric_col], bins=bins, include_lowest=True)

    # losses
    if "loss_assigned" in df.columns:
        loss_series = df.loc[kpi_only.index, "loss_assigned"].fillna(0).astype(int)
    elif "Discarded" in df.columns:
        loss_series = df.loc[kpi_only.index, "Discarded"].astype(bool).astype(int)
    else:
        loss_series = pd.Series(0, index=kpi_only.index, dtype=int)

    g_all = kpi_only.groupby(cat_all, observed=True)
    lost = loss_series.groupby(cat_all, observed=True).sum()
    n_recv = g_all.size()                          # number of received rows (regardless of latency availability)
    expected = (n_recv + lost).astype(float)
    loss_rate = (lost / expected).fillna(0)

    # metric centers for plotting
    metric_center = g_all[metric_col].mean()

    # 2) Latency view (requires latency)
    lat_df = df[[metric_col, "Uplink_Latency"]].copy().dropna(subset=[metric_col, "Uplink_Latency"])
    cat_lat = pd.cut(lat_df[metric_col], bins=bins, include_lowest=True)
    g_lat = lat_df.groupby(cat_lat, observed=True)
    avg_latency = g_lat["Uplink_Latency"].mean()

    # 3) Combine on the categorical bins (union)
    stats = pd.DataFrame({
        "metric_center": metric_center,
        "lost": lost,
        "n": n_recv,
        "expected": expected,
        "loss_rate": loss_rate,
    })

    stats_lat = pd.DataFrame({"avg_latency_ms": avg_latency})
    stats = stats.join(stats_lat, how="left")  # keep bins even if no latency points

    # Clean up & numeric index for plotting
    stats = stats.reset_index(drop=True)
    return stats

def fixed_edges(xmin, xmax, step):
    # step is your bin width (use your "lim")
    # make sure the right edge is included
    return np.arange(xmin, xmax + step, step, dtype=float)

def bin_stats_fixed(df: pd.DataFrame, metric_col: str, edges) -> pd.DataFrame:
    # 1) KPI-only view: loss & counts (no latency requirement)
    kpi_only = df[[metric_col]].dropna(subset=[metric_col]).copy()
    cats = pd.cut(kpi_only[metric_col], bins=edges, include_lowest=True)
    all_bins = pd.IntervalIndex.from_breaks(edges, closed=cats.cat.categories.closed)

    if "loss_assigned" in df.columns:
        loss_series = df.loc[kpi_only.index, "loss_assigned"].fillna(0).astype(int)
    elif "Discarded" in df.columns:
        loss_series = df.loc[kpi_only.index, "Discarded"].astype(bool).astype(int)
    else:
        loss_series = pd.Series(0, index=kpi_only.index, dtype=int)

    n_recv = kpi_only.groupby(cats, observed=True).size().reindex(all_bins, fill_value=0)
    lost   = loss_series.groupby(cats, observed=True).sum().reindex(all_bins, fill_value=0)

    expected  = (n_recv + lost).astype(float)
    loss_rate = (lost / expected).where(expected > 0, 0.0)

    metric_center = kpi_only.groupby(cats, observed=True)[metric_col].mean() \
                            .reindex(all_bins)
    # fall back center to bin midpoints when a bin has no KPI samples
    mids = (pd.Series([iv.left for iv in all_bins]).values +
            pd.Series([iv.right for iv in all_bins]).values) / 2
    metric_center = metric_center.fillna(pd.Series(mids, index=all_bins))

    # 2) Latency view (requires latency)
    lat_df = df[[metric_col, "Uplink_Latency"]].dropna(subset=[metric_col, "Uplink_Latency"]).copy()
    lat_cats = pd.cut(lat_df[metric_col], bins=edges, include_lowest=True)
    avg_latency = lat_df.groupby(lat_cats, observed=True)["Uplink_Latency"].mean() \
                        .reindex(all_bins)

    stats = pd.DataFrame({
        "metric_center": metric_center.values,
        "avg_latency_ms": avg_latency.values,   # may be NaN where no latency in bin
        "loss_rate": loss_rate.values,
        "n": n_recv.values,
        "lost": lost.values,
        "expected": expected.values,
    })
    return stats


def plot_latency_vs_metric(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    bins: np.ndarray | list | None = None,
    marker_size: int = 6,
    xlim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    ylim: tuple[float, float] | None = None,
    rotate_xticks: bool = True,
):
    if metric_col not in df.columns:
        print(f"[Aviso] No existe la columna '{metric_col}'.")
        return None

    scatter_df = df[[metric_col, "Uplink_Latency"]].dropna()
    if scatter_df.empty:
        print("[Aviso] No hay datos para graficar.")
        return None

    if bins is None:
        lo, hi = np.nanpercentile(scatter_df[metric_col], [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None
        bins = np.linspace(lo, hi, 25)

    stats = bin_stats(df, metric_col, bins)

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2), dpi=150)
    if "nr" in metric_col:
        color_ = 'orange'
    else:
        color_ = 'tab:blue'
    
    ax1.scatter(
        scatter_df[metric_col].values,
        scatter_df["Uplink_Latency"].values,
        s=marker_size, alpha=0.7, marker="*",
        color=color_, label="Per-packet latency"
    )
    if not stats.empty:
        ax1.plot(
            stats["metric_center"].values,
            stats["avg_latency_ms"].values,
            linewidth=2, color="tab:green", label="Average latency"
        )
    ax1.set_xlabel(metric_col.replace("_", " "))
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(title)
    ax1.grid(True, which="major", axis="both", alpha=0.25)
    if xlim is not None:
        ax1.set_xlim(xlim)
    if xticks is not None:
        ax1.set_xticks(xticks)
    if rotate_xticks:
        ax1.tick_params(axis="x", rotation=90)
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Packet loss rate")
    ax2.set_ylim(0, 1)
    if not stats.empty:
        ax2.plot(
            stats["metric_center"].values,
            stats["loss_rate"].values,
            linestyle="--", linewidth=2, color="tab:red", label="Packet loss rate"
        )
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    plt.tight_layout()
    return ax1, ax2


def ecdf(series):
    """Return sorted values and ECDF y-values."""
    x = np.asarray(series.dropna())
    if x.size == 0:
        return None, None
    x_sorted = np.sort(x)
    y = np.arange(1, x_sorted.size + 1) / x_sorted.size
    return x_sorted, y

def ecdf_percentiles(series):
    """Compute selected ECDF percentiles (ignore NaN)."""
    series = series.dropna()
    return {
        "80%": np.percentile(series, 80),
        "90%": np.percentile(series, 90),
        "95%": np.percentile(series, 95),
        "99%": np.percentile(series, 99)
    }

    