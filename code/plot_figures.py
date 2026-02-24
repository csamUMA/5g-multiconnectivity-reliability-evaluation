import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from pathlib import Path

MEASUREMENT_COL = ["Uplink_Latency", "Downlink_latency"]  # , "RTT"
technologies = ["lte", "nr"]
kpis = ["rsrp_", "rsrq_", "rssi_", "sinr_"]

MEASUREMENT_COL = ["Uplink_Latency"]

kpis = ["rsrp_", "rsrq_", "rssi_", "sinr_"]
kpis = ["rsrp_"]
#kpis = ["sinr_"]
technologies = ["lte"]

use_ONLY_radio = False   # 0 = "FULL", 1 = "RADIO ONLY"

# broken axis ranges
Y_LOW_MAX  = 3500    # ms
Y_HIGH_MIN = 9900    # ms  -> packets above this are considered "loss" / timeout

INTERFACES_ = ["5G_1"]


def plot_rsrp_vs_latency_distribution(
    subset: pd.DataFrame,
    mbps: float,
    scenario: str,
    interface: str = "5G_1",
    latency_col: str = "Uplink_Latency",
    kpi: str = "rsrp",
    tech: str = "lte",
):

    kpi = f"{kpi}_{tech}"

    print(f"\n\n******** MEASUREMENT: {latency_col.upper()} ********\n")
    print(f"\nKPI: {kpi}\n")

    # RSSI not available on NR
    if kpi == "rssi_nr":
        print("Skipping RSSI for NR (not available).")
        return

    # Only rows where measurement exists
    subset_m = subset.dropna(subset=[latency_col, "has_5G"]).copy()

    #subset_m.sort_values(kpi, inplace=True)
    subset.sort_values(kpi, inplace=True)

    kpi_vals = subset[kpi].values
    has_5G_vals = subset["has_5G"].values

    if len(kpi_vals) < 10:
        return

    # ------------------------------------------
    # 2) Define bins & labels for bottom plot
    # ------------------------------------------
    if "rsrp" in kpi or "rssi" in kpi:
        step_ = 5
        bins = np.arange(kpi_vals.min(), kpi_vals.max(), step_)
        bins = np.arange(-126, -51, step_)
        #bins = np.arange(-106, -51, step_)
        labels = [f"[{bins[i]} {bins[i+1]}) dBm"
                    for i in range(len(bins) - 1)]
        x_label = "RSRP [dBm]" if "rsrp" in kpi else "RSSI [dBm]"
        legend_ = "RSRP" if "rsrp" in kpi else "RSSI"
    elif "rsrq" in kpi or "sinr" in kpi:
        step_ = 2
        bins = np.arange(kpi_vals.min(), kpi_vals.max(), step_)
        bins = np.arange(-14, 30, step_)
        labels = [f"[{bins[i]} {bins[i+1]}) dB"
                    for i in range(len(bins) - 1)]
        x_label = "RSRQ [dB]" if "rsrq" in kpi else "SINR [dB]"
        legend_ = "RSRQ" if "rsrq" in kpi else "SINR"

    subset[f"{kpi}_bin"] = pd.cut(
        subset[kpi],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    #subset[f"{kpi}_bin"] = subset[f"{kpi}_bin"].cat.remove_unused_categories()

    cats = subset[f"{kpi}_bin"].cat.categories
    if len(cats) == 0:
        return

    # =====================================================
    # 3) Figure with 3 rows:
    #    row0: PDF
    #    row1: packet-loss percentage bars
    #    row2: latency 0–3500 ms
    # =====================================================
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1, 1, 3],
        hspace=0.05
    )

    ax_pdf        = fig.add_subplot(gs[0, 0])
    ax_loss       = fig.add_subplot(gs[1, 0])                       # packet-loss %
    ax_lat_bottom = fig.add_subplot(gs[2, 0])                       # 0–3500 ms

    # -------------------- TOP: KPI KDE PDF --------------------
    kde_kpi = gaussian_kde(kpi_vals)
    lo, hi = kpi_vals.min(), kpi_vals.max()
    x_grid = np.linspace(-146, hi, 400)
    pdf_kpi = kde_kpi(x_grid)

    alpha_band = 0.95
    lower_q = (1 - alpha_band) / 2.0
    upper_q = 1 - lower_q
    lower_b = np.percentile(kpi_vals, lower_q * 100.0)
    upper_b = np.percentile(kpi_vals, upper_q * 100.0)

    inside_mask = (x_grid >= lower_b) & (x_grid <= upper_b)
    left_mask   = x_grid < lower_b
    right_mask  = x_grid > upper_b

    ax_pdf.plot(x_grid, pdf_kpi, linewidth=2, color="#08306B",
                label=f"{legend_} PDF", rasterized=True)

    ax_pdf.fill_between(
        x_grid[left_mask],
        0,
        pdf_kpi[left_mask],
        facecolor="#9ECAE1",
        alpha=0.3,
        hatch="//",
        edgecolor="#08306B",
        linewidth=0.0,
        label=f"Residual (outside central {int(alpha_band*100)}%)",
        zorder=1
    )
    ax_pdf.fill_between(
        x_grid[right_mask],
        0,
        pdf_kpi[right_mask],
        facecolor="#9ECAE1",
        alpha=0.3,
        hatch="//",
        edgecolor="#08306B",
        linewidth=0.0,
        zorder=1
    )
    ax_pdf.fill_between(
        x_grid[inside_mask],
        0,
        pdf_kpi[inside_mask],
        facecolor="#9ECAE1",
        alpha=0.6,
        label=f"Central {int(alpha_band*100)}% band",
        zorder=2
    )

    mu_kpi = np.mean(kpi_vals)
    ax_pdf.axvline(mu_kpi, color="black", linestyle="-",
                    linewidth=1, label="μ")

    ax_pdf.set_ylabel("PDF")
    ax_pdf.grid(True, axis='y', alpha=0.25)

    ax_pdf.legend(frameon=False)
    

    # ---------------- BOTTOM: latency vs KPI (0–3500 ms) -----
    x_positions = np.arange(len(cats))

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_blues",
        ["#08306B", "#2171B5", "#6BAED6", "#9ECAE1"],
        N=len(cats)
    )

    custom_cmap_oranges = LinearSegmentedColormap.from_list(
        "custom_oranges",
        ["#E6550D", "#FD8D3C", "#FDAE6B", "#FDD0A2"], # Dark to light orange
        N=len(cats)
    )

    palette = [custom_cmap(i / max(len(cats) - 1, 1))
                for i in range(len(cats))]
    palette_orange = [custom_cmap_oranges(i / max(len(cats) - 1, 1))
                for i in range(len(cats))]

    total_points = len(subset)
    

    bin_counts = np.zeros(len(cats), dtype=int)
    loss_counts = np.zeros(len(cats), dtype=int)

    for x_pos, (cat, color, color_pdf) in enumerate(zip(cats, palette, palette_orange)):
        
        # 1. RETRIEVE DATA FOR LOSS CALCULATION
        # Get all latency values for the current bin, dropping only NaNs in the measurement column.
        y_vals_all_loss = subset.loc[
            subset[f"{kpi}_bin"] == cat, latency_col
        ].dropna().values

        #print(f"y_vals_all_loss dtype: {y_vals_all_loss.dtype}")
        #print(f"Max value in bin: {y_vals_all_loss.max()}")
        #print(f"Number of points >= Y_HIGH_MIN: {np.sum(y_vals_all_loss >= Y_HIGH_MIN)}")
        
        if len(y_vals_all_loss) < 2:
            continue
        
        # Calculate counts based on the loss data set
        bin_counts[x_pos] = len(y_vals_all_loss)
        #print(f"Bin '{cat}': total points = {bin_counts[x_pos]}")
        loss_counts[x_pos] = np.sum(y_vals_all_loss >= Y_HIGH_MIN)
        #print(f"Bin '{cat}': loss points = {loss_counts[x_pos]}")

        # ------------------------------------------------------------------
        # 2. RETRIEVE DATA FOR PLOTTING (requires has_5G)
        # We only need this for the KDE and Jitter plot, not for loss %
        # ------------------------------------------------------------------
        bin_data = subset.loc[
            subset[f"{kpi}_bin"] == cat
        ].dropna(subset=[latency_col, "has_5G"]).copy()
        
        y_vals_all = bin_data[latency_col].values
        has_5G_for_bin = bin_data["has_5G"].values
        
        # normal (non-timeout) latencies
        normal_mask = (y_vals_all < Y_HIGH_MIN)
        y_normal = y_vals_all[normal_mask]
        has_5G_normal = has_5G_for_bin[normal_mask]

        if len(y_normal) >= 2:
            kde_lat = gaussian_kde(y_normal)
            y_grid = np.linspace(y_normal.min(), y_normal.max(), 400)
            pdf_lat = kde_lat(y_grid)
            pdf_lat = pdf_lat / pdf_lat.max()

            bin_fraction = len(y_vals_all) / total_points
            pdf_scaled = pdf_lat * bin_fraction

            width = 0.4
            gap = 0.12

            # only show the part of KDE up to 3500 ms
            mask_low = (y_grid <= Y_LOW_MAX)
            if mask_low.any():
                x_right_low = x_pos + pdf_scaled[mask_low] * width / max(bin_fraction, 0.01)
                ax_lat_bottom.fill_betweenx(
                    y_grid[mask_low],
                    x_pos,
                    x_right_low,
                    facecolor=color_pdf,
                    alpha=0.95
                )
        else:
            width = 0.4
            gap = 0.12

        # center vertical line in bottom panel
        ax_lat_bottom.axvline(x_pos, color="0.90", linewidth=1, zorder=0)

        # scatter for y_normal within 0–3500
        low_mask = (y_normal <= Y_LOW_MAX)
        y_low_points = y_normal[low_mask]
        # Corresponding 'has_5G' values for the low points
        has_5G_low_points = has_5G_normal[low_mask]

        if len(y_low_points) > 0:
            x_left_low = x_pos - gap - np.random.rand(len(y_low_points)) * width
            
            # MODIFIED: Use a list comprehension for reliable color assignment
            scatter_colors = [
                #'pink' if is_5g == 0 else color 
                'pink' if False else color 
                for is_5g in has_5G_low_points
            ]
            
            ax_lat_bottom.scatter(
                x_left_low,
                y_low_points,
                s=14,
                alpha=0.9,
                color=scatter_colors, # MODIFIED
                edgecolors="black",
                linewidths=0.25
            )

    # ---- MIDDLE: packet-loss percentage per bin ----
    loss_pct = np.where(
        bin_counts > 0,
        loss_counts / bin_counts * 100.0,
        0.0
    )

    bars = ax_loss.bar(
        x_positions,
        loss_pct,
        color="#9ECAE1",
        edgecolor="#08306B",
        linewidth=0.8,
        label="Packet loss (%)"
    )

    ax_loss.set_ylim(0, max(5, loss_pct.max() * 1.2))
    ax_loss.set_ylabel("Packet loss [%]")
    ax_loss.grid(True, alpha=0.25)

    # hide x tick labels on middle panel (bottom has them)
    ax_loss.set_xticks(x_positions)
    ax_loss.set_xticklabels([])

    # optional: add values on top of bars if you want
    # for x, val in zip(x_positions, loss_pct):
    #     if val > 0:
    #         ax_loss.text(x, val + 0.5, f"{val:.1f}",
    #                     ha="center", va="bottom", fontsize=8)

    # bottom axis formatting
    ax_lat_bottom.set_ylim(0, Y_LOW_MAX)
    ax_lat_bottom.set_xticks(x_positions)
    ax_lat_bottom.set_xticklabels(cats, rotation=45, ha="right")
    ax_lat_bottom.set_xlabel(x_label)

    ax_lat_bottom.set_ylabel("Latency [ms]")
    #ax_lat_bottom.set_title(f"{scenario} for {mbps} Mbps")

    # diagonal arrow & texts on the bottom band
    ax_lat_bottom.annotate(
        "",
        xy=(0.55, 0.9), xycoords="axes fraction",
        xytext=(0.3, 0.9), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<-", lw=2.0, color="black"),
    )
    ax_lat_bottom.text(
        0.55, 0.9,
        f"Uplink Control Power Exceeds",
        transform=ax_lat_bottom.transAxes,
        color="darkred",
        fontsize=11,
        ha="center",
        va="bottom",
        #rotation=-25,
        rotation_mode="anchor"
    )
    
    """
    # diagonal arrow & texts on the bottom band
    ax_lat_bottom.annotate(
        "",
        xy=(0.68, 0.35), xycoords="axes fraction",
        xytext=(0.35, 0.65), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.0, color="black"),
    )
    ax_lat_bottom.text(
        0.40, 0.62,
        f"↓ {legend_}\n↑ Latency",
        transform=ax_lat_bottom.transAxes,
        color="darkred",
        fontsize=11,
        ha="center",
        va="bottom",
        rotation=-25,
        rotation_mode="anchor"
    )
    ax_lat_bottom.text(
        0.62, 0.51,
        f"↑ {legend_}\n↓ Latency",
        transform=ax_lat_bottom.transAxes,
        color="seagreen",
        fontsize=11,
        ha="center",
        va="top",
        rotation=-25,
        rotation_mode="anchor"
    )
    """

    ax_lat_bottom.grid(True, alpha=0.25)
    # Force perfect alignment of PDF, packet-loss bars, and latency violins
    if "rsrp" in kpi:
        hspace_ = -0.8
    else:
        hspace_ = -0.0

    ax_lat_bottom.set_xlim(hspace_, len(cats))
    
    ax_loss.set_xlim(hspace_, len(cats))
    
    ax_pdf.set_xticklabels([])
    ax_pdf.set_xticks([])


    
    # Compute bin centers in KPI units
    bin_edges = bins
    bin_widths = np.diff(bin_edges)
    bin_centers_all = bin_edges[:-1] + bin_widths / 2

    # Map each used category (cats) to its bin center
    cats_list = list(cats)
    labels_list = list(labels)
    x_positions = np.array([bin_centers_all[labels_list.index(cat)] for cat in cats_list])

#                    print(bin_edges)

    # --- NEW: grey background from left axis to the 5th bin ---
    # left edge = current left x-limit (hspace_)
    # right edge = just after the 5th bin center -> index 4 + 0.5
    right_edge = min(4, len(cats)) - 0.5    # in case you have <5 bins
    ax_lat_bottom.axvspan(
        hspace_,          # x-min
        right_edge,       # x-max
        facecolor="0.4",  # light grey
        alpha=0.4,
        zorder=-1         # keep it behind violins & points
    )

    # A "width" in KPI units for jitter / bar width
    bin_width = np.min(bin_widths)
    width = 0.8 * bin_width
    gap = 0.1 * bin_width
    #ax_pdf.set_xticks(False)

    ax_pdf.set_xlim(bin_edges[0] - 1, bin_edges[-1] + step_)   # optional, but keeps the top axis tidy
    
    ax_pdf.set_xlim(bin_edges[0]+1, bin_edges[-1] + step_)   # optional, but keeps the top axis tidy
    
    print("edges:,", bin_edges[0]  , bin_edges[-1])

    ax_loss.legend(frameon=False)

    #plt.tight_layout()

    BASE_DIR = Path(__file__).resolve().parent.parent
    FIGURES_DIR = BASE_DIR / "figures" / "RSRP_distribution_Latency"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = FIGURES_DIR / f"rsrp_latency_{interface}_{scenario}_{mbps}.pdf"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")



def plot_tx_power_vs_latency_distribution(
    subset: pd.DataFrame,
    mbps: float,
    scenario: str,
    interface: str = "5G_1",
    lat_col: str = "Uplink_Latency",
    kpi: str = "rsrp",
    tech: str = "lte",
):

    technologies = ["lte", "nr"]        

    y_axis_limit_lower = 3500  # ms
    y_axis_limit_upper = 9800  # ms
    y_axis_limit_total = 10040  # ms

    curves = "CDF" #PDF
            
    x = subset["tx_power_dbm"].values
    y = subset[lat_col].values

    # ----- 3-axes layout (broken y-axis on the left + marginals) -----
    plt.rcParams.update({
        "axes.labelsize": 14,      # axis label font size
        "xtick.labelsize": 12,     # x tick labels
        "ytick.labelsize": 12,     # y tick labels
        "legend.fontsize": 11,     # legend
    })
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(
        3, 2,
        height_ratios=[0.5, 2.5, 1.0],   # top latency band, bottom latency band, TX marginal
        width_ratios=[4, 1],             # left: scatter, right: latency marginal
        hspace=0.07,
        wspace=0.10,
        figure=fig,
    )

    # LEFT: broken y-axis scatter
    ax_scatter_top    = fig.add_subplot(gs[0, 0])                          # 9000–10000 ms
    ax_scatter_bottom = fig.add_subplot(gs[1, 0], sharex=ax_scatter_top)   # 0–3500 ms

    # RIGHT: broken y-axis latency PDF/ECDF
    ax_latpdf_top    = fig.add_subplot(gs[0, 1], sharey=ax_scatter_top)
    ax_latpdf_bottom = fig.add_subplot(gs[1, 1], sharey=ax_scatter_bottom)

    # BOTTOM: TX-power marginal
    ax_txpdf = fig.add_subplot(gs[2, 0], sharex=ax_scatter_top)
    ax_dummy = fig.add_subplot(gs[2, 1]); ax_dummy.axis("off")

    # -------------------------------
    # Masks for latency ranges
    # -------------------------------
    bottom_mask  = (subset[lat_col] <= y_axis_limit_lower)
    top_mask     = (subset[lat_col] >= y_axis_limit_upper)
    special_mask = (subset[lat_col] == 10000)   # your ~10000 ms packets

    no5g_mask = (subset["has_5G"] == 0)
    yes5g_mask = (subset["has_5G"] == 1)

    # bottom band masks
    bottom_valid = (bottom_mask & ~special_mask)

    # 5G points → normal style (blue circles by default)
    ax_scatter_bottom.scatter(
        subset.loc[bottom_valid & yes5g_mask, "tx_power_dbm"],
        subset.loc[bottom_valid & yes5g_mask, lat_col],
        alpha=0.4,
        label="Latency (5G)",
        color="#287cb6ff",
        marker="o",
        s=10
    )

    # non-5G points → purple diamonds
    ax_scatter_bottom.scatter(
        subset.loc[bottom_valid & no5g_mask, "tx_power_dbm"],
        subset.loc[bottom_valid & no5g_mask, lat_col],
        alpha=0.4,
        #label="Latency (no 5G)",
        #color="purple",
        color="#287cb6ff",
        marker="D",     # ◇ diamond
        s=10
    )

    ax_scatter_bottom.set_ylim(0, y_axis_limit_lower)
    ax_scatter_bottom.set_ylabel(f"{lat_col} [ms]")
    ax_scatter_bottom.grid(True)

    # -------------------------------
    # LEFT: scatter, top band 9000–10000
    # -------------------------------
    ax_scatter_top.scatter(
        subset.loc[top_mask & ~special_mask, "tx_power_dbm"],
        subset.loc[top_mask & ~special_mask, lat_col],
        alpha=0.4
    )

    # Highlight ~10000 ms as red crosses
    ax_scatter_top.scatter(
        subset.loc[top_mask & special_mask, "tx_power_dbm"],
        subset.loc[top_mask & special_mask, lat_col],
        color="red",
        marker="x",
        s=50,
        linewidths=1.5,
        label="Latency = 10000 ms (Dropped packets)"
    )

    ax_scatter_bottom.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False
    )

    ax_scatter_top.set_ylim(y_axis_limit_upper, y_axis_limit_total)
    ax_scatter_top.grid(True)
    #ax_scatter_top.set_title(
    #    f"{lat_col} vs TX Power  |  scenario={scenario}, mbps={mbps}"
    #)

    # Hide x ticks on top scatter
    plt.setp(ax_scatter_top.get_xticklabels(), visible=False)

    # Legend on bottom panel
    # Collect handles from BOTH axes
    handles_bottom, labels_bottom = ax_scatter_bottom.get_legend_handles_labels()
    handles_top,    labels_top    = ax_scatter_top.get_legend_handles_labels()

    # Combine them
    handles = handles_bottom + handles_top
    labels  = labels_bottom  + labels_top

    # Create a unified legend on the bottom axis
    ax_scatter_bottom.legend(
        handles, labels,
        loc="upper left",
        fontsize=9
    )

    # -------------------------------
    # RIGHT: latency marginal (PDF / ECDF), broken as well
    # -------------------------------
    if curves == "CDF":
        # ECDF over all y
        y_clean = y[~np.isnan(y)]
        y_sorted = np.sort(y_clean)
        n = len(y_sorted)
        ecdf = np.arange(1, n + 1) / n

        # Top band 9000–10000
        ax_latpdf_top.step(ecdf, y_sorted, where="post")
        ax_latpdf_top.fill_betweenx(y_sorted, 0, ecdf, step="post", alpha=0.3)
        ax_latpdf_top.set_ylim(y_axis_limit_upper, y_axis_limit_total)
        ax_latpdf_top.set_xlim(0, 1)
        ax_latpdf_top.xaxis.tick_top()
        ax_latpdf_top.xaxis.set_label_position("top")
        ax_latpdf_top.set_xlabel("ECDF")
        ax_latpdf_top.grid(True)

        # Bottom band 0–3500
        ax_latpdf_bottom.step(ecdf, y_sorted, where="post")
        ax_latpdf_bottom.fill_betweenx(y_sorted, 0, ecdf, step="post", alpha=0.3)
        ax_latpdf_bottom.set_ylim(0, y_axis_limit_lower)
        ax_latpdf_bottom.set_xlim(0, 1)
        ax_latpdf_bottom.grid(True)
    else:
        # KDE over all y
        ys = np.linspace(np.nanmin(y), np.nanmax(y), 300)
        kde_lat = gaussian_kde(y)
        pdf_lat = kde_lat(ys)

        # Top band 9000–10000
        ax_latpdf_top.plot(pdf_lat, ys, rasterized=True)
        ax_latpdf_top.fill_betweenx(ys, pdf_lat, alpha=0.3)
        ax_latpdf_top.set_ylim(y_axis_limit_upper, y_axis_limit_total)
        ax_latpdf_top.invert_xaxis()
        ax_latpdf_top.xaxis.tick_top()
        ax_latpdf_top.xaxis.set_label_position("top")
        ax_latpdf_top.set_xlabel("PDF")
        ax_latpdf_top.grid(True)

        # Bottom band 0–3500
        ax_latpdf_bottom.plot(pdf_lat, ys, rasterized=True)
        ax_latpdf_bottom.fill_betweenx(ys, pdf_lat, alpha=0.3)
        ax_latpdf_bottom.set_ylim(0, y_axis_limit_lower)
        ax_latpdf_bottom.invert_xaxis()
        ax_latpdf_bottom.grid(True)

    # Hide y tick labels on right panels
    plt.setp(ax_latpdf_top.get_yticklabels(), visible=False)
    plt.setp(ax_latpdf_bottom.get_yticklabels(), visible=False)

    # -------------------------------
    # BOTTOM-LEFT: TX power marginal
    # -------------------------------
    if curves == "CDF":
        x_sorted = np.sort(x)
        n = len(x_sorted)
        ecdf_x = np.arange(1, n + 1) / n

        # 2) Collapse duplicates: use unique TX powers and the ECDF
        #    value at the *last* occurrence of each TX power
        x_unique, counts = np.unique(x_sorted, return_counts=True)
        cum_counts = np.cumsum(counts)          # cumulative # of samples
        ecdf_unique = cum_counts / n            # ECDF at each unique x

        # 3) Smooth monotonic interpolation on a finer grid
        x_fine = np.linspace(x_unique.min(), x_unique.max(), 1000)
        pchip = PchipInterpolator(x_unique, ecdf_unique)
        ecdf_smooth = pchip(x_fine)

        # 4) Plot smooth ECDF curve
        ax_txpdf.plot(x_fine, ecdf_smooth, linewidth=2, rasterized=True)

        # Optional: lightly show the original step ECDF underneath
        # ax_txpdf.step(x_sorted, ecdf_x, where="post", alpha=0.15)

        # Optional: fill under the smooth curve
        ax_txpdf.fill_between(x_fine, ecdf_smooth, alpha=0.3)

        ax_txpdf.set_ylabel("ECDF")
        ax_txpdf.set_xlabel("TX Power [dBm]")
        ax_txpdf.grid(True)
        ax_txpdf.set_ylim(0, 1.0)
    else:
        xs = np.linspace(x.min(), x.max(), 300)
        kde_tx = gaussian_kde(x)
        pdf_tx = kde_tx(xs)

        ax_txpdf.plot(xs, pdf_tx, rasterized=True)
        ax_txpdf.fill_between(xs, pdf_tx, alpha=0.3)
        ax_txpdf.set_ylabel("PDF")
        ax_txpdf.set_ylim(0, 0.3)

    ax_txpdf.set_xlabel("TX Power [dBm]")
    ax_txpdf.grid(True)

    plt.tight_layout()
    BASE_DIR = Path(__file__).resolve().parent.parent
    FIGURES_DIR = BASE_DIR / "figures" / "latency_vs_txpower"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = FIGURES_DIR / f"tx_power_latency_{scenario}_{mbps}.png"
    fig.savefig(out_name, bbox_inches="tight")
    plt.show()


def plot_rsrp_ultx_half_datarate(
    subset: pd.DataFrame,
    scenario: str,
    mbps_list: list,
    interface: str = "5G_2",
):
    def y_fmt(y, _):
        if abs(y - 7.4) < 1e-6:
            return "7.4"
        return f"{int(y)}"


    lines = ['-', '-.', ':']
    labels = ['1 Mbps', '2 Mbps', '4 Mbps']

    color_set = {
        400: "#1f77b4",
        200: "#ff7f0e",
        100: "#2ca02c",
        50:  "#474747ff",
        25:  "#9467bd",
    }

    RSRP_COL = "rsrp_lte"

    rsrp_min = subset[RSRP_COL].min()
    rsrp_max = subset[RSRP_COL].max()

    cmap = plt.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=rsrp_min, vmax=rsrp_max)

    x_ref = -85.0  # RSRP reference [dBm]

    fig_corr2d, ax = plt.subplots(figsize=(7, 6))

    # ── vertical reference line at -85 dBm (draw once) ──
    ax.axvline(
        x_ref,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7
    )

    # container for Y-axis ticks at intersections
    y_ref_info = []

    for mbps in mbps_list:

            subset_ready = subset[
                (subset['scenario'] == scenario) &
                (subset['mbps'] == mbps) &
                (subset['Interface'] == interface)
            ].copy()

            subset_plot = subset_ready[subset_ready['has_radio_parameters'] == 1].copy()            
            subset_plot["time_s"] = (subset_plot["Now_ms"] - subset_plot["Now_ms"].iloc[0]) / 1000.0

            corr_df = subset_plot[
                (subset_plot["Uplink_Latency"] != 10000) &
                (subset_plot["tx_power_dbm"] <= 22)
            ][[RSRP_COL, "tx_power_dbm"]].dropna()

            rsrp_median = corr_df[RSRP_COL].median()

            # ── color & marker by Mbps ──
            if mbps == mbps_list[0]:
                color_ = "black"
                marker_ = "+"
            else:
                color_ = "red"
                marker_ = "^"

            # ── scatter points ──
            ax.scatter(
                corr_df[RSRP_COL],
                corr_df["tx_power_dbm"],
                color=color_,
                marker=marker_,
                s=16,
                alpha=0.4
            )

            # ── quadratic fit ──
            x_fit = corr_df[RSRP_COL].to_numpy()
            y_fit = corr_df["tx_power_dbm"].to_numpy()

            if len(x_fit) < 3:
                continue

            a, b, c = np.polyfit(x_fit, y_fit, deg=2)

            x_line = np.linspace(x_fit.min(), x_fit.max(), 300)
            y_line = a * x_line**2 + b * x_line + c

            ax.plot(
                x_line,
                y_line,
                color=color_,
                linewidth=2,
                alpha=0.9,
                label=f"{mbps} Mbps",
                rasterized=True
            )

            # ── intersection at RSRP = -85 dBm ──
            y_ref = a * x_ref**2 + b * x_ref + c
            y_ref_info.append((y_ref, color_))


            # ── intersection at RSRP = -85 dBm ──
            y_ref = a * x_ref**2 + b * x_ref + c
            y_ref_info.append((y_ref, color_))

            # horizontal dashed line from y-axis to x_ref
            ax.hlines(
                y_ref,
                xmin=ax.get_xlim()[0],
                xmax=x_ref,
                colors=color_,
                linestyles="--",
                alpha=0.7
            )

  
            
    # ── add Y-axis ticks at intersection TX-power values ──
    yticks = ax.get_yticks()
    y_ref_ticks = [y for y, _ in y_ref_info]
    yticks = np.unique(np.concatenate([yticks, y_ref_ticks]))

    yticks = [-20, -10, 0, 7.4, 10, 11, 20]

    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))

    ax.set_yticks(np.sort(yticks))

    ax.set_xlabel("RSRP [dBm]")
    ax.set_ylabel("UL Tx Power [dB]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    ax.set_yticks(np.sort(yticks))

    ax.set_ylim(-20, 25)

    xticks = ax.get_xticks()
    xticks = np.unique(np.append(xticks, x_ref))
    ax.set_xticks(np.sort(xticks))

    ax.tick_params(axis='y', labelrotation=30)
    for label in ax.get_yticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

    plt.tight_layout()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    FIGURES_DIR = BASE_DIR / "figures" / "latenrsrp_tx_power_half_dataratecy_vs_txpower"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = FIGURES_DIR / f"UL_Tx_Pwr_vs_RSRP_4mbps_vs_0_25mbps.pdf"
    fig_corr2d.savefig(out_name, bbox_inches="tight")

    plt.show()
