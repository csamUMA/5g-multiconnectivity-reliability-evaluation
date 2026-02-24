# link_aggregation.py
import numpy as np
import pandas as pd

# Link Aggregation: Combines data from multiple interfaces into a single logical interface.
def simulate_link_aggregation(subset, mbps_target: float):

    print(f"Simulating Link Aggregation with target Mbps: {mbps_target*2}")

    #print(subset.iloc[0])

    # Orden temporal/por secuencia
    subset_link_aggreg = subset.sort_values("Seq").copy()

    # Hacemos link aggregation simple
    subset_link_aggreg.loc[subset_link_aggreg['Interface'] == 1, 'Seq'] *= 2
    subset_link_aggreg.loc[subset_link_aggreg['Interface'] == 2, 'Seq'] *= 2
    subset_link_aggreg.loc[subset_link_aggreg['Interface'] == 2, 'Seq'] += 1

    # Último Seq total (usamos max normal, no np.max)
    last_seq = subset_link_aggreg["Seq"].max()

    return subset_link_aggreg

# Full Duplication: For each Seq, choose the interface with minimum latency.
def simulate_full_duplication(subset: pd.DataFrame):
    """
    Implements Full Duplication:
    For each Seq, choose the interface with minimum latency.
    """

    # Orden temporal/por secuencia
    subset = subset.sort_values("Seq").copy()

    subset_1 = subset[subset['Interface'] == 1].copy()
    subset_2 = subset[subset['Interface'] == 2].copy()

    # Máximo Seq disponible en cada interfaz
    max_seq_1 = subset_1["Seq"].max()
    max_seq_2 = subset_2["Seq"].max()

    # Último Seq total (usamos max normal, no np.max)
    last_seq = int(max(max_seq_1, max_seq_2))

    # Guardaremos la decisión por cada Seq en un dict
    iface_by_seq = {}

    current_iface = 3
    last_switch_ms = subset["Now_ms"].iloc[0]

    # Recorremos todos los posibles Seq desde 0 hasta last_seq (incluido)
    rows = []   # aquí acumulamos las filas correctas

    for seq in range(0, last_seq + 1):

        r1 = subset_1[subset_1['Seq'] == seq]
        r2 = subset_2[subset_2['Seq'] == seq]
        
        # Elegimos fila según la interfaz actual, con fallback a la otra
        default_row = r1
        other_row = r2

        # row es un DataFrame -> cogemos la primera fila como Serie
        def_row = default_row.iloc[0]
        oth_row = other_row.iloc[0]

        lat_def = def_row["Uplink_Latency"]
        lat_other = oth_row["Uplink_Latency"]
        
        if lat_def < lat_other:
            r = def_row 
            r["Interface"] = 3
        else:
            r = oth_row 
            r["Interface"] = 4

        rows.append(r.to_dict())
        
        iface_by_seq[seq] = current_iface

    subset_swap = pd.DataFrame(rows)
    return subset_swap



# Partial Duplication: For each Seq:
#      - Choose reference row from subset[best_latency_col] (best latency decision).
#      - If bad conditions => duplicate: output rows for Interface=1 and Interface=2.    
#      - Else => output only for the best interface.
def simulate_partial_duplication_strategy(
    subset: pd.DataFrame,
    rsrp_col: str = "rsrp_lte",
    tx_col: str = "tx_power_dbm",
    latency_col: str = "Uplink_Latency",
    best_latency_col: str = "best_latency",
    lat_bad_th: float = 150.0,
    rsrp_bad_th: float = -100.0,
    tx_bad_th: float = 20.8,
    mode: str = "only_latency",
    default_iface: int = 1,
):
    """
    For each Seq:
      - Choose reference row from subset[best_latency_col] (best latency decision).
      - If bad conditions => duplicate: output rows for Interface=1 and Interface=2.
      - Else => output only for the best interface.

    Output has:
      - duplication (0/1)
      - best_iface (1/2)
    """

    subset = subset.sort_values(["Seq", "Packet_Sent"]).copy()

    def _to_float(x):
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return np.nan
            x = x.iloc[0]
        try:
            return float(x)
        except Exception:
            return np.nan

    max_seq = int(subset["Seq"].max())
    rows_out = []

    # Pre-split for speed
    if default_iface == 1:
        primary_iface = subset[subset["Interface"] == 1].copy()
        secondary_iface = subset[subset["Interface"] == 2].copy()
    else:
        primary_iface = subset[subset["Interface"] == 2].copy()
        secondary_iface = subset[subset["Interface"] == 1].copy()
    
    do_duplicate = False
    old_decision_duplication = False
    time_4_change = 0
    enable_comparison = False
    trigger_change = False
    
    for seq in range(0, max_seq + 1):
        
        primary_row_df = primary_iface[primary_iface["Seq"] == seq].copy()
        secondary_row_df = secondary_iface[secondary_iface["Seq"] == seq].copy()        
        primary_row = primary_row_df.iloc[0]     
        secondary_row = secondary_row_df.iloc[0] 
        
        t_ms = _to_float(primary_row.get("Packet_Sent", np.nan))

        if trigger_change and (t_ms >= time_4_change):
            do_duplicate = not do_duplicate
            trigger_change = False

        out = dict(primary_row.to_dict())

        rsrp_primary = _to_float(primary_row[rsrp_col])
        tx_primary   = _to_float(primary_row[tx_col])
        lat_primary  = _to_float(primary_row[latency_col])
        lat_secondary  = _to_float(secondary_row[latency_col])

        if do_duplicate:
            latency = _to_float(primary_row[best_latency_col])
            out["Interface"] = 3
        else:
            latency = lat_primary
            out["Interface"] = default_iface

        out["Packet_Sent"] = t_ms
        out["Mbps_x100"] = _to_float(primary_row.get("Mbps_x100", np.nan))
        out["has_5G"] = primary_row.get("has_5G", np.nan)
        out[rsrp_col] = rsrp_primary
        out[tx_col] = tx_primary
        out[latency_col] = latency
        rows_out.append(out)
        
        #Compare primary link to duplicate or not decision thresholds
        bad_radio        = (rsrp_primary < rsrp_bad_th) or (tx_primary > tx_bad_th)
        bad_rsrp_latency = (rsrp_primary < rsrp_bad_th) or (lat_primary > lat_bad_th)
        bad_tx_latency   = (tx_primary > tx_bad_th) or (lat_primary > lat_bad_th)
        bad_full         = (rsrp_primary < rsrp_bad_th) and (tx_primary > tx_bad_th) or (lat_primary > lat_bad_th)

        old_decision_duplication = do_duplicate

        if mode == "rsrp_only":
            do_duplicate = (rsrp_primary < rsrp_bad_th)
        elif mode == "tx_only": 
            do_duplicate = (tx_primary > tx_bad_th)
        elif mode == "latency_only":
            do_duplicate = (lat_primary > lat_bad_th)
        elif mode == "rsrp_tx":
            do_duplicate = bad_radio
        elif mode == "rsrp_latency":
            do_duplicate = bad_rsrp_latency
        elif mode == "tx_latency":
            do_duplicate = bad_tx_latency
        elif mode == "full":
            do_duplicate = bad_full

        if do_duplicate != old_decision_duplication:
            rtt = min(_to_float(primary_row.get("RTT", 0)), 800) # 800 is as late loss threshold, so if RTT is missing or very high, we use that as the time to next decision point

            if mode == "rsrp_only" or mode == "tx_only" or mode == "rsrp_tx":
                rtt = 0
            elif mode == "full":
                if (rsrp_primary < rsrp_bad_th) or (tx_primary > tx_bad_th):
                    rtt = 0
            elif mode == "rsrp_latency":
                if (rsrp_primary < rsrp_bad_th):
                    rtt = 0
            elif mode == "tx_latency":
                if (tx_primary > tx_bad_th):
                    rtt = 0
            
            if trigger_change == False or (t_ms + rtt < time_4_change):
                time_4_change = t_ms + rtt
                trigger_change = True
            
        do_duplicate = old_decision_duplication

    return pd.DataFrame(rows_out)


# Switching: For each Seq, choose one interface as active. Switch if conditions met.
def simulate_switching_strategy(
    subset: pd.DataFrame,
    default_iface: int,                     # this is your primary operator p
    mode: str = "rsrp_only",                # "rsrp_only" | "tx_only" | "latency_only" | "radio_combo" | "full"
    rsrp_col: str = "rsrp_lte",
    tx_col: str = "tx_power_dbm",
    latency_col: str = "Uplink_Latency",
    # thresholds (theta_*)
    rsrp_th: float = -100.0,                # θ_R
    tx_th: float = 21.0,                    # θ_U
    lat_th: float = 150.0,                  # θ_L
    # sampling periods
    radio_period_ms: int = 1000,            # 1 Hz
    latency_period_ms: int = 100,           # 10 Hz
    # anti ping-pong
    min_dwell_ms: int = 1000,
    # packet loss encoding
    loss_latency_value: float = 10000.0,

    # score policy params (S_i)
    w_R: float = 0.7,
    w_U: float = 0.8,
    w_L: float = 1.0,
    delta_R: float = 10.0,                  # Δ_R (e.g. 10 dB)
    delta_U: float = 5.0,                   # Δ_U (pick something sensible, tune as needed)
    delta_L: float = 100.0,                 # Δ_L (e.g. 100 ms)
):
    """
    Implements the rules exactly as in your LaTeX.

    Mode mapping:
      - "rsrp_only": policy (i)
      - "tx_only": policy (ii)
      - "latency_only": policy (iii)
      - "rsrp_tx": score policy with w_L forced to 0 (RSRP+Tx)
      - "rsrp_latency": score policy with w_U forced to 0 (RSRP+Latency)
      - "tx_latency": score policy with w_R forced to 0 (Tx+Latency)
      - "full": score policy using all terms (RSRP+Tx+Latency)
    """

    assert default_iface in (1, 2), "default_iface must be 1 or 2"
    p = int(default_iface)
    q = 1 if p == 2 else 2

    subset = subset.sort_values(["Seq", "Packet_Sent"]).copy()

    s1 = subset[subset["Interface"] == 1].copy()
    s2 = subset[subset["Interface"] == 2].copy()

    max_seq_1 = s1["Seq"].max() if not s1.empty else -1
    max_seq_2 = s2["Seq"].max() if not s2.empty else -1
    last_seq = int(max(max_seq_1, max_seq_2))
    if last_seq < 0:
        return pd.DataFrame()

    # --- state ---
    current_iface = p
    t0 = float(subset["Packet_Sent"].iloc[0])
    last_switch_ms = t0

    last_radio_eval_ms = t0 - radio_period_ms
    last_lat_eval_ms   = t0 - latency_period_ms

    # last seen full row per iface (for copying metadata out)
    last_row = {1: None, 2: None}

    def _to_float(x):
        try:
            v = float(x)
            return v
        except Exception:
            return np.nan

    def _update_metric(iface: int, row: pd.Series):
        # RSRP
        rv = _to_float(row.get(rsrp_col, np.nan))

        # TX
        tv = _to_float(row.get(tx_col, np.nan))

        # LAT
        lv = _to_float(row.get(latency_col, np.nan))

    def _score(r, u, l, wR, wU, wL):
        # require all used terms finite on this iface
        if wR != 0 and not np.isfinite(r): return np.nan
        if wU != 0 and not np.isfinite(u): return np.nan
        if wL != 0 and not np.isfinite(l): return np.nan

        eR = max(0.0, rsrp_th - r)          # θ_R - r  (positive when r < θ_R)
        eU = max(0.0, u - tx_th)            # u - θ_U  (positive when u > θ_U)
        eL = max(0.0, l - lat_th)           # l - θ_L  (positive when l > θ_L)

        SR = (wR * (eR / delta_R)) if wR != 0 else 0.0
        SU = (wU * (eU / delta_U)) if wU != 0 else 0.0
        SL = (wL * (eL / delta_L)) if wL != 0 else 0.0
        return SR + SU + SL

    rows_out = []

    time_4_change = 0
    time_4_compare = 0
    min_dwell_ms = latency_period_ms

    decission_change = current_iface

    enable_comparison = True
    
    for seq in range(0, last_seq + 1):
        r1_df = s1[s1["Seq"] == seq]
        r2_df = s2[s2["Seq"] == seq]

#        if not r1_df.empty:
#            last_row[1] = r1_df.iloc[0]
#            _update_metric(1, last_row[1])
#        if not r2_df.empty:
#            last_row[2] = r2_df.iloc[0]
#            _update_metric(2, last_row[2])

#        if last_row[1] is None and last_row[2] is None:
#            continue

        if current_iface == 1:
            ref_row = r1_df.iloc[0]
            other_ref_row = r2_df.iloc[0]
        else:
            ref_row = r2_df.iloc[0]
            other_ref_row = r1_df.iloc[0]

        # pick a time reference (prefer current iface row, else the other)
#        ref_row = last_row[current_iface] if last_row[current_iface] is not None else last_row[q if current_iface == p else p]
        t_ms = _to_float(ref_row.get("Packet_Sent", np.nan))

        if t_ms > time_4_compare:
            time_4_compare = t_ms + latency_period_ms
            enable_comparison = True
        else:
            enable_comparison = False

        if t_ms > time_4_change:
            current_iface = decission_change
        
        # output: copy current iface row if possible, but overwrite Interface and fill metrics
        out_base = ref_row.copy()
        out = dict(out_base.to_dict())
        out["Interface"] = current_iface

        # fill metrics with last-known values for *selected* iface
        out[rsrp_col] = ref_row[rsrp_col]
        out[tx_col]   = ref_row[tx_col]
        out[latency_col] = ref_row[latency_col]

        rows_out.append(out)

        if enable_comparison == False:
            continue        

        # fetch current metrics for p and q (sample-and-hold)
        r_p, u_p, l_p = ref_row[rsrp_col], ref_row[tx_col], ref_row[latency_col]
        r_q, u_q, l_q = other_ref_row[rsrp_col], other_ref_row[tx_col], other_ref_row[latency_col]

        if mode == "baseline":
            continue

        # ---------------------------
        # (i) RSRP-only (1 Hz)
        # ---------------------------

        if mode == "rsrp_only":
            if current_iface == p:
                # failover: if r_p < θ_R and r_q > r_p
                if (r_p < rsrp_th) and (r_q > r_p):
                    decission_change = q
            else:
                # fallback: if r_p >= θ_R or r_p >= r_q
                if (r_p >= rsrp_th) or (r_p >= r_q):
                    decission_change = p
            continue

        # ---------------------------
        # (ii) Tx-only (1 Hz)
        # ---------------------------
        if mode == "tx_only":
            if current_iface == p:
                # failover: u_p > θ_U and u_q < u_p
                if (u_p > tx_th) and (u_q < u_p):
                    decission_change = q
            else:
                # fallback: u_p <= θ_U or u_p <= u_q
                if (u_p <= tx_th) or (u_p <= u_q):
                    decission_change = p
            continue

        # ---------------------------
        # (iii) Latency-only (100 ms)
        # ---------------------------
        if mode == "latency_only":
            if current_iface == p:
                # failover: l_p > θ_L and l_q < l_p
                if (l_p > lat_th) and (l_q < l_p):
                    decission_change = q
            else:
                # fallback: l_p <= θ_L or l_p <= l_q
                if (l_p <= lat_th) or (l_p <= l_q):
                    decission_change = p
            continue

        # ---------------------------
        # Combined score policies
        # radio_combo => w_L = 0 (RSRP+Tx)
        # full       => use w_L (RSRP+Tx+Latency)
        # ---------------------------
        if mode in ("radio_combo", "rsrp_latency", "tx_latency", "full"):
            if mode == "radio_combo":
                wL_eff = 0.0
            elif mode == "rsrp_latency":
                w_U = 0.0
            elif mode == "tx_latency":
                w_R = 0.0

            # compute scores; skip if any used term missing
            S_p = _score(r_p, u_p, l_p, w_R, w_U, w_L)
            S_q = _score(r_q, u_q, l_q, w_R, w_U, w_L)

            if current_iface == p:
                # If attached to p and S_p>0 and S_q<S_p => switch to q
                if (S_p > 0.0) and (S_q < S_p):
                    decission_change = q
            else:
                # If attached to q and (S_p==0) or (S_p<=S_q) => return to p
                if (S_p == 0.0) or (S_p <= S_q):
                    decission_change = p
            continue

        if current_iface != decission_change:
            rtt = min(_to_float(ref_row.get("RTT", 0)), _to_float(other_ref_row.get("RTT", 0)))

            if mode == "rsrp_only" or mode == "tx_only" or mode == "radio_combo":
                rtt = 0
            
            time_4_change = t_ms + rtt

        raise ValueError(f"Unknown mode: {mode}")

    return pd.DataFrame(rows_out)