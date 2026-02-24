# paaf_switching_simulation.py
import numpy as np
import pandas as pd


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