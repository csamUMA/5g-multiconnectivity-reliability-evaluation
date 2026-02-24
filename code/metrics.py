# metrics.py

import numpy as np


def compute_statistics(df):
    LAT_COL = "Uplink_Latency"
    RSRP_COL = "rsrp_lte"
    PTX_COL = "tx_power_dbm"

    lat = df[LAT_COL].values
    rsrp = df[RSRP_COL].values
    ptx = df[PTX_COL].values

    use_counts = df["Interface"].value_counts(normalize=True) * 100.0
    pct_if1 = use_counts.get(1, 0.0)
    pct_if2 = use_counts.get(2, 0.0)

    stats = {

        "IF1_pct": pct_if1,
        "IF2_pct": pct_if2,

        "rsrp_p5": np.percentile(rsrp, 5),
        "rsrp_p50": np.percentile(rsrp, 50),
        "rsrp_p95": np.percentile(rsrp, 95),

        "ptx_p5": np.percentile(ptx, 5),
        "ptx_p50": np.percentile(ptx, 50),
        "ptx_p95": np.percentile(ptx, 95),
        "ptx_p99": np.percentile(ptx, 99),

        "lat_p90": np.percentile(lat, 90),
        "lat_p95": np.percentile(lat, 95),
        "lat_p99": np.percentile(lat, 99),
        "lat_p999": np.percentile(lat, 99.9),

        "loss_800_pct": np.mean(lat > 800) * 100,
        "loss_10000_pct": np.mean(lat == 10000) * 100,
        "below_150_pct": np.mean(lat <= 150) * 100,
    }

    return stats
