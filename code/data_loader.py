# data_loader.py

import pandas as pd
from functions import fill_missing_packets, compute_best_latency
from pathlib import Path


def load_experiment_data(
    experiment: int,
    scenario_filter: str | None = None,
    mbps_filter: float | list[float] | None = None
):
    """
    Load experiment data with optional filtering.

    Parameters
    ----------
    experiment : int
        1 or 2
    scenario_filter : str or None
        Optional scenario to load (e.g., 'URBAN')
    mbps_filter : float or None
        Optional Mbps value to load (e.g., 4, 0.25)

    Returns
    -------
    df_full, df_radio, scenarios, mbps_list
    """

    interfaces = ["5G_1", "5G_2"]

    # Define experiment structure
    if experiment == 1:
        mbps_list = [4, 2, 1]
        scenarios = ["URBAN", "HYBRID", "RURAL"]
    elif experiment == 2:
        mbps_list = [4, 2, 1, 0.5, 0.25]
        scenarios = ["RURAL"]
    else:
        raise ValueError("Experiment must be 1 or 2")

    # Apply filters if provided
    if scenario_filter is not None:
        if scenario_filter not in scenarios:
            raise ValueError(f"Scenario '{scenario_filter}' not valid for experiment {experiment}")
        scenarios = [scenario_filter]

    if mbps_filter is not None:

        # Convert single value to list
        if not isinstance(mbps_filter, (list, tuple, set)):
            mbps_filter = [mbps_filter]

        # Validate values
        invalid = [m for m in mbps_filter if m not in mbps_list]
        if invalid:
            raise ValueError(
                f"Mbps value(s) {invalid} not valid for experiment {experiment}"
            )

        mbps_list = list(mbps_filter)

    df_radio = pd.DataFrame()
    df_full = pd.DataFrame()

    csv_models = ["radio", "full"]

    for csv_model in csv_models:
        for scenario in scenarios:
            for mbps in mbps_list:

                # Rural special case naming (experiment 1 only)
                factor = 10 if (experiment == 1 and scenario == "RURAL") else 1

                # Ensure mbps is numeric
                mbps_value = float(mbps)

                # Apply factor
                file_value = mbps_value * factor

                # Format cleanly (avoid .0)
                if file_value.is_integer():
                    file_mbps = str(int(file_value))
                else:
                    file_mbps = str(file_value)

                BASE_DIR = Path(__file__).resolve().parent.parent
                DATA_DIR = BASE_DIR / "data" 
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                file_name = DATA_DIR / f"df_{csv_model}_{scenario}_{file_mbps}mbps.csv"
                            
                try:
                    df_raw = pd.read_csv(file_name)
                    #df_raw.to_csv(out_name, index=False)
                except FileNotFoundError:
                    print(f"[WARNING] File not found: {file_name}")
                    continue

                df_raw["mbps"] = mbps
                df_raw["scenario"] = scenario

                # Compute best interface per Seq
                idx_min_latency = df_raw.groupby("Seq")["Uplink_Latency"].idxmin()

                best_interface_map = (
                    df_raw.loc[idx_min_latency, ["Seq", "Interface"]]
                    .set_index("Seq")["Interface"]
                )
                best_latency_map = (
                    df_raw.loc[idx_min_latency, ["Seq", "Uplink_Latency"]]
                    .set_index("Seq")["Uplink_Latency"]
                )

                df_raw["best_interface"] = df_raw["Seq"].map(best_interface_map)
                df_raw["best_latency"] = df_raw["Seq"].map(best_latency_map)

                if csv_model == "full":
                    df_full = pd.concat([df_full, df_raw], ignore_index=True)
                else:
                    df_radio = pd.concat([df_radio, df_raw], ignore_index=True)

    # Sort safely
    for df in [df_full, df_radio]:
        if not df.empty:
            try:
                df.sort_values("Packet_Sent", inplace=True)
            except:
                df.sort_values("Now_ms", inplace=True)

    # Fill missing packets
    if not df_full.empty:
        df_full = fill_missing_packets(df_full, interfaces)
        df_full["Timestamp"] = pd.to_datetime(df_full["Now_ms"], unit="ms")
        df_full = compute_best_latency(df_full)

    if not df_radio.empty:
        df_radio = fill_missing_packets(df_radio, interfaces)
        df_radio["Timestamp"] = pd.to_datetime(df_radio["Now_ms"], unit="ms")
        df_radio = compute_best_latency(df_radio)

    return df_full, df_radio, scenarios, mbps_list
