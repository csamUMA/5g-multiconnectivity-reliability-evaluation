# run_fd_analysis.py

import argparse
from data_loader import load_experiment_data
from MC_strategies import *
from route import show_route_info
from metrics import compute_statistics
from plot_figures import *
import pandas as pd
from functions import *

def main():

    scenarios = ["URBAN", "HYBRID", "RURAL"]
    mbps_list = [4, 2, 1, 0.5, 0.25]
    strategies = ["FD", "Baseline", "Switching", "PD", "LinkAggregation"]
    modes = ["rsrp_only", "tx_only", "latency_only", "rsrp_tx", "rsrp_latency", "tx_latency", "full"]

    modes = ["radio_combo", "full"]
    dft_ifaces = [1, 2]
    
    interfaces = ["5G_1","5G_2"]
    INTERFACES = [1,2]

    parser = argparse.ArgumentParser(description="Run Full Duplication analysis")

    parser.add_argument("--experiment", type=int, required=True,
                        help="1 for Experiment 1, 2 for Experiment 2")

    parser.add_argument("--scenario", type=str, choices=scenarios, default=None,
                        help="Optional: run only one scenario (e.g., URBAN, RURAL, HYBRID)")

    parser.add_argument("--mbps", type=float, choices=mbps_list, default=None,
                        help="Optional: run only one Mbps value (e.g., 4, 2, 1, 0.5, 0.25)")

    parser.add_argument("--strategy", type=str, choices=strategies, default=None,
                        help="Optional: run only one strategy value (e.g., FD, Baseline, Switching, PD, LinkAggregation)")
    
    parser.add_argument("--mode", type=str, choices=modes, default=None,
                        help="Optional: run only one mode value (e.g., rsrp_only, tx_only, latency_only, rsrp_tx, rsrp_latency, tx_latency, full)")

    parser.add_argument("--dft_iface", type=int, choices=dft_ifaces, default=None,
                        help="Optional: run only one default interface value (e.g., 1, 2)")

    parser.add_argument("--show_route", type=bool, default=False,
                        help="Optional: show route information of the scenario")

    parser.add_argument("--show_ultx_gain_half_datarate", type=bool, default=False,
                        help="Optional: show gain in UL Tx Power when using half datarate (e.g., for Link Aggregation)")

    parser.add_argument("--plot_stats", type=bool, default=False,
                        help="Optional: plot statistics for each scenario")

    args = parser.parse_args()

    # If no mbps provided → use all
    if args.mbps is None:
        mbps = mbps_list
    else:
        mbps = [args.mbps]

    if args.strategy == "LinkAggregation":
        if mbps not in mbps_list:
            raise ValueError(f"Mbps '{args.mbps}' not valid for experiment {args.experiment}")
        mbps = args.mbps/2.0
        mbps_list = [args.mbps/2.0]
        print("Note: For Link Aggregation strategy, Mbps target is halved to simulate the combined throughput of both interfaces.")

        df_full, df_radio, scenarios, mbps_list = load_experiment_data(
            experiment=args.experiment,
            scenario_filter=args.scenario,
            mbps_filter=mbps
        )
    elif args.show_ultx_gain_half_datarate:
        mbps_list = [4.0, 0.25] if args.experiment == 2 else [4.0, 1.0]

        df_full, df_radio, scenarios, mbps_list = load_experiment_data(
            experiment=args.experiment,
            scenario_filter=args.scenario,
            mbps_filter=mbps_list
        )
    else:
        
        df_full, df_radio, scenarios, mbps_list = load_experiment_data(
            experiment=args.experiment,
            scenario_filter=args.scenario,
            mbps_filter=mbps
        )

    # Filter scenario if requested
    if args.scenario is not None:
        if args.scenario not in scenarios:
            raise ValueError(f"Scenario '{args.scenario}' not valid for experiment {args.experiment}")
        scenarios = [args.scenario]

    if args.strategy is not None:
        valid_strategies = ["FD", "Baseline", "Switching", "PD", "LinkAggregation"]
        if args.strategy not in valid_strategies:
            raise ValueError(f"Strategy '{args.strategy}' not valid. Choose from {valid_strategies}")
        print(f"Running only strategy: {args.strategy}")
        strategies = [args.strategy]

    if args.mode is not None:
        valid_modes = ["rsrp_only", "tx_only", "latency_only", "radio_combo", "full"]
        if args.mode not in valid_modes:
            raise ValueError(f"Mode '{args.mode}' not valid. Choose from {valid_modes}")
        print(f"Running only mode: {args.mode}")
        modes = [args.mode]

    if args.dft_iface is not None:
        valid_ifaces = [1, 2]
        if args.dft_iface not in valid_ifaces:
            raise ValueError(f"Default interface '{args.dft_iface}' not valid. Choose from {valid_ifaces}")
        print(f"Running only default interface: {args.dft_iface}")
        dft_ifaces = [args.dft_iface]

    print(f"\nRunning experiment {args.experiment}")
    print(f"Scenarios: {scenarios}")
    print(f"Mbps list: {mbps_list}")
    print(f"Strategies: {strategies}")

    if args.show_route:
        for scenario in scenarios:
            for mbps in mbps_list:
                subset = pd.DataFrame()
                subset_1 = pd.DataFrame()
                subset_2 = pd.DataFrame()
                subset_all = pd.DataFrame()
                for interface in interfaces:    
                    subset_raw = df_full[
                        (df_full["scenario"] == scenario) &
                        (df_full["Interface"] == interface) &
                        (df_full["mbps"] == mbps)
                    ].copy()

                    if interface == "5G_1":
                        subset_1 = subset_raw.copy()
                        subset_ready = prepare_radio_dataframe(subset_1)
                    elif interface == "5G_2":
                        subset_2 = subset_raw.copy()
                        subset_ready = prepare_radio_dataframe(subset_2)
                    
                    subset_all = pd.concat([subset_all, subset_ready], ignore_index=True)
                    subset = subset_all.copy()

                SCENARIO = 1 if scenario == "RURAL" else (2 if scenario == "URBAN" else 3)

                show_route_info(subset, scenario=SCENARIO)
        
        return True

    if args.show_ultx_gain_half_datarate:
        for scenario in scenarios:
            subset = pd.DataFrame()
            subset_1 = pd.DataFrame()
            subset_2 = pd.DataFrame()
            subset_all = pd.DataFrame()        
            
            for mbps in mbps_list:    
                for interface in interfaces:    
                    subset_raw = df_full[
                        (df_full["scenario"] == scenario) &
                        (df_full["Interface"] == interface) &
                        (df_full["mbps"] == mbps)
                    ].copy()

                    if interface == "5G_1":
                        subset_1 = subset_raw.copy()
                        subset_ready = prepare_radio_dataframe(subset_1)
                    elif interface == "5G_2":
                        subset_2 = subset_raw.copy()
                        subset_ready = prepare_radio_dataframe(subset_2)
                    
                    subset_all = pd.concat([subset_all, subset_ready], ignore_index=True)
                    subset = subset_all.copy()

            SCENARIO = 1 if scenario == "RURAL" else (2 if scenario == "URBAN" else 3)

            plot_rsrp_ultx_half_datarate(subset, scenario=SCENARIO, mbps_list=mbps_list, interface=2)
        
        return True    

    for strategy in strategies:
        for scenario in scenarios:
            for mbps in mbps_list:
                subset_1 = pd.DataFrame()
                subset_2 = pd.DataFrame()
                subset_all = pd.DataFrame()
                for interface in interfaces:    
                    subset_raw = df_full[
                        (df_full["scenario"] == scenario) &
                        (df_full["Interface"] == interface) &
                        (df_full["mbps"] == mbps)
                    ].copy()

                    if interface == "5G_1":
                        subset_1 = subset_raw.copy()
                        subset_ready = prepare_radio_dataframe(subset_1)
                    elif interface == "5G_2":
                        subset_2 = subset_raw.copy()
                        subset_ready = prepare_radio_dataframe(subset_2)
                    
                    subset_all = pd.concat([subset_all, subset_ready], ignore_index=True)                
                    subset = subset_all.copy()

                if strategy == "FD":
                    subset_fd = simulate_full_duplication(subset)

                    stats = compute_statistics(subset_fd)
                    print("\n--------------------------------------------")
                    print(f"Scenario: {scenario}")
                    print(f"Mbps: {mbps}")
                    print("--------------------------------------------")

                    for k, v in stats.items():
                        print(f"{k}: {v:.4f}")

                    print("--------------------------------------------\n")

                elif strategy == "Switching":
                    for mode in modes:
                        for default_iface in dft_ifaces:
                            subset_switching = simulate_switching_strategy(subset, default_iface=default_iface, mode=mode)

                            stats = compute_statistics(subset_switching)
                            print("\n--------------------------------------------")
                            print(f"Scenario: {scenario}")
                            print(f"Mbps: {mbps}")
                            print(f"Mode: {mode}")
                            print(f"Default Interface: {default_iface}")
                            print("--------------------------------------------")

                            for k, v in stats.items():
                                print(f"{k}: {v:.4f}")

                            print("--------------------------------------------\n")

                elif strategy == "PD":
                    for mode in modes:
                        for default_iface in dft_ifaces:
                            subset_switching = simulate_partial_duplication_strategy(subset, default_iface=default_iface, mode=mode)

                            stats = compute_statistics(subset_switching)
                            print("\n--------------------------------------------")
                            print(f"Scenario: {scenario}")
                            print(f"Mbps: {mbps}")
                            print(f"Mode: {mode}")
                            print(f"Default Interface: {default_iface}")
                            print("--------------------------------------------")

                            for k, v in stats.items():
                                print(f"{k}: {v:.4f}")

                            print("--------------------------------------------\n")


                if strategy == "Baseline":
                    for interface in INTERFACES:
                        #subset_baseline = subset_all[subset_all["Interface"] == interface].copy()

                        subset_baseline = subset.copy()
                        
                        stats = compute_statistics(subset_baseline)
                        print("\n--------------------------------------------")
                        print(f"Scenario: {scenario}")
                        print(f"Mbps: {mbps}")
                        print(f"Interface: {interface}")
                        print("--------------------------------------------")

                        for k, v in stats.items():
                            print(f"{k}: {v:.4f}")

                        if args.plot_stats:
                            iface = "5G_1" if interface == 1 else "5G_2"
                            plot_rsrp_vs_latency_distribution(subset_baseline, mbps=mbps, scenario=scenario, interface=iface)
                            plot_tx_power_vs_latency_distribution(subset_baseline, mbps=mbps, scenario=scenario, interface=iface)

                if strategy == "LinkAggregation":
                    subset_link_aggr = subset.copy()

                    
                    
                    subset_link_aggr = simulate_link_aggregation(subset_link_aggr, mbps_target = mbps)
                    
                    stats = compute_statistics(subset_link_aggr)
                    print("\n--------------------------------------------")
                    print(f"Scenario: {scenario}")
                    print(f"Mbps: {mbps}")
                    print(f"Interface: {interface}")
                    print("--------------------------------------------")

                    for k, v in stats.items():
                        print(f"{k}: {v:.4f}")

                    if args.plot_stats:
                        iface = "5G_1" if interface == 1 else "5G_2"
                        plot_rsrp_vs_latency_distribution(subset_link_aggr, mbps=mbps, scenario=scenario, interface=iface)
                        plot_tx_power_vs_latency_distribution(subset_link_aggr, mbps=mbps, scenario=scenario, interface=iface)



if __name__ == "__main__":
    main()
