# paaf_switching_simulation.py
import numpy as np
import pandas as pd


def simulate_link_aggregation(subset):
    
    # Orden temporal/por secuencia
    subset_link_aggreg = subset.sort_values("Seq").copy()

    # Hacemos link aggregation simple
    subset_link_aggreg.loc[subset['Interface'] == 1, 'Seq'] *= 2
    subset_link_aggreg.loc[subset['Interface'] == 2, 'Seq'] += 1

    # Último Seq total (usamos max normal, no np.max)
    last_seq = subset["Seq"].max()

    return subset_link_aggreg