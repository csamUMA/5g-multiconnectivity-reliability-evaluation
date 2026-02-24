# fd_simulation.py

import pandas as pd


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