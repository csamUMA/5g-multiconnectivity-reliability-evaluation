import pandas as pd
import folium
from pathlib import Path

# Load your processed base‑station positions
bts = pd.read_csv("/home/carlos/UMA/Aalborg/rtp/MultiConnecitivity_5G_zenodo/data/aalborg_4G_5G_filtered.csv")
bts_unique = bts[['lat', 'lon', 'Operator']].drop_duplicates()

use_ONLY_radio = True

def show_route_info(
    subset: pd.DataFrame,
    scenario: str
):

    subset_scenario = subset[(subset['scenario'] == scenario) & (subset['has_radio_parameters'] == 1)].copy()

    # Aalborg bounding box (recommended)
    subset_scenario = subset_scenario[
        (subset_scenario["lat"].between(55, 62)) &
        (subset_scenario["lon"].between(8, 12))
    ]

    # center on the radio data
    center_lat = subset_scenario["lat"].mean()
    center_lon = subset_scenario["lon"].mean()

    # base map with terrain tiles
    m = folium.Map(location=[center_lat, center_lon],
                zoom_start=13, tiles=None, control_scale=True)
    #folium.TileLayer(
    #    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    #    attr='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC‑BY‑SA)',
    #    name="OpenTopoMap (Terrain)"
    #).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="CartoDB Positron",
        control=False
    ).add_to(m)

    # draw the route polyline
    line_coords = list(zip(subset_scenario["lat"], subset_scenario["lon"]))

    # small pink dots for each GPS point
    for _, row in subset_scenario.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            color="#171717ff", #FF00FF
            fill=True,
            fill_color="#171717ff",
            fill_opacity=0.8,
            weight=0
        ).add_to(m)

    # add gNB markers: blue square = Telenor, red circle = TDC, yellow diamond = both
    for _, row in bts_unique.iterrows():
        lat, lon, op = row['lat'], row['lon'], row['Operator']
        if op == "Telenor":
            # blue square
            folium.RegularPolygonMarker(
                location=[lat, lon],
                number_of_sides=4,
                radius=6,
                rotation=0,       # square
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.9,
                popup="Telenor gNB"
            ).add_to(m)
        elif op == "TDC":
            # red circle
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#2ca02c",
                fill=True,
                fill_color="#2ca02c",
                fill_opacity=0.9,
                popup="TDC gNB"
            ).add_to(m)
        else:
            # yellow diamond for both operators
            folium.RegularPolygonMarker(
                location=[lat, lon],
                number_of_sides=3,
                radius=6,
                rotation=-30,      # triangle shape
                color="#ff7f0e",
                fill=True,
                fill_color="#ff7f0e",
                fill_opacity=0.9,
                popup="TDC & Telenor gNB"
            ).add_to(m)

    # add a simple legend (positioned in the bottom-left corner)
    legend_html = """
        <div style="
            position: fixed;
            bottom: 40px;
            left: 40px;
            z-index: 9999;
            background-color: white;
            border: 1px solid #999;
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 12px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        ">
        <b>Legend</b><br><br>

        <!-- Route -->
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
        <div style="
            width: 10px;
            height: 10px;
            background: #171717ff;
            border-radius: 50%;
            margin-right: 8px;">
        </div>
        Route (GPS points)
        </div>

        <!-- Telenor -->
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
        <div style="
            width: 10px;
            height: 10px;
            background: #1f77b4;
            margin-right: 8px;">
        </div>
        Telenor gNB
        </div>

        <!-- TDC -->
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
        <div style="
            width: 10px;
            height: 10px;
            background: #2ca02c;
            border-radius: 50%;
            margin-right: 8px;">
        </div>
        TDC gNB
        </div>

        <!-- TDC & Telenor (triangle) -->
        <div style="display: flex; align-items: center;">
        <div style="
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-bottom: 10px solid #ff7f0e;
            transform: rotate(180deg);
            margin-right: 8px;">
        </div>
        TDC & Telenor gNB
        </div>

        </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    scenario_name = "RURAL" if scenario == 1 else ("URBAN" if scenario == 2 else "HYBRID")

    # display and save the map
    html_path = f"/home/carlos/UMA/Aalborg/rtp/MultiConnecitivity_5G_zenodo/figures/route_maps/route_{scenario_name}_radio_only.html"
    m.save(html_path)
