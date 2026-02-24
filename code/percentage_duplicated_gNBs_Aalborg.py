import pandas as pd
from pathlib import Path

# Read your CSV
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
out_name = DATA_DIR / f"aalborg_4G_5G_filtered.csv"

df = pd.read_csv(out_name)


# Clean/normalize operator names (handles e.g. "TDc" vs "TDC")
op = (
    df["Operator"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.upper()
)

# Count percentages
pct = (
    op.value_counts(normalize=True)
      .mul(100)
      .reindex(["TELENOR", "TDC", "TDC & TELENOR"], fill_value=0)
      .round(2)
)

print(pct)