import pandas as pd

files = [
    "data/archive/iron_alloys.csv",
    "data/archive (1)/Alloy_Yield_Strength.csv",
    "data/archive (2)/tensile-properties-from-jeffs-ml-sheet-20220208.csv",
    "data/archive (3)/High Entropy Alloy Properties.csv",
    "data/archive (4)/Alloys.csv",
    "data/raw/steel_strength.csv"
]

for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\n--- {f} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
