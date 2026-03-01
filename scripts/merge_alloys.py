import pandas as pd
import numpy as np
import os

df_steel = pd.read_csv("data/raw/steel_strength.csv")
# steel cols: 'formula', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti', 'yield strength', 'tensile strength', 'elongation'
df_steel["source"] = "matminer"

df_jeff = pd.read_csv("data/archive (2)/tensile-properties-from-jeffs-ml-sheet-20220208.csv")
# jeff cols: 'Fe', 'C', 'Cr', ... 'Yield Stress, [MPa]', 'Ultimate Tensile Stress, [MPa]', 'Tensile elongation [%]'
rename_jeff = {
    "Yield Stress, [MPa]": "yield strength",
    "Ultimate Tensile Stress, [MPa]": "tensile strength",
    "Tensile elongation [%]": "elongation"
}
df_jeff = df_jeff.rename(columns=rename_jeff)
df_jeff.columns = [c.lower() for c in df_jeff.columns]
df_jeff["source"] = "jeff_ml"

df_alloys = pd.read_csv("data/archive (4)/Alloys.csv")
# alloys cols: 'Alloy', 'Tensile Strength: Ultimate (UTS) (psi)', 'Al', 'As', ...
df_alloys = df_alloys.rename(columns={"Tensile Strength: Ultimate (UTS) (psi)": "tensile strength"})
df_alloys["tensile strength"] = df_alloys["tensile strength"] * 0.00689476  # psi to MPa
df_alloys.columns = [c.lower() for c in df_alloys.columns]
df_alloys["source"] = "alloys_db"
import re

def parse_formula(formula):
    # Extracts element symbols and numbers from strings like "Al0.5Co1FeNi2"
    matches = re.findall(r'([A-Z][a-z]?)([\d\.]*)', str(formula).replace(' ', ''))
    comps = {}
    for el, amt in matches:
        amt = float(amt) if amt else 1.0
        comps[el.lower()] = comps.get(el.lower(), 0.0) + amt
    # Convert fractions to percentages for ML tracking
    total = sum(comps.values())
    if total > 0:
        for k in comps:
            comps[k] = (comps[k] / total) * 100
    return comps

df_alloy_yield = pd.read_csv("data/archive (1)/Alloy_Yield_Strength.csv")
# parse formula from 'Alloy'
parsed_comps1 = df_alloy_yield["Alloy"].apply(parse_formula).apply(pd.Series)
df_alloy_yield = pd.concat([df_alloy_yield, parsed_comps1], axis=1)
df_alloy_yield = df_alloy_yield.rename(columns={"YS (MPa)": "yield strength"})
df_alloy_yield.columns = [c.lower() for c in df_alloy_yield.columns]
df_alloy_yield["source"] = "archive1"

df_hea = pd.read_csv("data/archive (3)/High Entropy Alloy Properties.csv")
parsed_comps3 = df_hea["FORMULA"].apply(parse_formula).apply(pd.Series)
df_hea = pd.concat([df_hea, parsed_comps3], axis=1)
df_hea = df_hea.rename(columns={
    "PROPERTY: YS (MPa)": "yield strength",
    "PROPERTY: UTS (MPa)": "tensile strength", 
    "PROPERTY: Elongation (%)": "elongation"
})
df_hea.columns = [c.lower() for c in df_hea.columns]
df_hea["source"] = "archive3"

dfs = [df_steel, df_jeff, df_alloys, df_alloy_yield, df_hea]
combined = pd.concat(dfs, ignore_index=True)

# Find all element columns (1 or 2 letter lower case strings, or explicitly known)
valid_elements = set()
for c in combined.columns:
    if len(c) <= 2 and c.isalpha():
        valid_elements.add(c)
    elif c in ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']:
        valid_elements.add(c)

targets = ["yield strength", "tensile strength", "elongation"]
keep_cols = list(valid_elements) + targets + ["source"]

combined = combined[[c for c in keep_cols if c in combined.columns]]

# Fill missing elements with 0.0
for el in valid_elements:
    if el in combined.columns:
        combined[el] = combined[el].fillna(0.0)

# Drop rows where ALL targets are missing
combined = combined.dropna(subset=targets, how="all")

# Print stats
print(f"Combined Shape: {combined.shape}")
print(f"Total Elements Tracked: {len(valid_elements)}")
print("Null values in targets:")
print(combined[targets].isna().sum())

combined.to_csv("data/raw/combined_alloys.csv", index=False)
print("Saved to data/raw/combined_alloys.csv")
