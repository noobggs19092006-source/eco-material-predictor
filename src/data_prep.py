import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")

POLYMER_FEATURES = [
    "repeat_unit_MW", "backbone_flexibility", "polarity_index", 
    "hydrogen_bond_capacity", "aromatic_content", "crystallinity_tendency", "eco_score"
]

ALLOY_FEATURES = [
    "atomic_radius_difference", "mixing_enthalpy", "valence_electrons", 
    "electronegativity_diff", "shear_modulus", "melting_temp", "eco_score"
]

FEATURES = POLYMER_FEATURES + [f for f in ALLOY_FEATURES if f not in POLYMER_FEATURES]

TARGETS = [
    "Tg_celsius", "tensile_strength_MPa", "youngs_modulus_GPa",
    "density_gcm3", "thermal_conductivity_WmK", "log10_elec_conductivity",
    "elongation_at_break_pct", "dielectric_constant",
    "water_absorption_pct", "oxygen_permeability_barrer"
]

def load_and_split():
    df = pd.read_csv(RAW_CSV)
    
    # Fill class-specific NaNs with 0 (Polymers have 0 for Metal features, etc.)
    for f in FEATURES:
        if f in df.columns:
            df[f] = df[f].fillna(0.0)

    # 1. Standardize numerical features
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join(ROOT, "models", "scaler.pkl"))
    print(f"[data_prep] Scaler saved → {os.path.join(ROOT, 'models', 'scaler.pkl')}")

    # 2. Split 70/10/20 entirely preserving class ratio natively
    train_val, test_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df, val_df = train_test_split(train_val, test_size=0.125, random_state=42) # 0.125 of 0.8 is 0.10

    print(f"[data_prep] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    out_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(out_dir, "features_train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "features_val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "features_test.csv"), index=False)
    
    print(f"[data_prep] Processed splits saved → {out_dir}")

if __name__ == "__main__":
    load_and_split()
