"""
data_prep.py — Data loading, cleaning, feature engineering, and splitting.
Produces processed train/val/test CSVs for the ML pipeline.
Split: 70% train / 10% validation / 20% test.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV     = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")
PROC_DIR    = os.path.join(ROOT, "data", "processed")
MODELS_DIR  = os.path.join(ROOT, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "repeat_unit_MW",
    "backbone_flexibility",
    "polarity_index",
    "hydrogen_bond_capacity",
    "aromatic_content",
    "crystallinity_tendency",
    "eco_score",
    "is_alloy",
    "mw_flexibility",
    "polar_hbond",
]

TARGET_COLS = [
    "Tg_celsius",
    "tensile_strength_MPa",
    "youngs_modulus_GPa",
    "density_gcm3",
    "thermal_conductivity_WmK",
    "log10_elec_conductivity",
    "elongation_at_break_pct",
    "dielectric_constant",
    "water_absorption_pct",
    "oxygen_permeability_barrer",
]

# Human-readable metadata for display and plots
TARGET_META = {
    "Tg_celsius":                 ("Glass Transition Temp",   "°C",      "🌡"),
    "tensile_strength_MPa":       ("Tensile Strength",        "MPa",     "🔩"),
    "youngs_modulus_GPa":         ("Young's Modulus",         "GPa",     "📐"),
    "density_gcm3":               ("Density",                 "g/cm³",   "⚖"),
    "thermal_conductivity_WmK":   ("Thermal Conductivity",    "W/m·K",   "🌡"),
    "log10_elec_conductivity":    ("Elec. Conductivity",      "log₁₀S/m","⚡"),
    "elongation_at_break_pct":    ("Elongation at Break",     "%",       "🔗"),
    "dielectric_constant":        ("Dielectric Constant",     "—",       "🔋"),
    "water_absorption_pct":       ("Water Absorption",        "%",       "💧"),
    "oxygen_permeability_barrer": ("O₂ Permeability",         "Barrers", "💨"),
}


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[data_prep] Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_alloy"]      = (df["material_class"] == "alloy").astype(int)
    df["mw_flexibility"] = df["repeat_unit_MW"] * df["backbone_flexibility"]
    df["polar_hbond"]   = df["polarity_index"] * df["hydrogen_bond_capacity"]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["material_name", "material_class"], errors="ignore")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def split_and_scale(df: pd.DataFrame, seed: int = 42):
    """
    Three-way split: 70% train / 10% validation / 20% test.
    Validation is used for hyperparameter tuning.
    Test is completely held out for final evaluation.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    # First split: 80% train+val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    # Second split: 70/10 from the 80% → train=87.5%, val=12.5% of trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=seed
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val), columns=FEATURE_COLS, index=X_val.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    joblib.dump(scaler, SCALER_PATH)
    print(f"[data_prep] Scaler saved → {SCALER_PATH}")
    print(f"[data_prep] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test


def run(save: bool = True):
    df_raw  = load_raw()
    df_feat = engineer_features(df_raw)
    df      = clean(df_feat)

    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(df)

    if save:
        X_train.join(y_train).to_csv(
            os.path.join(PROC_DIR, "features_train.csv"), index=False
        )
        X_val.join(y_val).to_csv(
            os.path.join(PROC_DIR, "features_val.csv"), index=False
        )
        X_test.join(y_test).to_csv(
            os.path.join(PROC_DIR, "features_test.csv"), index=False
        )
        print(f"[data_prep] Processed splits saved → {PROC_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    run()
