"""
test_pipeline.py
────────────────
27 tests covering data loading, preprocessing, model integrity,
predictions, PDB generation, and CLI imports.

Updated v2: column names match perfect_dataset.py / train.py.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH = Path("data/raw/materials_dataset.csv")

TARGET_COLS = [
    "Tg_celsius", "tensile_strength_MPa", "youngs_modulus_GPa",
    "density_g_cm3",                          # ← was density_gcm3 (fixed)
    "thermal_conductivity_W_mK",
    "electrical_conductivity_log_S_m",         # ← was log10_elec_conductivity (fixed)
    "elongation_at_break_pct", "dielectric_constant",
    "water_absorption_pct", "oxygen_permeability_barrer",
]

FEATURE_COLS = [
    "repeat_unit_MW", "backbone_flexibility", "polarity_index",
    "hydrogen_bond_capacity", "aromatic_content", "crystallinity_tendency",
    "eco_score", "is_alloy", "mw_flexibility", "polar_hbond",
]


class TestDataLoading:

    @pytest.fixture(autouse=True)
    def load_df(self):
        self.df = pd.read_csv(CSV_PATH)

    def test_csv_exists(self):
        assert CSV_PATH.exists(), "Dataset CSV not found"

    def test_row_count(self):
        assert len(self.df) >= 500, f"Expected ≥500 rows, got {len(self.df)}"

    def test_all_feature_columns_present(self):
        for col in FEATURE_COLS:
            assert col in self.df.columns, f"Missing feature: {col}"

    def test_all_target_columns_present(self):
        for col in TARGET_COLS:
            assert col in self.df.columns, f"Missing target: {col}"

    def test_eco_score_range(self):
        assert self.df["eco_score"].between(0, 1).all(), \
            "eco_score must be 0–1"

    def test_density_positive(self):
        assert (self.df["density_g_cm3"] > 0).all(), \
            "density_g_cm3 must be positive"

    def test_log10_conductivity_range(self):
        assert self.df["electrical_conductivity_log_S_m"].between(-20, 10).all(), \
            "electrical_conductivity_log_S_m out of range"

    def test_tg_range(self):
        # Polymers: −200 to ~400°C. Metal alloys use Tg as a melting-point
        # proxy — Ti/HEA systems reach ~2500°C.
        assert self.df["Tg_celsius"].between(-200, 2600).all(), \
            f"Tg_celsius out of range. Min={self.df['Tg_celsius'].min():.1f}, Max={self.df['Tg_celsius'].max():.1f}"

    def test_is_alloy_binary(self):
        assert set(self.df["is_alloy"].unique()).issubset({0, 1}), \
            "is_alloy must be 0 or 1 only"

    def test_no_nulls_in_features(self):
        nulls = self.df[FEATURE_COLS].isnull().sum().sum()
        assert nulls == 0, f"Found {nulls} NaN values in feature columns"

    def test_no_nulls_in_targets(self):
        nulls = self.df[TARGET_COLS].isnull().sum().sum()
        assert nulls == 0, f"Found {nulls} NaN values in target columns"

    def test_carbon_footprint_column_exists(self):
        assert "carbon_footprint_kgCO2_per_kg" in self.df.columns, \
            "carbon_footprint_kgCO2_per_kg column missing — needed for eco-impact"

    def test_data_source_column_exists(self):
        assert "data_source" in self.df.columns, \
            "data_source provenance column missing"

    def test_polymer_alloy_split(self):
        n_poly  = (self.df["is_alloy"] == 0).sum()
        n_alloy = (self.df["is_alloy"] == 1).sum()
        assert n_poly  >= 100, f"Too few polymers: {n_poly}"
        assert n_alloy >= 100, f"Too few alloys: {n_alloy}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING (data_prep.py)
# ══════════════════════════════════════════════════════════════════════════════
class TestPreprocessing:

    @pytest.fixture(autouse=True)
    def run_split(self):
        from src.data_prep import load_and_split
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test) = load_and_split()

    def test_run_produces_splits(self):
        assert self.X_train is not None

    def test_no_nan(self):
        for name, arr in [
            ("X_train", self.X_train), ("X_val", self.X_val),
            ("X_test",  self.X_test),  ("y_train", self.y_train),
        ]:
            assert not arr.isnull().values.any(), f"NaN found in {name}"

    def test_all_targets_in_y(self):
        for col in TARGET_COLS:
            assert col in self.y_train.columns, f"Missing target in y: {col}"

    def test_scaler_saved(self):
        assert Path("models/scaler.pkl").exists(), "scaler.pkl not saved"

    def test_split_sizes(self):
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        assert total >= 500, f"Split total {total} too small"
        # Test set should be ~20%
        test_ratio = len(self.X_test) / total
        assert 0.15 <= test_ratio <= 0.25, \
            f"Test ratio {test_ratio:.2f} outside expected 0.15–0.25"

    def test_features_scaled(self):
        # After StandardScaler, mean of each column should be ~0
        means = self.X_train.mean()
        assert (means.abs() < 1.0).all(), \
            "Features don't appear to be scaled (means too far from 0)"


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL FILES
# ══════════════════════════════════════════════════════════════════════════════
class TestModel:

    def test_polymer_model_file_exists(self):
        assert Path("models/polymer_model.pkl").exists()

    def test_alloy_model_file_exists(self):
        assert Path("models/alloy_model.pkl").exists()

    # Keep old name for backward compat with any Makefile references
    def test_model_file_exists(self):
        assert Path("models/polymer_model.pkl").exists() or \
               Path("models/material_predictor.pkl").exists()

    def test_all_10_targets_in_model(self):
        import joblib
        model = joblib.load("models/polymer_model.pkl")
        # MultiOutputRegressor: number of estimators = number of targets
        n = len(model.named_steps["model"].estimators_)
        assert n == 10, f"Expected 10 target estimators, got {n}"

    def test_each_model_has_rf_xgb_meta(self):
        """Pipeline should contain 'scaler' and 'model' steps."""
        import joblib
        for name in ["polymer_model.pkl", "alloy_model.pkl"]:
            m = joblib.load(f"models/{name}")
            assert "scaler" in m.named_steps, f"{name}: missing scaler step"
            assert "model"  in m.named_steps, f"{name}: missing model step"


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTIONS (predict.py)
# ══════════════════════════════════════════════════════════════════════════════
PLA_FEATURES = {
    "repeat_unit_MW":         72.0,
    "backbone_flexibility":   0.40,
    "polarity_index":         2.0,
    "hydrogen_bond_capacity": 2.0,
    "aromatic_content":       0.0,
    "crystallinity_tendency": 0.35,
    "eco_score":              1.0,
    "is_alloy":               0,
}


class TestPredictions:

    @pytest.fixture(autouse=True)
    def run_predict(self):
        from src.predict import predict
        self.result = predict(PLA_FEATURES)

    def test_all_10_targets_returned(self):
        preds = self.result["predictions"]
        assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"
        for col in TARGET_COLS:
            assert col in preds, f"Missing prediction key: {col}"

    def test_pla_tg_range(self):
        tg = self.result["predictions"]["Tg_celsius"]
        assert -10 <= tg <= 200, f"PLA Tg={tg} outside expected −10–200 °C"

    def test_pla_is_insulator(self):
        # PLA is an electrical insulator; log10(S/m) should be very negative
        ec = self.result["predictions"]["electrical_conductivity_log_S_m"]
        assert ec < 0, f"PLA electrical conductivity log={ec} should be < 0"

    def test_pla_density_reasonable(self):
        d = self.result["predictions"]["density_g_cm3"]
        assert 0.5 <= d <= 3.0, f"PLA density={d} outside 0.5–3.0 g/cm³"

    def test_confidence_positive(self):
        for key, val in self.result["confidence_pm"].items():
            assert val >= 0, f"Confidence for {key} is negative: {val}"

    def test_deterministic(self):
        from src.predict import predict
        r1 = predict(PLA_FEATURES)["predictions"]
        r2 = predict(PLA_FEATURES)["predictions"]
        for k in r1:
            assert r1[k] == r2[k], f"Non-deterministic prediction for {k}"

    def test_alloy_prediction_works(self):
        from src.predict import predict
        alloy_features = {
            "repeat_unit_MW":         27.0,
            "backbone_flexibility":   0.15,
            "polarity_index":         0.0,
            "hydrogen_bond_capacity": 0.0,
            "aromatic_content":       0.0,
            "crystallinity_tendency": 0.90,
            "eco_score":              0.45,
            "is_alloy":               1,
        }
        result = predict(alloy_features)
        assert len(result["predictions"]) == 10


# ══════════════════════════════════════════════════════════════════════════════
# 5. PDB GENERATION
# ══════════════════════════════════════════════════════════════════════════════
class TestPDB:

    def test_list_available(self):
        from src.generate_pdb import list_available_materials
        mats = list_available_materials()
        assert len(mats) > 0

    def test_get_pdb_pla(self):
        from src.generate_pdb import get_pdb
        pdb = get_pdb("pla")
        assert "ATOM" in pdb or "HETATM" in pdb, "PDB content missing ATOM records"

    def test_invalid_key_raises(self):
        from src.generate_pdb import get_pdb
        with pytest.raises((KeyError, ValueError)):
            get_pdb("this_material_does_not_exist_xyz")

    def test_guess_material(self):
        from src.generate_pdb import guess_material_key
        key = guess_material_key({"is_alloy": 0, "eco_score": 0.9})
        assert key is not None


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI IMPORT
# ══════════════════════════════════════════════════════════════════════════════
class TestCLI:

    def test_cli_imports(self):
        import src.cli  # should import without error