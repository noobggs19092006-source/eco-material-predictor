"""
test_pipeline.py — Updated for 10 predicted properties.
"""
import os, sys, pytest, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RAW_CSV    = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")
PROC_DIR   = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

ALL_TARGETS = [
    "Tg_celsius", "tensile_strength_MPa", "youngs_modulus_GPa",
    "density_gcm3", "thermal_conductivity_WmK", "log10_elec_conductivity",
    "elongation_at_break_pct", "dielectric_constant",
    "water_absorption_pct", "oxygen_permeability_barrer",
]

PLA_FEATURES = dict(
    repeat_unit_MW=72.0, backbone_flexibility=0.40,
    polarity_index=2.0, hydrogen_bond_capacity=2.0,
    aromatic_content=0.0, crystallinity_tendency=0.35,
    eco_score=1.0, is_alloy=0.0,
    atomic_radius_difference=0.0, mixing_enthalpy=0.0,
    valence_electrons=0.0, electronegativity_diff=0.0,
    shear_modulus=0.0, melting_temp=0.0
)

# ─ 1. Data Loading ──────────────────────────────────────────────────────────
class TestDataLoading:
    def test_csv_exists(self):
        assert os.path.isfile(RAW_CSV)

    def test_all_target_columns_present(self):
        df = pd.read_csv(RAW_CSV)
        for col in ALL_TARGETS:
            assert col in df.columns, f"Missing target: {col}"

    def test_row_count(self):
        assert len(pd.read_csv(RAW_CSV)) >= 100

    def test_tg_range(self):
        df = pd.read_csv(RAW_CSV)
        assert df["Tg_celsius"].between(-250, 3500).all()

    def test_eco_score_range(self):
        df = pd.read_csv(RAW_CSV)
        assert df["eco_score"].between(0, 1).all()

    def test_density_positive(self):
        df = pd.read_csv(RAW_CSV)
        assert (df["density_gcm3"] > 0).all()

    def test_log10_conductivity_range(self):
        df = pd.read_csv(RAW_CSV)
        assert df["log10_elec_conductivity"].between(-20, 10).all()

# ─ 2. Preprocessing ─────────────────────────────────────────────────────────
class TestPreprocessing:
    def test_run_produces_splits(self):
        from data_prep import load_and_split
        load_and_split()
        assert os.path.isfile(os.path.join(PROC_DIR, "features_train.csv"))
        assert os.path.isfile(os.path.join(PROC_DIR, "features_val.csv"))
        assert os.path.isfile(os.path.join(PROC_DIR, "features_test.csv"))

    def test_no_nan(self):
        from data_prep import load_and_split
        load_and_split()
        assert not pd.read_csv(os.path.join(PROC_DIR, "features_train.csv")).isna().any().any()
        assert not pd.read_csv(os.path.join(PROC_DIR, "features_val.csv")).isna().any().any()
        assert not pd.read_csv(os.path.join(PROC_DIR, "features_test.csv")).isna().any().any()

    def test_all_targets_in_y(self):
        from data_prep import load_and_split, TARGETS
        load_and_split()
        df = pd.read_csv(os.path.join(PROC_DIR, "features_train.csv"))
        for t in TARGETS:
            assert t in df.columns



    def test_scaler_saved(self):
        from data_prep import load_and_split
        load_and_split()
        assert os.path.isfile(os.path.join(MODELS_DIR, "scaler.pkl"))

    def test_split_sizes(self):
        from data_prep import load_and_split
        load_and_split()
        train_len = len(pd.read_csv(os.path.join(PROC_DIR, "features_train.csv")))
        val_len = len(pd.read_csv(os.path.join(PROC_DIR, "features_val.csv")))
        te_len = len(pd.read_csv(os.path.join(PROC_DIR, "features_test.csv")))
        total = train_len + val_len + te_len
        assert total >= 285
        assert te_len > 50   # 20%
        assert val_len >= 10  # ~10%

# ─ 3. Model ─────────────────────────────────────────────────────────────────
class TestModel:
    @pytest.fixture(scope="class", autouse=True)
    def ensure_model(self):
        path = os.path.join(MODELS_DIR, "material_predictor.pkl")
        if not os.path.isfile(path):
            from train import train_all
            train_all()

    def test_model_file_exists(self):
        assert os.path.isfile(os.path.join(MODELS_DIR, "material_predictor.pkl"))

    def test_all_10_targets_in_model(self):
        import joblib
        b = joblib.load(os.path.join(MODELS_DIR, "material_predictor.pkl"))
        for cls_name in ["polymer", "metal"]:
            for t in ALL_TARGETS:
                assert t in b[cls_name], f"Target missing from {cls_name} model: {t}"

    def test_each_model_has_rf_xgb_meta(self):
        import joblib
        b = joblib.load(os.path.join(MODELS_DIR, "material_predictor.pkl"))
        for cls_name in ["polymer", "metal"]:
            for t, m in b[cls_name].items():
                assert "rf" in m and "xgb" in m and "meta" in m

# ─ 4. Predictions ───────────────────────────────────────────────────────────
class TestPredictions:
    def test_all_10_targets_returned(self):
        from predict import predict
        r = predict(PLA_FEATURES)
        for t in ALL_TARGETS:
            assert t in r["predictions"]

    def test_pla_tg_range(self):
        from predict import predict
        tg = predict(PLA_FEATURES)["predictions"]["Tg_celsius"]
        assert 20 <= tg <= 110, f"PLA Tg={tg:.1f} outside 20–110°C"

    def test_pla_is_insulator(self):
        from predict import predict
        ec = predict(PLA_FEATURES)["predictions"]["log10_elec_conductivity"]
        assert ec < -5, f"PLA should be an insulator, got log10={ec:.2f}"

    def test_pla_density_reasonable(self):
        from predict import predict
        d = predict(PLA_FEATURES)["predictions"]["density_gcm3"]
        assert 0.8 <= d <= 2.0, f"PLA density={d:.3f} unreasonable"

    def test_confidence_positive(self):
        from predict import predict
        r = predict(PLA_FEATURES)
        for t, ci in r["confidence"].items():
            assert ci >= 0

    def test_deterministic(self):
        from predict import predict
        r1 = predict(PLA_FEATURES)["predictions"]
        r2 = predict(PLA_FEATURES)["predictions"]
        for t in ALL_TARGETS:
            assert abs(r1[t] - r2[t]) < 1e-6

# ─ 5. PDB Generator ─────────────────────────────────────────────────────────
class TestPDB:
    def test_list_available(self):
        from generate_pdb import list_available
        keys = list_available()
        assert "PLA" in keys
        assert "CELLULOSE" in keys

    def test_get_pdb_pla(self):
        from generate_pdb import get_pdb
        pdb = get_pdb("PLA")
        assert "HETATM" in pdb        # small molecules use HETATM, not ATOM

    def test_invalid_key_raises(self):
        from generate_pdb import get_pdb
        with pytest.raises(KeyError):
            get_pdb("UNOBTANIUM")

    def test_guess_material(self):
        from generate_pdb import guess_material_key
        key = guess_material_key(PLA_FEATURES)
        assert key == "PLA"

# ─ 6. CLI ────────────────────────────────────────────────────────────────────
class TestCLI:
    def test_cli_imports(self):
        import importlib
        cli = importlib.import_module("cli")
        assert hasattr(cli, "main")
        assert hasattr(cli, "render_results")
        assert hasattr(cli, "offer_pdb")
