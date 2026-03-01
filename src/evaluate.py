import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR  = os.path.join(ROOT, "models")
PROC_DIR    = os.path.join(ROOT, "data", "processed")
RAW_CSV     = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import TARGETS

TARGET_META = {
    "Tg_celsius": ("Glass Transition / Melting Temp", "°C", "🌡️"),
    "tensile_strength_MPa": ("Tensile Strength", "MPa", "🏗️"),
    "youngs_modulus_GPa": ("Young's Modulus", "GPa", "📏"),
    "density_gcm3": ("Density", "g/cm³", "⚖️"),
    "thermal_conductivity_WmK": ("Thermal Conductivity", "W/m·K", "🔥"),
    "log10_elec_conductivity": ("Elec. Conductivity", "log₁₀ S/m", "⚡"),
    "elongation_at_break_pct": ("Elongation at Break", "%", "〰️"),
    "dielectric_constant": ("Dielectric Constant", "—", "🔋"),
    "water_absorption_pct": ("Water Absorption", "%", "💧"),
    "oxygen_permeability_barrer": ("O₂ Permeability", "Barrers", "🌬️"),
}

sns.set_theme(style="darkgrid", font_scale=1.0)
PALETTE = sns.color_palette("tab10", n_colors=10)


def load_artifacts():
    bundle   = joblib.load(os.path.join(MODELS_DIR, "material_predictor.pkl"))
    test_df  = pd.read_csv(os.path.join(PROC_DIR, "features_test.csv"))
    
    # We must remap test_df back to classes via the raw dataset indices since
    # data_prep.py standardizes the dataframe but doesn't store 'is_alloy' column identically.
    # We can detect class because Metals have e.g. 'atomic_radius_difference' while Polymers don't natively.
    # Actually wait: features_test.csv has the full unified 14 feature array. Metals have `shear_modulus` > 0
    # Let's rely on raw dataset index since indexing was perfectly aligned in train.py
    
    df_raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "materials_dataset.csv"))
    
    # Actually, simpler: data_prep.py saved `is_alloy` column inside `test_df` maybe?
    # No, it just scales `FEATURES`.
    # BUT, poly inherently has `shear_modulus` == 0 in scaled? No, scaling shifts 0 to a negative constant.
    
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    # The clean way is just re-split using the identical seed:
    df_clean = pd.read_csv(RAW_CSV)
    class_series = df_clean['material_class']
    
    features = bundle["feature_cols"]
    for f in features:
        if f in df_clean.columns: df_clean[f] = df_clean[f].fillna(0.0)
    
    X_raw = df_clean[features]
    from sklearn.model_selection import train_test_split
    tv_X, test_X, tv_C, test_C = train_test_split(X_raw, class_series, test_size=0.20, random_state=42)
    _, _, _, _ = train_test_split(tv_X, tv_C, test_size=0.125, random_state=42)
    
    X_te_scaled = pd.DataFrame(scaler.transform(test_X), columns=features, index=test_X.index)
    
    y_test = df_clean.loc[test_X.index, bundle["target_cols"]]
    
    return bundle, X_te_scaled, y_test, test_C


def print_report(results, name_prefix, title_suffix):
    lines = [
        "═" * 80,
        f"  🌿  Eco-Material Property Predictor — Evaluation Report ({title_suffix})",
        "═" * 80,
        f"  Test samples : {len(next(iter(results.values()))['y'])}",
        "",
        f"  {'Property':<32}  {'Unit':<10}  {'MAE':>10}  {'RMSE':>10}  {'R²':>8}",
        "  " + "─" * 76,
    ]
    for target, res in results.items():
        name, unit, icon = TARGET_META[target]
        lines.append(
            f"  {icon} {name:<30}  {unit:<10}  "
            f"{res['mae']:>10.3f}  {res['rmse']:>10.3f}  {res['r2']:>8.4f}"
        )
    lines += ["", "═" * 72]
    text = "\n".join(lines)
    print(text)
    path = os.path.join(RESULTS_DIR, f"evaluation_report_{name_prefix}.txt")
    with open(path, "w") as f: f.write(text)


def run():
    print("[evaluate] Loading model & re-aligning test data...")
    bundle, X_test, y_test, test_C = load_artifacts()

    for class_label in ["polymer", "metal"]:
        mask = (test_C == class_label)
        X_sub = X_test.loc[mask]
        y_sub = y_test.loc[mask]
        
        if len(X_sub) == 0: continue
            
        print(f"\n[evaluate] --- {class_label.upper()} EVALUATION ({len(X_sub)} samples) ---")
        
        results = {}
        for target in TARGETS:
            m = bundle[class_label][target]
            base = np.column_stack([m["rf"].predict(X_sub), m["xgb"].predict(X_sub)])
            preds = m["meta"].predict(base)
            
            if "density" in target or "water" in target or "oxygen" in target:
                preds = np.maximum(0.001, preds)
                
            y = y_sub[target].values
            results[target] = {
                "preds": preds, "y": y,
                "mae": mean_absolute_error(y, preds),
                "rmse": np.sqrt(mean_squared_error(y, preds)),
                "r2": r2_score(y, preds)
            }
            
        print_report(results, class_label, class_label.title() + "s")

    print("\n[evaluate] ✅  Evaluation reports saved to results/")

if __name__ == "__main__":
    run()
