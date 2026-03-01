import numpy as np
import pandas as pd
from src.predict import _load_bundle, _load_scaler
from src.data_prep import FEATURES, TARGETS

def run_large_simulation():
    np.random.seed(42)
    N = 10000

    # 1. Polymers
    p_df = pd.DataFrame({
        "repeat_unit_MW": np.random.uniform(10, 600, N),
        "backbone_flexibility": np.random.uniform(0, 1, N),
        "polarity_index": np.random.uniform(0, 3, N),
        "hydrogen_bond_capacity": np.random.uniform(0, 5, N),
        "aromatic_content": np.random.uniform(0, 1, N),
        "crystallinity_tendency": np.random.uniform(0, 1, N),
        "eco_score": np.random.uniform(0, 1, N)
    })
    p_df["is_alloy"] = -1.0
    p_df["mw_flexibility"] = p_df["repeat_unit_MW"] * p_df["backbone_flexibility"]
    p_df["polar_hbond"] = p_df["polarity_index"] * p_df["hydrogen_bond_capacity"]
    for f in FEATURES:
        if f not in p_df.columns:
            p_df[f] = 0.0

    # 2. Alloys
    a_df = pd.DataFrame({
        "repeat_unit_MW": np.random.uniform(10, 300, N),
        "backbone_flexibility": np.random.uniform(0, 1, N),
        "polarity_index": np.random.uniform(0, 3, N),
        "aromatic_content": np.random.uniform(0, 1, N),
        "eco_score": np.random.uniform(0, 1, N)
    })
    hbond = np.random.uniform(0, 1, N) * 3.5
    a_df["hydrogen_bond_capacity"] = hbond
    a_df["crystallinity_tendency"] = hbond / 3.5
    a_df["is_alloy"] = -1.0
    a_df["mw_flexibility"] = a_df["repeat_unit_MW"] * a_df["backbone_flexibility"]
    a_df["polar_hbond"] = a_df["polarity_index"] * a_df["hydrogen_bond_capacity"]
    for f in FEATURES:
        if f not in a_df.columns:
            a_df[f] = 0.0

    # 3. Metals
    m_df = pd.DataFrame({
        "atomic_radius_difference": np.random.uniform(0, 15, N),
        "mixing_enthalpy": np.random.uniform(-50, 20, N),
        "valence_electrons": np.random.uniform(3, 12, N),
        "electronegativity_diff": np.random.uniform(0, 0.6, N),
        "shear_modulus": np.random.uniform(10, 150, N),
        "melting_temp": np.random.uniform(400, 3500, N),
        "eco_score": np.random.uniform(0, 1, N)
    })
    m_df["is_alloy"] = 1.0
    for f in FEATURES:
        if f not in m_df.columns:
            m_df[f] = 0.0
            
    # Load models
    bundle = _load_bundle()
    scaler = _load_scaler()
    features_ordered = scaler.feature_names_in_
    
    def predict_batch(df_raw, mode_bundle):
        X_sc = scaler.transform(df_raw[features_ordered])
        X_np = X_sc
        preds = {}
        for t in TARGETS:
            m = mode_bundle[t]
            rf_trees = np.array([e.predict(X_np) for e in m["rf"].estimators_])
            rf_mean = rf_trees.mean(axis=0)
            xgp = m["xgb"].predict(X_sc)
            meta_in = np.column_stack([rf_mean, xgp])
            preds[t] = m["meta"].predict(meta_in)
        return pd.DataFrame(preds)

    print("Predicting Polymers...")
    p_preds = predict_batch(p_df, bundle["polymer"])
    print("Predicting Alloys...")
    a_preds = predict_batch(a_df, bundle["polymer"])
    print("Predicting Metals...")
    m_preds = predict_batch(m_df, bundle["alloy"])

    print("\n--- GLOBAL MAXIMUMS (10,000 Samples per class) ---")
    metrics = ["tensile_strength_MPa", "Tg_celsius", "youngs_modulus_GPa", "density_gcm3", "thermal_conductivity_WmK", "elongation_at_break_pct"]
    
    print("\nPOLYMER MAXES:")
    for metric in metrics:
        val = p_preds[metric].max() * 1.05  # add 5% headroom
        print(f"  {metric}: {val:.2f}")

    print("\nALLOY MAXES:")
    for metric in metrics:
        val = a_preds[metric].max() * 1.05  # add 5% headroom
        print(f"  {metric}: {val:.2f}")

    print("\nMETAL MAXES:")
    for metric in metrics:
        val = m_preds[metric].max() * 1.05  # add 5% headroom
        print(f"  {metric}: {val:.2f}")

if __name__ == "__main__":
    run_large_simulation()
