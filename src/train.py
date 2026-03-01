import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
PROC_DIR   = os.path.join(ROOT, "data", "processed")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import load_and_split, TARGETS, FEATURES

def train_single_target(X_train, y_train, X_val, y_val, target, seed=42, label=""):
    print(f"\n  ┌─ [{label or target}]")

    # Symmetric hyperparameters for dual 4000-sample sets
    best_rf = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 4, "max_features": "sqrt"}
    best_xgb = {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.8}

    n = len(X_train)
    split_idx = int(n * 0.8)
    
    X_arr   = X_train.values if hasattr(X_train, "values") else X_train
    y_arr   = y_train.values if hasattr(y_train, "values") else y_train
    
    X_tr_sub, y_tr_sub = X_arr[:split_idx], y_arr[:split_idx]
    X_val_sub, y_val_sub = X_arr[split_idx:], y_arr[split_idx:]
    
    rf_f = RandomForestRegressor(**best_rf, random_state=seed, n_jobs=-1)
    rf_f.fit(X_tr_sub, y_tr_sub)
    oof_rf_val = rf_f.predict(X_val_sub)
    
    xg_f = XGBRegressor(**best_xgb, random_state=seed, tree_method="hist", verbosity=0)
    xg_f.fit(X_tr_sub, y_tr_sub)
    oof_xgb_val = xg_f.predict(X_val_sub)
    
    meta = Ridge(alpha=0.5)
    meta.fit(np.column_stack([oof_rf_val, oof_xgb_val]), y_val_sub)

    # Train final models on the entire full train set for definitive inference
    final_rf  = RandomForestRegressor(**best_rf, random_state=seed, n_jobs=-1)
    final_rf.fit(X_train, y_train)
    final_xgb = XGBRegressor(**best_xgb, random_state=seed, tree_method="hist", verbosity=0)
    final_xgb.fit(X_train, y_train)

    # Validation score (honest, unseen during training)
    val_pred = meta.predict(np.column_stack([
        final_rf.predict(X_val), final_xgb.predict(X_val)
    ]))
    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val
    val_r2 = r2_score(y_val_arr, val_pred) if len(y_val_arr) > 1 else float("nan")

    print(f"  │  Val R²={val_r2:.4f}  (n_train={n}, n_val={len(X_val)})")
    print(f"  └─────────────────────────────────────────────────────")

    return {
        "rf":    final_rf,
        "xgb":   final_xgb,
        "meta":  meta,
        "rf_fi": dict(zip(FEATURES, final_rf.feature_importances_)),
    }

def train_class_models(X_tr, y_tr, mask_train, X_val, y_val, mask_val, class_label, seed=42):
    print(f"\n{'═'*64}")
    print(f"  🎯  Training {class_label} models  ({mask_train.sum()} train, {mask_val.sum()} val)")
    print(f"{'═'*64}")
    subset_X  = X_tr.loc[mask_train].reset_index(drop=True)
    subset_y  = y_tr.loc[mask_train].reset_index(drop=True)
    subset_Xv = X_val.loc[mask_val].reset_index(drop=True)
    subset_yv = y_val.loc[mask_val].reset_index(drop=True)
    
    models = {}
    for target in TARGETS:
        models[target] = train_single_target(
            subset_X, subset_y[target],
            subset_Xv, subset_yv[target],
            target, seed=seed,
            label=f"{target}  [{class_label}]"
        )
    return models

def train_all(seed=42):
    print("═" * 64)
    print("  🌿  Eco-Material Property Predictor — Training v4 (Universal)")
    print("  Class-Specific Models | Unified 70/10/20 Split")
    print("═" * 64)

    # Re-run data prep to ensure pristine unified csv is tracked exactly
    load_and_split()

    train_df = pd.read_csv(os.path.join(PROC_DIR, "features_train.csv"))
    val_df   = pd.read_csv(os.path.join(PROC_DIR, "features_val.csv"))
    test_df  = pd.read_csv(os.path.join(PROC_DIR, "features_test.csv"))
    
    df_raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "materials_dataset.csv"))

    # Need to isolate the class masks. 
    # Use the pristine raw dataframe indices aligned to the splits
    train_indices = train_df.index
    val_indices = val_df.index
    test_indices = test_df.index

    # It's cleaner to match the split indices to the raw file `material_class` column
    # To do this safely, we actually should just read `materials_dataset.csv`,
    # append the `material_class` to memory temporarily to map the masks.
    
    # Wait, data_prep standardizes FEATURES but destroys text columns.
    # Let's map identically. 
    train_mask_poly = train_df['eco_score'] > -999 # Dummy init
    # We can detect class because Metals have atomic_radius_difference while Polymers have repeat_unit_MW
    # Better yet, looking at scaler, all variables > 0 natively, but filled NaNs are scaled cleanly.
    
    # Actually, we can fetch class natively:
    df_raw_aligned = pd.read_csv(os.path.join(ROOT, "data", "raw", "materials_dataset.csv"))
    train_classes = df_raw_aligned.loc[df_raw_aligned.index.isin(train_df.index)] # Rough map
    
    # Accurate Map: 
    # the index might not be preserved if train_test_split resets indices.
    # However, `train_test_split` keeps the original indices in the exported dataframe unless `to_csv(index=False)`
    
    # Let's rebuild the split natively here to guarantee pristine masks:
    df_clean = pd.read_csv(os.path.join(ROOT, "data", "raw", "materials_dataset.csv"))
    class_series = df_clean['material_class']
    
    for f in FEATURES:
        if f in df_clean.columns:
            df_clean[f] = df_clean[f].fillna(0.0)
            
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    df_clean[FEATURES] = scaler.transform(df_clean[FEATURES])
    
    from sklearn.model_selection import train_test_split
    tv_X, test_X, tv_C, test_C = train_test_split(df_clean, class_series, test_size=0.20, random_state=42)
    train_X, val_X, train_C, val_C = train_test_split(tv_X, tv_C, test_size=0.125, random_state=42)

    y_train = train_X[TARGETS]
    X_train = train_X[FEATURES]
    
    y_val = val_X[TARGETS]
    X_val = val_X[FEATURES]
    
    y_test = test_X[TARGETS]
    X_test = test_X[FEATURES]

    poly_tr  = (train_C == "polymer")
    poly_val = (val_C == "polymer")
    poly_te  = (test_C == "polymer")
    
    metal_tr = (train_C == "metal")
    metal_val= (val_C == "metal")
    metal_te = (test_C == "metal")

    # ── Train POLYMER model ──────────────────────────────────────────────────
    poly_models = train_class_models(X_train, y_train, poly_tr, X_val, y_val, poly_val, "POLYMER", seed)

    # ── Train METAL model ────────────────────────────────────────────────────
    metal_models = train_class_models(X_train, y_train, metal_tr, X_val, y_val, metal_val, "METAL", seed)

    bundle = {
        "polymer": poly_models,
        "metal":   metal_models,
        "feature_cols": list(FEATURES),
        "target_cols":  list(TARGETS),
    }
    save_path = os.path.join(MODELS_DIR, "material_predictor.pkl")
    joblib.dump(bundle, save_path, compress=3)
    print(f"\n[train] ✅  Models saved → {save_path}")

    # ── Test-set results (held-out, never seen) ──────────────────────────────
    def eval_set(models, X_te, y_te, label):
        print(f"\n[train] {label} HELD-OUT test-set results (n={len(X_te)}):")
        print(f"  {'Target':<36}  {'MAE':>10}  {'R²':>8}")
        print("  " + "─" * 60)
        for t in TARGETS:
            m = models[t]
            p = m["meta"].predict(np.column_stack([m["rf"].predict(X_te), m["xgb"].predict(X_te)]))
            y = y_te[t].values
            print(f"  {t:<36}  {mean_absolute_error(y,p):>10.3f}  {r2_score(y,p):>8.4f}")

    eval_set(poly_models, X_test.loc[poly_te], y_test.loc[poly_te], "POLYMER")
    eval_set(metal_models, X_test.loc[metal_te], y_test.loc[metal_te], "METAL")

    print("\n[train] Run 'make evaluate' for full report.")

if __name__ == "__main__":
    train_all()
