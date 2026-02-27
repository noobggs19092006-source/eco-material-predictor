"""
train.py — Trains TWO specialized ensemble models:
  • polymer_predictor  — trained only on polymer rows
  • alloy_predictor    — trained only on alloy rows
Both stored in material_predictor.pkl under keys 'polymer' and 'alloy'.

Split: 70% train / 10% validation / 20% test
  - Train: model fitting
  - Validation: hyperparameter selection (RandomizedSearchCV)  
  - Test: final held-out evaluation (reported in evaluate.py)

Architecture per target: RandomForest + XGBoost → Ridge meta-learner.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
PROC_DIR   = os.path.join(ROOT, "data", "processed")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import run as prepare_data, TARGET_COLS, FEATURE_COLS

RF_PARAM_GRID = {
    "n_estimators":      [150, 250, 400],
    "max_depth":         [None, 8, 15, 20],
    "min_samples_split": [2, 4],
    "min_samples_leaf":  [1, 2],
    "max_features":      ["sqrt", "log2", 0.6],
}
XGB_PARAM_GRID = {
    "n_estimators":     [150, 250, 350],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.03, 0.05, 0.10],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "gamma":            [0, 0.1],
}


def train_single_target(X_train, y_train, X_val, y_val, target, seed=42, label=""):
    print(f"\n  ┌─ [{label or target}]")

    n_cv = min(5, max(3, len(X_train) // 6))

    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=1),
        RF_PARAM_GRID, n_iter=25, cv=n_cv, scoring="r2",
        random_state=seed, n_jobs=1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_params_

    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=seed, tree_method="hist", verbosity=0),
        XGB_PARAM_GRID, n_iter=25, cv=n_cv, scoring="r2",
        random_state=seed, n_jobs=1, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_params_

    # OOF stacking on training data
    kf      = KFold(n_splits=n_cv, shuffle=True, random_state=seed)
    n       = len(X_train)
    oof_rf  = np.zeros(n)
    oof_xgb = np.zeros(n)
    X_arr   = X_train.values if hasattr(X_train, "values") else X_train
    y_arr   = y_train.values if hasattr(y_train, "values") else y_train

    for tr_idx, val_idx in kf.split(X_arr):
        rf_f = RandomForestRegressor(**best_rf, random_state=seed, n_jobs=1)
        rf_f.fit(X_arr[tr_idx], y_arr[tr_idx])
        oof_rf[val_idx] = rf_f.predict(X_arr[val_idx])

        xg_f = XGBRegressor(**best_xgb, random_state=seed,
                             tree_method="hist", verbosity=0)
        xg_f.fit(X_arr[tr_idx], y_arr[tr_idx])
        oof_xgb[val_idx] = xg_f.predict(X_arr[val_idx])

    # Final models on full training data
    final_rf  = RandomForestRegressor(**best_rf, random_state=seed, n_jobs=1)
    final_rf.fit(X_train, y_train)
    final_xgb = XGBRegressor(**best_xgb, random_state=seed,
                              tree_method="hist", verbosity=0)
    final_xgb.fit(X_train, y_train)

    meta = Ridge(alpha=0.5)
    meta.fit(np.column_stack([oof_rf, oof_xgb]), y_arr)

    # Report metrics on both train (OOF) and validation (honest)
    oof_r2 = r2_score(y_arr, meta.predict(np.column_stack([oof_rf, oof_xgb])))

    # Validation score (honest, unseen during training)
    val_pred = meta.predict(np.column_stack([
        final_rf.predict(X_val), final_xgb.predict(X_val)
    ]))
    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val
    val_r2 = r2_score(y_val_arr, val_pred) if len(y_val_arr) > 1 else float("nan")

    print(f"  │  OOF R²={oof_r2:.4f}  |  Val R²={val_r2:.4f}  (n_train={n}, n_val={len(X_val)})")
    print(f"  └─────────────────────────────────────────────────────")

    return {
        "rf":    final_rf,
        "xgb":   final_xgb,
        "meta":  meta,
        "rf_fi": dict(zip(FEATURE_COLS, final_rf.feature_importances_)),
    }


def train_class_models(X_tr, y_tr, mask_train, X_val, y_val, mask_val, class_label, seed=42):
    """Train all 10 targets for one material class."""
    print(f"\n{'═'*64}")
    print(f"  🎯  Training {class_label} models  ({mask_train.sum()} train, {mask_val.sum()} val)")
    print(f"{'═'*64}")
    subset_X  = X_tr[mask_train].reset_index(drop=True)
    subset_y  = y_tr[mask_train].reset_index(drop=True)
    subset_Xv = X_val[mask_val].reset_index(drop=True)
    subset_yv = y_val[mask_val].reset_index(drop=True)
    models = {}
    for target in TARGET_COLS:
        models[target] = train_single_target(
            subset_X, subset_y[target],
            subset_Xv, subset_yv[target],
            target, seed=seed,
            label=f"{target}  [{class_label}]"
        )
    return models


def train_all(seed=42):
    print("═" * 64)
    print("  🌿  Eco-Material Property Predictor — Training v3")
    print("  Class-Specific Models | 70/10/20 Split")
    print("═" * 64)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(save=True)

    # Reset indices for clean boolean masking
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # Load processed files for unscaled is_alloy masks
    train_df = pd.read_csv(os.path.join(PROC_DIR, "features_train.csv"))
    val_df   = pd.read_csv(os.path.join(PROC_DIR, "features_val.csv"))
    test_df  = pd.read_csv(os.path.join(PROC_DIR, "features_test.csv"))

    poly_tr  = (train_df["is_alloy"] < 0).values
    poly_val = (val_df["is_alloy"]   < 0).values
    poly_te  = (test_df["is_alloy"]  < 0).values
    alloy_tr = (train_df["is_alloy"] > 0).values
    alloy_val= (val_df["is_alloy"]   > 0).values
    alloy_te = (test_df["is_alloy"]  > 0).values

    # ── Train POLYMER model ──────────────────────────────────────────────────
    poly_models = train_class_models(
        X_train, y_train, poly_tr, X_val, y_val, poly_val, "POLYMER", seed
    )

    # ── Train ALLOY model ────────────────────────────────────────────────────
    alloy_models = train_class_models(
        X_train, y_train, alloy_tr, X_val, y_val, alloy_val, "ALLOY", seed
    )

    # ── Save combined bundle ─────────────────────────────────────────────────
    bundle = {
        "polymer": poly_models,
        "alloy":   alloy_models,
        "feature_cols": FEATURE_COLS,
        "target_cols":  TARGET_COLS,
    }
    save_path = os.path.join(MODELS_DIR, "material_predictor.pkl")
    joblib.dump(bundle, save_path)
    print(f"\n[train] ✅  Models saved → {save_path}")

    # ── Test-set results (held-out, never seen) ──────────────────────────────
    def eval_set(models, X_te, y_te, label):
        print(f"\n[train] {label} HELD-OUT test-set results (n={len(X_te)}):")
        print(f"  {'Target':<36}  {'MAE':>10}  {'R²':>8}")
        print("  " + "─" * 60)
        for t in TARGET_COLS:
            m = models[t]
            p = m["meta"].predict(
                np.column_stack([m["rf"].predict(X_te), m["xgb"].predict(X_te)])
            )
            y = y_te[t].values
            print(f"  {t:<36}  {mean_absolute_error(y,p):>10.3f}  {r2_score(y,p):>8.4f}")

    X_te_poly  = X_test[poly_te].reset_index(drop=True)
    y_te_poly  = y_test[poly_te].reset_index(drop=True)
    X_te_alloy = X_test[alloy_te].reset_index(drop=True)
    y_te_alloy = y_test[alloy_te].reset_index(drop=True)

    eval_set(poly_models,  X_te_poly,  y_te_poly,  "POLYMER")
    if len(X_te_alloy) > 0:
        eval_set(alloy_models, X_te_alloy, y_te_alloy, "ALLOY")

    print("\n[train] Run 'make evaluate' for full report + 5 graphs.")
    return bundle, X_test, y_test


if __name__ == "__main__":
    train_all()
