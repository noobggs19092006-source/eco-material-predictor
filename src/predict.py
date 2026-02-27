"""
predict.py — Inference API for all 10 material properties + confidence intervals.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import FEATURE_COLS, TARGET_COLS, TARGET_META


def _load_bundle():
    path = os.path.join(MODELS_DIR, "material_predictor.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}.\nRun  'make train'  first.")
    return joblib.load(path)


def _load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at {path}.\nRun  'make train'  first.")
    return joblib.load(path)


def predict(features: dict) -> dict:
    """
    Predict all 10 material properties from molecular/structural features.

    Parameters
    ----------
    features : dict
        Keys: repeat_unit_MW, backbone_flexibility, polarity_index,
              hydrogen_bond_capacity, aromatic_content,
              crystallinity_tendency, eco_score, is_alloy

    Returns
    -------
    dict:
        predictions  → {target: value}
        confidence   → {target: ±std from RF ensemble}
        input_used   → full engineered feature dict
    """
    feat = dict(features)
    feat["mw_flexibility"] = feat["repeat_unit_MW"] * feat["backbone_flexibility"]
    feat["polar_hbond"]    = feat["polarity_index"] * feat["hydrogen_bond_capacity"]

    X_raw = pd.DataFrame([feat])[FEATURE_COLS]
    scaler = _load_scaler()
    X_sc   = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_COLS)
    X_np   = X_sc.values

    bundle  = _load_bundle()
    
    if feat.get("is_alloy", 0) == 1:
        models = bundle["alloy"]
    else:
        models = bundle["polymer"]

    predictions, confidence = {}, {}
    for target in TARGET_COLS:
        m        = models[target]
        rf_trees = np.array([e.predict(X_np) for e in m["rf"].estimators_])
        rf_mean  = rf_trees.mean(axis=0)
        rf_std   = rf_trees.std(axis=0)
        xgp      = m["xgb"].predict(X_sc)
        pred     = m["meta"].predict(np.column_stack([rf_mean, xgp]))[0]
        predictions[target] = float(pred)
        confidence[target]  = float(rf_std[0])

    return {"predictions": predictions, "confidence": confidence, "input_used": feat}


def format_value(target: str, val: float) -> str:
    """Convert log10 conductivity → human-readable scientific notation."""
    if target == "log10_elec_conductivity":
        return f"10^{val:.2f} S/m  ({val:.2f} log₁₀)"
    return f"{val:.3f}"


if __name__ == "__main__":
    demo = dict(repeat_unit_MW=72, backbone_flexibility=0.40,
                polarity_index=2, hydrogen_bond_capacity=2,
                aromatic_content=0.0, crystallinity_tendency=0.35,
                eco_score=1.0, is_alloy=0)
    r = predict(demo)
    print("Demo (PLA-like polymer):")
    for t, v in r["predictions"].items():
        name, unit, icon = TARGET_META[t]
        print(f"  {icon} {name:<33} = {format_value(t, v):>25} {unit}  ±{r['confidence'][t]:.3f}")
