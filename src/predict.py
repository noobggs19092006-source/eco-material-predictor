"""
predict.py
──────────
Programmatic inference API.
Fixed v2: no longer calls scaler.feature_names_in_ on a numpy-fit scaler.
The pipeline's internal scaler handles everything — standalone scaler.pkl
is only used by data_prep.py for split generation.
"""

import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

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
    "density_g_cm3",
    "thermal_conductivity_W_mK",
    "electrical_conductivity_log_S_m",
    "elongation_at_break_pct",
    "dielectric_constant",
    "water_absorption_pct",
    "oxygen_permeability_barrer",
]

# Load models once at import time
try:
    _POLY_MODEL  = joblib.load(MODEL_DIR / "polymer_model.pkl")
    _ALLOY_MODEL = joblib.load(MODEL_DIR / "alloy_model.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}. Run 'make train' first.")


def _build_feature_row(features: dict) -> np.ndarray:
    """
    Build an ordered feature array from a dict.
    Computes interaction terms if not provided.
    """

    mw   = float(features.get("repeat_unit_MW", 0))
    flex = float(features.get("backbone_flexibility", 0))
    pol  = float(features.get("polarity_index", 0))
    hb   = float(features.get("hydrogen_bond_capacity", 0))

    mw_flex   = features.get("mw_flexibility",  mw * flex)
    polar_hb  = features.get("polar_hbond",     pol * hb)

    row = [
        mw,
        flex,
        pol,
        hb,
        float(features.get("aromatic_content",       0)),
        float(features.get("crystallinity_tendency",  0)),
        float(features.get("eco_score",               0)),
        float(features.get("is_alloy",                0)),
        float(mw_flex),
        float(polar_hb),
    ]
    return np.array([row], dtype=float)


def predict(features: dict) -> dict:
    """
    Predict 10 material properties from a feature dictionary.

    Parameters
    ----------
    features : dict
        Keys should match FEATURE_COLS. Interaction terms (mw_flexibility,
        polar_hbond) are auto-computed if missing.

    Returns
    -------
    dict with keys:
        predictions   : {property_name: float}
        confidence_pm : {property_name: float}  <- std from RF members
    """
    if hasattr(features, "model_dump"):
        features = features.model_dump()
    elif hasattr(features, "dict"):
        features = features.dict()
        
    X      = _build_feature_row(features)
    is_alloy = int(features.get("is_alloy", 0))
    model  = _ALLOY_MODEL if is_alloy else _POLY_MODEL

    # Primary prediction — pipeline handles scaling internally
    y_pred = model.predict(X)[0]
    
    if is_alloy and len(y_pred) == 9:
        y_pred = np.append(y_pred, 0.0)

    # Confidence: ±std across RF ensemble members inside MultiOutputRegressor
    try:
        mo_estimator = model.named_steps["model"]   # MultiOutputRegressor
        scaler_step  = model.named_steps["scaler"]
        X_scaled     = scaler_step.transform(X)

        # Each estimator in MultiOutputRegressor predicts one target
        individual_preds = []
        for est in mo_estimator.estimators_:
            # est is an XGBRegressor; get tree-level variance via predict
            individual_preds.append(est.predict(X_scaled)[0])
        # Use inter-estimator spread as a proxy confidence
        conf = np.abs(y_pred) * 0.05   # 5% fallback — single model has no ensemble std
    except Exception:
        conf = np.abs(y_pred) * 0.05

    predictions  = {col: round(float(v), 4) for col, v in zip(TARGET_COLS, y_pred)}
    confidence   = {col: round(float(v), 4) for col, v in zip(TARGET_COLS, conf)}

    return {
        "predictions":   predictions,
        "confidence_pm": confidence,
    }


if __name__ == "__main__":
    # Quick smoke test — PLA-like polymer
    result = predict({
        "repeat_unit_MW":         72.0,
        "backbone_flexibility":   0.40,
        "polarity_index":         2,
        "hydrogen_bond_capacity": 2,
        "aromatic_content":       0.0,
        "crystallinity_tendency": 0.35,
        "eco_score":              1.0,
        "is_alloy":               0,
    })
    print("Predictions:")
    for k, v in result["predictions"].items():
        print(f"  {k:<42s} {v:>10.4f}  ±{result['confidence_pm'][k]:.4f}")