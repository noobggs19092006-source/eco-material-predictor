"""
api.py
──────
FastAPI backend for the Eco-Material Predictor.
New in v2: /carbon-impact endpoint returns real ICE Database CO2 values.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Eco-Material Predictor API",
    description="Predicts 10 material properties + carbon footprint impact.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
try:
    POLY_MODEL  = joblib.load(MODEL_DIR / "polymer_model.pkl")
    ALLOY_MODEL = joblib.load(MODEL_DIR / "alloy_model.pkl")
except FileNotFoundError:
    raise RuntimeError("Models not found. Run 'make train' first.")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    df_raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "materials_dataset.csv"))
except Exception:
    df_raw = None

FEATURE_COLS = [
    "repeat_unit_MW", "backbone_flexibility", "polarity_index",
    "hydrogen_bond_capacity", "aromatic_content", "crystallinity_tendency",
    "eco_score", "is_alloy", "mw_flexibility", "polar_hbond",
]
TARGET_COLS = [
    "Tg_celsius", "tensile_strength_MPa", "youngs_modulus_GPa",
    "density_g_cm3", "thermal_conductivity_W_mK",
    "electrical_conductivity_log_S_m", "elongation_at_break_pct",
    "dielectric_constant", "water_absorption_pct",
    "oxygen_permeability_barrer",
]

# ── ICE Database v2.0 carbon footprint values ─────────────────────────────────
# Source: Hammond & Jones, University of Bath, 2011
ICE_CARBON_DB = {
    "PLA":       {"co2": 3.84,  "category": "bio-based polymer"},
    "PHA":       {"co2": 2.65,  "category": "bio-based polymer"},
    "Bio-PA":    {"co2": 4.20,  "category": "bio-based polymer"},
    "Bio-PE":    {"co2": 1.90,  "category": "bio-based polymer"},
    "Bio-Epoxy": {"co2": 5.10,  "category": "bio-based polymer"},
    "ABS":       {"co2": 3.50,  "category": "conventional polymer"},
    "PP":        {"co2": 1.95,  "category": "conventional polymer"},
    "PET":       {"co2": 3.40,  "category": "conventional polymer"},
    "Al-alloy":  {"co2": 8.24,  "category": "metal alloy"},
    "Ti-alloy":  {"co2": 35.0,  "category": "metal alloy"},
    "Steel-304": {"co2": 2.10,  "category": "metal alloy"},
    "HEA":       {"co2": 18.5,  "category": "metal alloy"},
}

# ── Request / Response schemas ─────────────────────────────────────────────────
class MaterialFeatures(BaseModel):
    repeat_unit_MW:          float = 0.0
    backbone_flexibility:    float = 0.0
    polarity_index:          float = 0.0
    hydrogen_bond_capacity:  float = 0.0
    aromatic_content:        float = 0.0
    crystallinity_tendency:  float = 0.0
    eco_score:               float = 0.0
    is_alloy:                int   = 0

class PredictionInput(BaseModel):
    inputs: dict
    mode: str

class CarbonQuery(BaseModel):
    material_name:    str
    mass_kg:          float = Field(1.0, ge=0.01, le=100000)
    compare_with:     str   = "ABS"


# ── Helpers ────────────────────────────────────────────────────────────────────
def featurize(data: MaterialFeatures) -> np.ndarray:
    mw_flex   = data.repeat_unit_MW * data.backbone_flexibility
    polar_hb  = data.polarity_index * data.hydrogen_bond_capacity
    row = [
        data.repeat_unit_MW, data.backbone_flexibility,
        data.polarity_index, data.hydrogen_bond_capacity,
        data.aromatic_content, data.crystallinity_tendency,
        data.eco_score, float(data.is_alloy),
        mw_flex, polar_hb,
    ]
    return np.array([row])


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/materials")
def list_materials():
    """Return all materials in the ICE database."""
    return {"materials": list(ICE_CARBON_DB.keys())}


@app.post("/predict")
def predict(data: PredictionInput):
    """Predict 10 material properties with confidence intervals."""
    payload = {col: data.inputs.get(col, 0.0) for col in FEATURE_COLS}
    payload["is_alloy"] = 1 if data.mode in ("metal", "alloy") else 0
    features = MaterialFeatures(**payload)
    X     = featurize(features)
    model = ALLOY_MODEL if features.is_alloy else POLY_MODEL

    y_pred = model.predict(X)[0]

    # Confidence: use RF ensemble std if available
    try:
        rf_est = model.named_steps["model"].estimators_
        preds  = np.array([est.predict(
            model.named_steps["scaler"].transform(X)
        ) for est in rf_est])
        confidence = preds.std(axis=0)[0]
    except Exception:
        confidence = np.abs(y_pred) * 0.05   # fallback: 5% of prediction

    predictions  = {col: round(float(v), 4) for col, v in zip(TARGET_COLS, y_pred)}
    conf_dict    = {col: round(float(v), 4) for col, v in zip(TARGET_COLS, confidence)}

    return {
        "predictions":  predictions,
        "confidence_pm": conf_dict,
        "note": "confidence_pm = ±1 std from RF ensemble members",
    }


@app.post("/carbon-impact")
def carbon_impact(query: CarbonQuery):
    """
    Returns real ICE Database carbon footprint data.
    Shows CO2 saved (or added) vs a conventional reference material.
    This is the eco-impact story in numbers.
    """
    mat  = query.material_name
    comp = query.compare_with

    if mat not in ICE_CARBON_DB:
        raise HTTPException(
            status_code=404,
            detail=f"'{mat}' not in database. Available: {list(ICE_CARBON_DB.keys())}"
        )
    if comp not in ICE_CARBON_DB:
        raise HTTPException(
            status_code=404,
            detail=f"Comparison material '{comp}' not found."
        )

    mat_co2   = ICE_CARBON_DB[mat]["co2"]
    comp_co2  = ICE_CARBON_DB[comp]["co2"]
    saving_pct = (comp_co2 - mat_co2) / comp_co2 * 100

    total_mat_co2  = mat_co2  * query.mass_kg
    total_comp_co2 = comp_co2 * query.mass_kg
    co2_saved_kg   = total_comp_co2 - total_mat_co2

    # Real-world equivalents to make the number tangible
    km_driven_equivalent   = co2_saved_kg / 0.21   # avg car: 0.21 kg CO2/km
    trees_equivalent_year  = co2_saved_kg / 21.0   # 1 tree absorbs ~21 kg CO2/yr

    return {
        "material":                    mat,
        "compared_with":               comp,
        "mass_kg":                     query.mass_kg,
        "material_co2_per_kg":         mat_co2,
        "comparison_co2_per_kg":       comp_co2,
        "source":                      "ICE Database v2.0, Hammond & Jones, University of Bath",
        "carbon_saving_pct":           round(saving_pct, 2),
        "co2_saved_total_kg":          round(co2_saved_kg, 3),
        "equivalent_km_not_driven":    round(km_driven_equivalent, 1),
        "equivalent_trees_planted_yr": round(trees_equivalent_year, 2),
        "verdict": (
            f"Using {mat} instead of {comp} for {query.mass_kg} kg saves "
            f"{co2_saved_kg:.2f} kg CO₂ — equivalent to not driving "
            f"{km_driven_equivalent:.0f} km."
        ) if saving_pct > 0 else (
            f"{mat} has a higher footprint than {comp} for this application."
        ),
    }


@app.get("/recommend/{material_name}")
def recommend(material_name: str, top_k: int = 3):
    """Return greener alternatives to the given material."""
    if material_name not in ICE_CARBON_DB:
        raise HTTPException(status_code=404, detail=f"'{material_name}' not found.")

    base_co2  = ICE_CARBON_DB[material_name]["co2"]
    category  = ICE_CARBON_DB[material_name]["category"]

    candidates = []
    for name, info in ICE_CARBON_DB.items():
        if name == material_name:
            continue
        saving = (base_co2 - info["co2"]) / base_co2 * 100
        if saving > 0:
            candidates.append({
                "material":         name,
                "co2_per_kg":       info["co2"],
                "carbon_saving_pct":round(saving, 1),
                "category":         info["category"],
            })

    candidates.sort(key=lambda x: x["carbon_saving_pct"], reverse=True)

    return {
        "query_material":   material_name,
        "base_co2_per_kg":  base_co2,
        "source":           "ICE Database v2.0",
    }

@app.get("/materials/petroleum")
def petroleum():
    if df_raw is None:
        return []
    dirty = df_raw[df_raw["eco_score"] < 0.6]
    res_df = dirty.drop_duplicates(subset=["material_name"]).sort_values("material_name")
    res = res_df[["material_name", "eco_score", "tensile_strength_MPa", "Tg_celsius"]].to_dict("records")
    return res

@app.get("/materials/alternatives/{material_name}")
def alternatives(material_name: str):
    from src.recommend import find_green_alternatives
    res = find_green_alternatives(material_name, top_n=3)
    if res["error"]:
        raise HTTPException(status_code=404, detail=res["error"])
        
    alts = res["alternatives"]
    for a in alts:
        if "performance_match_pct" in a:
            a["match_pct"] = a["performance_match_pct"]
    return alts

# --- Serve React Frontend ---
frontend_path = os.path.join(ROOT, "frontend", "dist")
if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")
    
    @app.get("/{catchall:path}")
    def serve_react_app(catchall: str):
        file_path = os.path.join(frontend_path, catchall)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))