import os, sys
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import TARGET_COLS

app = FastAPI(title="Eco-Material Predictor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH   = os.path.join(ROOT, "models", "material_predictor.pkl")
SCALER_PATH  = os.path.join(ROOT, "models", "scaler.pkl")
DATASET_PATH = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")

try:
    bundle  = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    df_raw  = pd.read_csv(DATASET_PATH)
except Exception:
    bundle = scaler = df_raw = None


class PredictionInput(BaseModel):
    repeat_unit_MW: float
    backbone_flexibility: float
    polarity_index: float
    hydrogen_bond_capacity: float
    aromatic_content: float
    crystallinity_tendency: float
    eco_score: float
    is_alloy: int
    mw_flexibility: float
    polar_hbond: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}


@app.post("/predict")
def predict(data: PredictionInput):
    if not bundle:
        raise HTTPException(500, "Model not loaded — run make train first")
    X = pd.DataFrame([data.model_dump()])
    X_scaled = scaler.transform(X)          # numpy array
    model_type = "alloy" if data.is_alloy else "polymer"
    preds, confs = {}, {}
    for target in TARGET_COLS:
        m = bundle[model_type][target]
        base = np.column_stack([m["rf"].predict(X_scaled), m["xgb"].predict(X_scaled)])
        meta = m["meta"].predict(base)[0]
        if target in ("density_gcm3", "water_absorption_pct"):
            meta = max(0.01, meta)
        preds[target] = float(meta)
        confs[target] = float(np.std([t.predict(X_scaled)[0] for t in m["rf"].estimators_]))
    return {"predictions": preds, "confidence": confs}


@app.get("/materials/petroleum")
def petroleum():
    if df_raw is None:
        return []
    d = df_raw[(df_raw["eco_score"] < 0.6) & (df_raw["material_class"] == "polymer")]
    return d[["material_name", "eco_score", "tensile_strength_MPa", "Tg_celsius"]].to_dict("records")


@app.get("/materials/alternatives/{material_name}")
def alternatives(material_name: str):
    if df_raw is None:
        raise HTTPException(500, "Dataset not loaded")
    # find the dirty material
    row = df_raw[df_raw["material_name"].str.lower() == material_name.strip().lower()]
    if row.empty:
        row = df_raw[df_raw["material_name"].str.lower().str.contains(material_name.strip().lower(), na=False)]
    if row.empty:
        raise HTTPException(404, f"Material '{material_name}' not found")
    target = row.iloc[0]
    candidates = df_raw[(df_raw["eco_score"] >= 0.7) & (df_raw["material_class"] == "polymer") &
                        (df_raw["material_name"] != target["material_name"])].copy()
    keys = ["tensile_strength_MPa", "youngs_modulus_GPa", "Tg_celsius", "density_gcm3", "elongation_at_break_pct"]
    for k in keys:
        rng = df_raw[k].max() - df_raw[k].min() or 1
        candidates[f"_d{k}"] = ((candidates[k] - target[k]) / rng) ** 2
    candidates["_dist"] = np.sqrt(candidates[[f"_d{k}" for k in keys]].sum(axis=1))
    
    # Calculate an overall match percentage where 0 distance = 100% match
    candidates["match_pct"] = np.clip(100 - (candidates["_dist"] * 100), 0, 100).round(1)
    
    top = candidates.nsmallest(3, "_dist")
    cols = ["material_name", "eco_score", "tensile_strength_MPa", "Tg_celsius",
            "youngs_modulus_GPa", "density_gcm3", "elongation_at_break_pct", "match_pct"]
    return top[cols].replace({np.nan: None}).to_dict("records")

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
