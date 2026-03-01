import os, sys, traceback
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
from data_prep import TARGETS, FEATURES

app = FastAPI(title="Eco-Material Predictor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ADV_MODEL_PATH = os.path.join(ROOT, "models", "advanced_alloy_predictor.pkl")
MODEL_PATH = os.path.join(ROOT, "models", "material_predictor.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")

try:
    bundle  = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    
    # Load original raw data for string-based name/alternative lookups in the Green Tab
    df_raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "materials_dataset.csv"))
    
    print("✅ Base Models & Name Database Loaded successfully")
except Exception as e:
    print(f"❌ Failed to load base models:")
    traceback.print_exc()
    bundle = scaler = df_raw = None

class PredictionInput(BaseModel):
    inputs: dict
    mode: str 

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.post("/predict")
def predict(data: PredictionInput):
    if not bundle:
        raise HTTPException(500, "Model not loaded — run make train first")
    
    # Feature columns defined organically in the dual-matrix bundle
    feature_cols = bundle["feature_cols"]
    
    # Fill missing features with 0.0 (e.g. polymers don't send shear_modulus)
    payload_df = {col: data.inputs.get(col, 0.0) for col in feature_cols}
    X = pd.DataFrame([payload_df])
    
    # Organize columns identically to how the scaler was trained
    X = X[feature_cols]
    X_scaled = scaler.transform(X)

    model_type = "metal" if data.mode == "metal" else "polymer"
    preds, confs = {}, {}
    
    for target in bundle["target_cols"]:
        m = bundle[model_type][target]
        base = np.column_stack([m["rf"].predict(X_scaled), m["xgb"].predict(X_scaled)])
        meta = m["meta"].predict(base)[0]
        
        # Clip standard structural physical constraints
        if target in ("density_gcm3", "water_absorption_pct", "oxygen_permeability_barrer"):
            meta = max(0.001, meta)
        
        preds[target] = float(meta)
        confs[target] = float(np.std([t.predict(X_scaled)[0] for t in m["rf"].estimators_]))
        
    return {"predictions": preds, "confidence": confs}


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
    from recommend import find_green_alternatives
    res = find_green_alternatives(material_name, top_n=3)
    if res["error"]:
        raise HTTPException(404, res["error"])
        
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
