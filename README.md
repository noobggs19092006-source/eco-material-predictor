# 🌿 Eco-Material Property Predictor

A machine-learning pipeline that predicts **10 material properties** of eco-friendly engineering plastics and alloys — eliminating costly physical lab testing.

> 5-Fold Cross-Validation · Real CO₂ data (ICE Database v2.0) · SHAP Explainability · 37/37 tests passing

---

## 📊 Model Performance — 5-Fold Cross-Validation

Scores are mean ± std across 5 held-out folds. Fixed `random_state=42` throughout. No seed selection performed.

| Property | Unit | Polymer R² ± std | Alloy R² ± std |
|---|---|---|---|
| Glass Transition Temperature | °C | 0.939 ± 0.007 | 0.651 ± 0.019 |
| Tensile Strength | MPa | 0.881 ± 0.008 | 0.751 ± 0.044 |
| Young's Modulus | GPa | 0.929 ± 0.006 | 0.797 ± 0.029 |
| Density | g/cm³ | 0.955 ± 0.008 | 0.980 ± 0.003 |
| Thermal Conductivity | W/m·K | 0.846 ± 0.024 | 0.834 ± 0.011 |
| Electrical Conductivity | log S/m | 0.949 ± 0.014 | 0.773 ± 0.035 |
| Elongation at Break | % | 0.933 ± 0.009 | 0.935 ± 0.009 |
| Dielectric Constant | — | 0.949 ± 0.003 | 0.927 ± 0.010 |
| Water Absorption | % | 0.948 ± 0.013 | 0.858 ± 0.006 |
| O2 Permeability | Barrers | 0.898 ± 0.015 | 0.000 ± 0.000 |
| **Mean R²** | | **0.923 ± 0.011** | **0.751 ± 0.016** |

Note: O2 permeability for metals is 0 by definition — metals are perfect O2 barriers.

---

## 🌍 Real Eco-Impact — ICE Database v2.0

Every prediction includes verified carbon footprint data from the Inventory of Carbon and Energy v2.0 (Hammond and Jones, University of Bath, 2011).

| Material | kg CO2e per kg | Category |
|---|---|---|
| Bio-PE | 1.91 | bio-based polymer |
| Steel-304 | 2.09 | metal alloy |
| PHA | 2.63 | bio-based polymer |
| PLA | 3.86 | bio-based polymer |
| Bio-PA | 4.22 | bio-based polymer |
| Bio-Epoxy | 5.10 | bio-based polymer |
| Al-alloy | 8.24 | metal alloy |
| HEA | 18.50 | metal alloy |
| Ti-alloy | 35.00 | metal alloy |

The `/carbon-impact` API endpoint converts these into real-world equivalents. Example:

```
POST /carbon-impact
{"material_name": "PHA", "compare_with": "PP", "mass_kg": 500}

Response:
"Using PHA instead of PP for 500 kg saves 175 kg CO2 —
 equivalent to not driving 833 km."
```

---

## 🚀 Quick Start — Judges and First-Time Users

```bash
# 1. Clone the repo
git clone https://github.com/noobggs19092006-source/eco-material-predictor.git
cd eco-material-predictor

# 2. Create virtual environment and install all dependencies (about 2 min)
bash setup.sh
source venv/bin/activate

# 3. Generate the 800-row QSPR dataset with ICE carbon footprint data (5 sec)
python scripts/perfect_dataset.py

# 4. Train polymer and alloy models with 5-Fold CV (about 5 min)
make train
# Prints mean +/- std R2 across 5 folds for every property in the terminal

# 5. Generate 7 evaluation charts including SHAP and CO2 impact chart
make evaluate

# 6. Verify integrity — all 37 tests should pass
make test

# 7. Launch the full web app
make app
# API docs:  http://localhost:8000/docs
# Frontend:  http://localhost:5173
```

---

## ⚡ Developer Quick Resume

```bash
source venv/bin/activate

# Full retrain from scratch
make clean
python scripts/perfect_dataset.py
make train
make evaluate

# Just relaunch the app (models already trained)
make app

# Run CLI predictor
make predict

# Run full test suite
make test
```

---

## 📁 Project Structure

```
eco-material-predictor/
├── data/
│   ├── raw/
│   │   └── materials_dataset.csv        800-row QSPR dataset (400 polymers + 400 alloys)
│   └── processed/                       auto-generated after make train
│       ├── features_train.csv           70 percent — 560 rows
│       ├── features_val.csv             10 percent — 80 rows
│       └── features_test.csv            20 percent — 160 rows
├── models/
│   ├── polymer_model.pkl                XGBoost pipeline for polymers
│   ├── alloy_model.pkl                  XGBoost pipeline for alloys
│   └── scaler.pkl                       fitted StandardScaler
├── results/
│   ├── cv_report_polymers.txt           5-Fold CV mean +/- std R2 (polymers)
│   ├── cv_report_alloys.txt             5-Fold CV mean +/- std R2 (alloys)
│   ├── evaluation_report_polymers.txt   MAE / RMSE / R2 per target
│   ├── evaluation_report_alloys.txt
│   ├── 01_actual_vs_predicted_polymers.png
│   ├── 02_actual_vs_predicted_alloys.png
│   ├── 03_feature_importance_heatmap.png
│   ├── 05_shap_summary_polymers.png     SHAP: why each prediction was made
│   ├── 06_carbon_footprint_eco_impact.png  CO2 savings chart (ICE Database)
│   └── 07_shap_summary_alloys.png
├── scripts/
│   └── perfect_dataset.py              QSPR generator (Brandrup1999 / Ashby2011)
├── src/
│   ├── __init__.py
│   ├── data_prep.py                    feature engineering + 70/10/20 split
│   ├── train.py                        5-Fold CV + final model training
│   ├── evaluate.py                     7 charts: SHAP, CO2 impact, residuals
│   ├── predict.py                      programmatic inference API
│   ├── cli.py                          interactive terminal predictor
│   ├── api.py                          FastAPI: /predict /carbon-impact /recommend
│   ├── recommend.py                    Green Alternative Recommender engine
│   └── generate_pdb.py                 PDB file generator for VMD visualization
├── frontend/
│   ├── src/App.jsx                     UI: sliders, radar chart, CO2 dashboard
│   ├── src/index.css                   liquid-glass biopunk CSS
│   └── index.html
├── tests/
│   └── test_pipeline.py                37 pytest unit tests — all passing
├── requirements.txt
├── setup.sh                            Linux one-shot environment installer
├── pytest.ini
├── Makefile
├── Dockerfile
├── render.yaml
└── README.md
```

---

## 🤖 Model Architecture

```
800 rows total — 400 polymers + 400 alloys

Validation strategy: 5-Fold Cross-Validation
  Reports mean +/- std R2 across 5 held-out folds
  Final model trained on full dataset after CV confirms generalisation
  Fixed random_state=42 throughout — fully reproducible

Pipeline (one per material class):

  Raw features (10)
       |
  StandardScaler
       |
  MultiOutputRegressor
    └── XGBRegressor
          n_estimators = 300
          learning_rate = 0.03
          max_depth = 6
          subsample = 0.8
          colsample_bytree = 0.8
       |
  10 simultaneous property predictions

Two separate pipelines: polymer_model.pkl + alloy_model.pkl
```

---

## 🔬 Input Features

| Feature | Description | Range |
|---|---|---|
| repeat_unit_MW | Molecular weight of repeat unit (g/mol) | 10 – 600 |
| backbone_flexibility | Chain stiffness (0 = rigid, 1 = flexible) | 0.0 – 1.0 |
| polarity_index | Polarity (0 = nonpolar, 3 = highly polar) | 0 – 3 |
| hydrogen_bond_capacity | H-bond strength | 0 – 5 |
| aromatic_content | Fraction of aromatic carbons | 0.0 – 1.0 |
| crystallinity_tendency | Crystallinity (0 = amorphous, 1 = crystalline) | 0.0 – 1.0 |
| eco_score | Bio-based sustainability (0 = petroleum, 1 = bio-based) | 0.0 – 1.0 |
| is_alloy | Material class (0 = polymer, 1 = metal alloy) | 0 or 1 |
| mw_flexibility | Interaction term: MW × flexibility | computed |
| polar_hbond | Interaction term: polarity × H-bond capacity | computed |

---

## 🧪 Dataset

**800 rows** — 400 bio-based polymers + 400 eco-rated metal alloys.

Generated by `scripts/perfect_dataset.py` using peer-reviewed QSPR formulas:

- Polymers: Brandrup and Immergut (1999), van Krevelen (2009)
- Alloys: Ashby (2011), Callister (2018)

With 4 to 8 percent Gaussian noise matching real lab measurement uncertainty.

Every row includes:

- 10 input molecular and structural features
- 10 predicted material properties
- `carbon_footprint_kgCO2_per_kg` from ICE Database v2.0 (University of Bath)
- `carbon_saving_vs_conventional_pct` vs petroleum-based equivalent
- `data_source` full provenance label

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check + version |
| GET | /materials | List all materials in ICE database |
| POST | /predict | Predict 10 properties from input features |
| POST | /carbon-impact | Real CO2 footprint + savings calculation |
| GET | /recommend/{material} | Top 3 greener alternative materials |

Full interactive docs available at `http://localhost:8000/docs` when running locally.

---

## 💻 Programmatic Usage

```python
from src.predict import predict

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

print(result["predictions"])
# {"Tg_celsius": 62.4, "tensile_strength_MPa": 87.4, ...}

print(result["confidence_pm"])
# {"Tg_celsius": 3.1, ...}  — plus/minus std
```

---

## 🔍 SHAP Explainability

Running `make evaluate` generates SHAP summary plots for both polymer and alloy models showing which molecular features drive each prediction. Example interpretation:

- High `crystallinity_tendency` → higher Tg and tensile strength
- High `backbone_flexibility` → lower Tg, higher elongation at break
- High `eco_score` → correlates with bio-based material class

These plots are saved to `results/05_shap_summary_polymers.png` and `results/07_shap_summary_alloys.png`.

---

## 🧬 VMD Molecular Visualization

The `make predict` command generates `.pdb` structure files for any predicted material. To visualize:

1. Open VMD and load the `.pdb` file
2. Go to Graphics → Representations
3. Set Drawing Method to Licorice (recommended) or CPK
4. Set Coloring Method to Element (O = red, C = cyan, H = white, N = blue)
5. Go to Display → Display Settings → set Axes to Off for a clean screenshot

---

## 🌍 Deploy to Render (Free)

```bash
# Push to GitHub
git add .
git commit -m "deploy"
git push

# Then on render.com:
# New → Web Service → Connect your GitHub repo
# Render reads render.yaml and Dockerfile automatically
# Installs Python + Node.js, trains models, builds React, goes live
```

### ⚠️ Free Tier Constraints (502 Bad Gateway)
Render's free tier has a strict **512 MB RAM limit**. 
If you encounter a `502 Bad Gateway` or `No open ports detected` error during deployment:
1. **Model Size:** Our ensemble models (`polymer_model.pkl` and `alloy_model.pkl`) uncompress memory-map to ~220-300MB RAM. Do NOT increase `n_estimators` beyond `300` in `src/train.py`, or the Linux kernel's OOM (Out-of-Memory) killer will silently terminate your FastAPI container.
2. **Startup Timeouts:** The `src/api.py` utilizes asynchronous static mounts to bind `$PORT` instantly before unpacking the heavy ML models. This guarantees passing Render's 60-second health check.

---

## ✅ Test Suite

```
37 tests across 6 categories:

TestDataLoading     — 13 tests: CSV exists, column names, ranges, nulls, provenance
TestPreprocessing   —  6 tests: splits generated, no NaN, scaler saved, sizes correct
TestModel           —  5 tests: model files exist, 10 targets, pipeline structure
TestPredictions     —  7 tests: all targets returned, PLA physics, determinism
TestPDB             —  4 tests: list materials, get PDB, invalid key raises, guess key
TestCLI             —  2 tests: CLI imports without error
```

Run with: `make test`

---

## 📚 References

- Hammond, G. and Jones, C. (2011). Inventory of Carbon and Energy (ICE) v2.0. University of Bath.
- Brandrup, J. and Immergut, E.H. (1999). Polymer Handbook, 4th ed. Wiley.
- van Krevelen, D.W. (2009). Properties of Polymers, 4th ed. Elsevier.
- Ashby, M.F. (2011). Materials Selection in Mechanical Design, 4th ed. Butterworth-Heinemann.
- Callister, W.D. (2018). Materials Science and Engineering: An Introduction, 10th ed. Wiley.

---

## License

MIT — free to use, modify, and build upon.
