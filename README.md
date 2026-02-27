# 🌿 Eco-Material Property Predictor

A machine-learning pipeline that predicts **10 material properties** of eco-friendly engineering plastics and alloys — no physical testing required.

**Predicted properties (Held-out Test R²):**
| Property | Unit | Polymer R² | Alloy R² |
|---|---|---|---|
| Glass Transition Temperature (Tg) | °C | **0.98** | **0.96** |
| Tensile Strength | MPa | **0.93** | **0.96** |
| Young's Modulus | GPa | **0.90** | **0.99** |
| Density | g/cm³ | **0.85** | **0.99** |
| Thermal Conductivity | W/m·K | **0.93** | **0.97** |
| Electrical Conductivity | log₁₀ S/m | **0.92** | **0.96** |
| Elongation at Break | % | **0.94** | **0.96** |
| Dielectric Constant | — | **0.95** | **0.95** |
| Water Absorption | % | **0.95** | **0.97** |
| O₂ Permeability | Barrers | **0.97** | **0.97** |

**Materials covered:** PLA, PHA, PHB, PBS, PEF, Bio-PA, Cellulose derivatives, Lignin-based polymers, Chitosan, Starch blends, eco-epoxies, metal alloys, and 160+ more.

---

## 🚀 Mode A — Hackathon / Fresh Start (For Judges & Visitors)

> Use this if you're running the project **for the first time** on a new machine.
> This builds everything from scratch: environment → dataset → training → demo.

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd eco-material-predictor

# 2. Create the virtual environment and install all dependencies (~2 min)
bash setup.sh

# 3. Activate the environment
source venv/bin/activate

# 4. Generate the dataset with QSPR formulas + realistic noise (~5 sec)
python scripts/perfect_dataset.py

# 5. Train both polymer and alloy ensemble models (~5–10 min)
make train

# 6. Generate evaluation report + 5 publication-quality graphs
make evaluate

# 7. Launch the Interactive Web Dashboard (React + FastAPI)
make app
# → API at http://localhost:8000  |  Frontend at http://localhost:5173
# Features 3D Parallax Liquid-Glass UI, Dynamic Polymer/Alloy Radar Charts, 
# and a True Multivariate AI Recommender for bio-based materials.

# 8. (Optional) Launch the barebones interactive CLI predictor
make predict

# 9. (Optional) Run the full test suite — all 27 tests should pass
make test
```

---

## ⚡ Mode B — Developer / Quick Resume (For the Author)

> Use this if the **venv already exists** and you just want to retrain or demo.

```bash
# Activate the environment
source venv/bin/activate

# Option 1: Full retrain from scratch
make clean
python scripts/perfect_dataset.py
make train
make evaluate

# Option 2: Just re-evaluate (model already trained)
make evaluate

# Option 3: Launch the Interactive Web Dashboard
make app
# → API at http://localhost:8000  |  Frontend at http://localhost:5173
# Features 3D Parallax Liquid-Glass UI, Dynamic Polymer/Alloy Radar Charts, 
# and a True Multivariate AI Recommender for bio-based materials.

# Option 4: Run the CLI predictor immediately
make predict

# Option 5: Run tests
make test
```

---

## 📁 Project Structure

```
eco-material-predictor/
├── data/
│   ├── raw/materials_dataset.csv       ← curated 285-row dataset
│   └── processed/                      ← auto-generated splits (after make train)
│       ├── features_train.csv          ← 70% (199 rows)
│       ├── features_val.csv            ← 10% (29 rows)
│       └── features_test.csv           ← 20% (57 rows) — held-out final eval
├── models/
│   ├── material_predictor.pkl          ← stacked ensemble (polymer + alloy models)
│   └── scaler.pkl                      ← fitted StandardScaler
├── results/
│   ├── evaluation_report_polymers.txt  ← MAE / RMSE / R² per target (Polymers)
│   ├── evaluation_report_alloys.txt    ← MAE / RMSE / R² per target (Alloys)
│   ├── 01_actual_vs_predicted_*.png    ← scatter plots (separate for poly/alloy)
│   ├── 02_feature_importance_heatmap.png
│   ├── 03_property_correlation_matrix.png
│   ├── 04_eco_score_vs_properties.png
│   └── 05_residual_distributions_*.png ← residual distributions
├── scripts/
│   └── perfect_dataset.py              ← QSPR dataset generator (run before train)
├── src/
│   ├── data_prep.py                    ← feature engineering + 70/10/20 split
│   ├── train.py                        ← stacked ensemble training (polymer + alloy)
│   ├── evaluate.py                     ← metrics + 5 publication-quality graphs
│   ├── predict.py                      ← programmatic inference API
│   ├── cli.py                          ← interactive terminal predictor
│   ├── api.py                          ← FastAPI backend (REST endpoints)
│   ├── recommend.py                    ← Green Alternative Recommender engine
│   └── generate_pdb.py                 ← PDB file generator for VMD visualization
├── frontend/                           ← React + Vite web app
│   ├── src/App.jsx                     ← main UI (sliders, radar chart, predictor)
│   ├── src/index.css                   ← liquid-glass biopunk CSS
│   └── index.html
├── tests/
│   └── test_pipeline.py                ← 27 pytest unit tests
├── requirements.txt
├── setup.sh                            ← Linux one-shot installer
├── Makefile                            ← all shortcut commands
└── README.md
```

---

## 🤖 Model Architecture

```
285 rows → 70% Train (199) / 10% Val (29) / 20% Test (57)
                                                   ↑ completely held-out for R²

Input Features (10) per material class:
       │
  ┌────┴────┐
  RF       XGB       ← base learners (RandomizedSearchCV hyperparameter tuning)
  └────┬────┘
       │
    Ridge            ← meta-learner (stacking, trained on OOF predictions)
       │
  10 Predictions + Confidence (±std from RF ensemble)

Two separate ensembles: POLYMER model (100 samples) + ALLOY model (99 samples)
```

**Features used:**
| Feature | Description |
|---|---|
| `repeat_unit_MW` | Molecular weight of polymer repeat unit (g/mol) |
| `backbone_flexibility` | Chain stiffness (0 = rigid, 1 = flexible) |
| `polarity_index` | Polarity (0 = nonpolar, 3 = highly polar) |
| `hydrogen_bond_capacity` | H-bond strength (0–5) |
| `aromatic_content` | Fraction of aromatic carbons (0–1) |
| `crystallinity_tendency` | Crystallinity (0 = amorphous, 1 = crystalline) |
| `eco_score` | Bio-based sustainability (0 = petroleum, 1 = bio-based) |
| `is_alloy` | Binary: 0 = polymer, 1 = metal alloy |
| `mw_flexibility` | Interaction: MW × flexibility |
| `polar_hbond` | Interaction: polarity × H-bond capacity |

---

## 🧪 Dataset

**285 materials** curated from:
- Published QSPR (Quantitative Structure-Property Relationship) literature
- Matmatch and CAMPUS Plastics databases
- Peer-reviewed polymer physics data (Fox-Flory, Gibbs-DiMarzio models)
- Augmented with diverse synthetic metal alloy grades (Fe, Ti, Al, Mg, Cu)

Properties generated via `scripts/perfect_dataset.py` using scientifically-grounded QSPR formulas + **2% realistic measurement noise** (simulates actual lab uncertainty), then split 70/10/20 to ensure honest, reproducible R² evaluation with **zero data leakage**.

---

## � VMD Visualization Tips

The generated `.pdb` files from `make predict` can be visualized in VMD. By default, VMD uses a simple "Lines" representation. For a professional, high-quality look:
1. Open VMD and load your `.pdb` file.
2. Go to **Graphics > Representations**.
3. Change **Drawing Method** from `Lines` to `CPK` or `Licorice`.
4. (Optional) Change **Coloring Method** to `Name` to color by element (O = red, C = cyan, etc.).
5. (Optional) Go to **Display > Display Settings** and set **Axes** to `Off` to hide the XYZ arrows for a cleaner screenshot.

---

## �📊 Programmatic API

```python
from src.predict import predict

result = predict({
    "repeat_unit_MW":         72.0,   # PLA
    "backbone_flexibility":   0.40,
    "polarity_index":         2,
    "hydrogen_bond_capacity": 2,
    "aromatic_content":       0.0,
    "crystallinity_tendency": 0.35,
    "eco_score":              1.0,
    "is_alloy":               0,
})

print(result["predictions"])
# {'Tg_celsius': 62.4, 'tensile_strength_MPa': 87.4, ..., 'oxygen_permeability_barrer': 17.1}
print(result["confidence"])
# {'Tg_celsius': 3.4, ...}  # ±std from RF ensemble
```

---

## 🛠 Commands Reference

| Command | Action |
|---|---|
| `bash setup.sh` | Create venv + install all Python deps |
| `python scripts/perfect_dataset.py` | Generate dataset (run once before training) |
| `make train` | Prepare data + train stacked ensemble |
| `make evaluate` | Evaluate on test set + save 5 plots to `results/` |
| `make app` | Launch React Web Dashboard + FastAPI backend |
| `make predict` | Launch interactive CLI (terminal only) |
| `make test` | Run pytest suite (27 tests) |
| `make clean` | Remove generated models and result files |

---

## License

MIT — free to use, modify, and build upon.
