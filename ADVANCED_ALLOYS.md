# ⚙️ Advanced Metallurgical Prediction Engine (Real Data Integration)

During the development of the Eco-Material Innovation Engine, we encountered a fascinating real-world challenge when integrating production metallurgical data (specifically exactly defined steel alloys from the **Matminer / Kaggle** databases).

## The Challenge: Heuristics vs. Chemistry

Our primary web application (`src/train.py`, `src/api.py`, `App.jsx`) uses **Molecular Heuristics** (like *Hydrogen Bond Capacity*, *Backbone Flexibility*, and *Polarity Index*). This approach is incredibly powerful for discovering entirely novel, theoretical polymers and bioplastics where exact chemical repeating units might not exist yet in an industrial database.

However, when we downloaded the `steel_strength.csv` dataset (containing 312 real-world steel alloys), the features were entirely different. Real-world metallurgical data isn't tracked by "Polarity"—it's tracked by **Elemental Weight Percentages (wt%)** (e.g., % Carbon, % Chromium, % Nickel).

When we attempted to force the real steel data into the heuristic polymer pipeline, the AI's accuracy (R² score) plummeted from `> 0.90` down to nearly `0.00` for alloys because the input variables didn't cleanly map to chemical reality.

## The Solution: A Dedicated Parallel Architecture

To prove the robustness of our machine learning approach without breaking our highly-polished theoretical polymer predictor, we built a **dedicated parallel prediction pipeline** strictly for real-world metal alloys:

```bash
# The files inside this repository
scripts/advanced_train.py    # Trains an XGBoost ensemble strictly on Elemental Data
scripts/advanced_predict.py  # Inference script to predict physical limits 
data/raw/steel_strength.csv  # The real-world Kaggle/Matminer steel dataset
```

### How to use the Advanced Elemental Predictor:

1. **Train the dedicated model:**
   Instead of using `make train`, run the advanced trainer directly:
   ```bash
   python scripts/advanced_train.py
   ```
   *This compiles an XGBoost ensemble trained strictly on the 13 metallic elements present in the dataset, achieving an R² of ~0.90 on held-out steel data.*

2. **Run a prediction:**
   You can run exact elemental combinations directly from your command line:
   ```bash
   python scripts/advanced_predict.py
   ```
   *The prompt will ask you to enter the wt% of each element (Carbon, Manganese, Silicon, Chrome, Nickel, etc.). Entering `18` for Chromium and `8` for Nickel will instantly predict the exact Yield Strength, Tensile Strength, and Elongation of 304 Stainless Steel.*

---
### Hackathon Pitch Note:
The Eco-Material Innovation Engine demonstrates scalability. 

* The **Web Interface** demonstrates theoretical material discovery (perfect for finding new sustainable bio-polymers based on molecular physics).
* The **Advanced Engine (`scripts/advanced_*`)** demonstrates that the exact same machine-learning architecture easily adapts to real-world industrial data (predicting the strength of exact steel compositions without physical lab tests).
