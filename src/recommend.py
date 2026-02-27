"""
Green Alternative Recommender
---------------------------------
Given a petroleum-based (eco_score < 0.6) material from the dataset,
find the top N bio-based alternatives that match its physical performance.

Usage:
    from src.recommend import find_green_alternatives
    results = find_green_alternatives("ABS (conventional)")
"""
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")

# Properties used for similarity scoring (normalised Euclidean distance)
MATCH_KEYS = [
    "tensile_strength_MPa",
    "youngs_modulus_GPa",
    "Tg_celsius",
    "density_gcm3",
    "elongation_at_break_pct",
]


def _load() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def list_petroleum_materials() -> list[str]:
    """Return the names of all petroleum-based polymers in the dataset."""
    df = _load()
    dirty = df[(df["eco_score"] < 0.6) & (df["material_class"] == "polymer")]
    return sorted(dirty["material_name"].tolist())


def find_green_alternatives(
    material_name: str,
    top_n: int = 3,
    eco_threshold: float = 0.7,
) -> dict:
    """
    Find the top_n most performance-similar bio-based alternatives
    for a given petroleum material.

    Returns
    -------
    dict with keys:
        'target'       : dict of the dirty material's properties
        'alternatives' : list of dicts (ranked, best first)
        'error'        : str | None
    """
    df = _load()

    # --- locate the target ---------------------------------------------------
    matches = df[df["material_name"].str.lower() == material_name.strip().lower()]
    if len(matches) == 0:
        # fuzzy fallback: partial name match
        matches = df[df["material_name"].str.lower().str.contains(material_name.strip().lower(), na=False)]
    if len(matches) == 0:
        return {"target": None, "alternatives": [], "error": f"Material '{material_name}' not found."}

    target_row = matches.iloc[0]

    # --- candidate pool: bio-based plastics only ----------------------------
    # We only recommend polymer alternatives (alloy vs alloy replacement
    # makes less sense for eco-scoring which focuses on bio-plastics)
    candidates = df[
        (df["eco_score"] >= eco_threshold) &
        (df["material_class"] == "polymer") &
        (df["material_name"] != target_row["material_name"])
    ].copy()

    if candidates.empty:
        return {"target": target_row.to_dict(), "alternatives": [], "error": "No green candidates found."}

    # --- normalise and compute Euclidean distance ---------------------------
    for key in MATCH_KEYS:
        max_val = df[key].max()
        min_val = df[key].min()
        rng = max_val - min_val if (max_val - min_val) != 0 else 1.0
        candidates[f"_norm_{key}"] = (candidates[key] - min_val) / rng
        target_norm = (target_row[key] - min_val) / rng
        candidates[f"_dist_{key}"] = (candidates[f"_norm_{key}"] - target_norm) ** 2

    dist_cols = [f"_dist_{k}" for k in MATCH_KEYS]
    candidates["_total_dist"] = np.sqrt(candidates[dist_cols].sum(axis=1))

    top = candidates.nsmallest(top_n, "_total_dist")

    result_cols = [
        "material_name", "eco_score",
        "tensile_strength_MPa", "youngs_modulus_GPa",
        "Tg_celsius", "density_gcm3", "elongation_at_break_pct",
        "_total_dist",
    ]
    top_records = top[result_cols].rename(columns={"_total_dist": "similarity_distance"}).to_dict(orient="records")

    # compute % similarity (0 = perfect clone, higher = more different)
    max_possible_dist = np.sqrt(len(MATCH_KEYS))   # all normalised, so max per axis = 1
    for rec in top_records:
        similarity_pct = max(0, 100 * (1 - rec["similarity_distance"] / max_possible_dist))
        rec["performance_match_pct"] = round(similarity_pct, 1)

    target_info = {k: target_row[k] for k in ["material_name", "eco_score"] + MATCH_KEYS}

    return {
        "target": target_info,
        "alternatives": top_records,
        "error": None,
    }
