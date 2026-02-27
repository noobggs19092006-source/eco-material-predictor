"""
⚠️  DEPRECATED — DO NOT RUN THIS SCRIPT DIRECTLY ⚠️

This script (expand_dataset.py) is superseded by:
    python scripts/perfect_dataset.py

Reasons this script is retired:
  1. The elongation formula (line 124) uses row["Tg_celsius"] which is also
     a TARGET column — this is a data-leakage bug.
  2. The Makefile pipeline now calls perfect_dataset.py, not this file.

expand_dataset.py — Adds 7 new property columns to materials_dataset.csv.
New columns (all QSPR-grounded, physically realistic):
  density_gcm3, thermal_conductivity_WmK, log10_elec_conductivity,
  elongation_at_break_pct, dielectric_constant,
  water_absorption_pct, oxygen_permeability_barrer

Run from project root:
    source venv/bin/activate
    python scripts/expand_dataset.py
"""

import os, sys
import numpy as np
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")

rng = np.random.RandomState(42)

# ── Load ∓ inspect ─────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

# Known check columns
for col in ["Tg_celsius", "tensile_strength_MPa", "youngs_modulus_GPa"]:
    assert col in df.columns, f"Missing column: {col}"

is_alloy = df["material_class"] == "alloy"
p        = df  # shorthand


# ─────────────────────────────────────────────────────────────────────────────
# 1. DENSITY  (g/cm³)
# Polymers: 0.85 – 1.80 based on polarity, MW, crystallinity, aromatic content
# Alloys:   1.7 (Mg) – 8.5 (steel)
# ─────────────────────────────────────────────────────────────────────────────
def density(row):
    if row["material_class"] == "alloy":
        # Estimate from Young's modulus (stiff metals tend to be denser)
        E = row["youngs_modulus_GPa"]
        if E > 300:    return round(rng.uniform(14.0, 15.5), 2)   # cemented carbide
        elif E > 150:  return round(rng.uniform(7.5, 8.0), 2)     # steel / Fe-based
        elif E > 90:   return round(rng.uniform(4.3, 4.7), 2)     # Ti alloys
        elif E > 60:   return round(rng.uniform(2.6, 2.85), 2)    # Al alloys
        else:          return round(rng.uniform(1.75, 2.0), 2)    # Mg alloys
    else:
        base  = 1.05
        base += 0.12 * row["polarity_index"] / 3
        base += 0.10 * row["aromatic_content"]
        base += 0.08 * row["crystallinity_tendency"]
        base += min(0.30, row["repeat_unit_MW"] / 1500)
        noise = rng.normal(0, 0.04)
        return round(np.clip(base + noise, 0.85, 1.80), 3)

df["density_gcm3"] = df.apply(density, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. THERMAL CONDUCTIVITY  (W/m·K)
# Polymers: 0.10 – 0.55 (crystalline & aromatic → higher)
# Alloys:   7 – 400
# ─────────────────────────────────────────────────────────────────────────────
def thermal_cond(row):
    if row["material_class"] == "alloy":
        E = row["youngs_modulus_GPa"]
        if E > 300:   return round(rng.uniform(50, 120), 1)    # cemented carbide
        elif E > 150: return round(rng.uniform(10, 55), 1)     # steel/Ti (~16-45)
        elif E > 90:  return round(rng.uniform(7, 25), 1)      # Ti alloys (~7-17)
        elif E > 60:  return round(rng.uniform(130, 220), 1)   # Al alloys (~130-220)
        else:         return round(rng.uniform(70, 120), 1)    # Mg alloys (~75-110)
    else:
        base  = 0.15
        base += 0.15 * row["crystallinity_tendency"]
        base += 0.10 * row["aromatic_content"]
        base -= 0.02 * row["hydrogen_bond_capacity"]  # H-bonded polymers slightly lower
        noise = rng.normal(0, 0.02)
        return round(np.clip(base + noise, 0.10, 0.60), 3)

df["thermal_conductivity_WmK"] = df.apply(thermal_cond, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOG10 ELECTRICAL CONDUCTIVITY  (log₁₀ S/m)
# Most polymers: –16 to –10 (insulators)
# Conductive polymers (aromatic, conjugated): –5 to +2
# Metals: +5 to +8
# ─────────────────────────────────────────────────────────────────────────────
def log10_conductivity(row):
    if row["material_class"] == "alloy":
        # Metallic glasses slightly lower than crystalline metals
        if row["crystallinity_tendency"] < 0.1:
            return round(rng.uniform(2.5, 4.5), 2)   # metallic glass
        else:
            return round(rng.uniform(5.5, 7.8), 2)   # typical metal
    else:
        # More aromatic + less polarity → more conjugation → higher conductivity
        base  = -14.0
        base += 4.0 * row["aromatic_content"]         # conjugation lift
        base += 1.0 * (row["polarity_index"] / 3)     # slight ionic contribution
        base -= 0.5 * row["backbone_flexibility"]     # rigid = more conjugated
        noise = rng.normal(0, 0.5)
        return round(np.clip(base + noise, -16.0, 2.0), 2)

df["log10_elec_conductivity"] = df.apply(log10_conductivity, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ELONGATION AT BREAK  (%)
# Rigid/glassy: 2–10 %     Semi-crystalline: 10–200 %
# Elastomers: 200–800 %    Metals: 2–60 %
# ─────────────────────────────────────────────────────────────────────────────
def elongation(row):
    if row["material_class"] == "alloy":
        E = row["youngs_modulus_GPa"]
        if E > 300:  return round(rng.uniform(0.5, 2.0), 1)    # cemented carbide (brittle)
        elif E > 180: return round(rng.uniform(2, 10), 1)      # stiff steel
        elif E > 60:  return round(rng.uniform(5, 25), 1)      # Al/Fe alloys
        else:         return round(rng.uniform(8, 40), 1)      # Mg alloys
    else:
        # High flexibility + low Tg → better elongation
        # High crystallinity or high aromatic → brittle
        # FIX: multiplier reduced from 200→0 to 80 to avoid >100% for moderate polymers
        base  = 5.0
        base += 80 * row["backbone_flexibility"] * (1 - row["crystallinity_tendency"])
        base -= 0.15 * max(row["Tg_celsius"], 0)           # high Tg = more brittle
        base -= 50 * row["aromatic_content"]               # aromatic = rigid
        noise = rng.normal(0, 5)
        return round(np.clip(base + noise, 1.0, 900.0), 1)

df["elongation_at_break_pct"] = df.apply(elongation, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. DIELECTRIC CONSTANT (relative permittivity, dimensionless, at 1 kHz)
# Non-polar polymers: 2.0–2.5   |   Polar: 3–8   |   Proteins: 5–10
# Metals/conductors: effectively ∞ → encode as –1 (means "conductor, not applicable")
# ─────────────────────────────────────────────────────────────────────────────
def dielectric(row):
    if row["material_class"] == "alloy":
        if row["crystallinity_tendency"] < 0.1:
            return round(rng.uniform(5.0, 15.0), 2)   # metallic glass (semi-conductor-like)
        return -1.0   # highly conductive → not meaningful
    else:
        base  = 2.2
        base += 0.8 * row["polarity_index"]
        base += 0.3 * row["hydrogen_bond_capacity"]
        base += 0.5 * row["aromatic_content"]
        noise = rng.normal(0, 0.2)
        return round(np.clip(base + noise, 1.9, 12.0), 2)

df["dielectric_constant"] = df.apply(dielectric, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 6. WATER ABSORPTION  (% by weight, 24 h immersion)  — WATER MEDIUM
# Hydrophobic: 0.01–0.1 %   |   Moderate: 0.5–3 %   |   Hydrophilic: 3–30 %
# Metals: 0 %
# ─────────────────────────────────────────────────────────────────────────────
def water_absorption(row):
    if row["material_class"] == "alloy":
        return 0.0
    else:
        # FIX: polarity and H-bond weights reduced 3×; eco multiplier removed
        # (bio-based ≠ necessarily more water-absorbent)
        base  = 0.02
        base += 0.8 * (row["hydrogen_bond_capacity"] / 5)   # was 2.5
        base += 0.5 * (row["polarity_index"] / 3)           # was 1.5
        base -= 0.5 * row["crystallinity_tendency"]         # crystallinity blocks water
        base -= 0.3 * row["aromatic_content"]               # hydrophobic aromatic rings
        noise = rng.normal(0, 0.08)
        return round(np.clip(base + noise, 0.001, 35.0), 3)

df["water_absorption_pct"] = df.apply(water_absorption, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 7. OXYGEN PERMEABILITY  (Barrers = 10⁻¹⁰ cm³·cm/cm²·s·cmHg) — AIR MEDIUM
# Crystalline/dense: 0.01–2 Barrers (good barrier)
# Flexible rubbery:  50–500 Barrers (poor barrier)
# Aromatic/polar:    0.01–5 Barrers
# Metals:            ~0 (impermeable)
# ─────────────────────────────────────────────────────────────────────────────
def o2_permeability(row):
    if row["material_class"] == "alloy":
        return 0.0
    else:
        # FIX: baseline reduced from 1.5 to 0.5 (10^0.5=3.2 Barrers, not 31.6)
        # PLA (flex=0.4, cryst=0.35, polar=2): log= 0.5+0.8-0.525-0.333 = 0.44 → 2.8 Barr OK
        log_perm  = 0.5   # was 1.5
        log_perm += 2.0 * row["backbone_flexibility"]
        log_perm -= 1.5 * row["crystallinity_tendency"]
        log_perm -= 0.5 * (row["polarity_index"] / 3)
        log_perm -= 0.5 * row["aromatic_content"]
        noise = rng.normal(0, 0.25)
        return round(np.clip(10 ** (log_perm + noise), 0.001, 500.0), 3)

df["oxygen_permeability_barrer"] = df.apply(o2_permeability, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
print("\nNew columns added:")
new_cols = ["density_gcm3", "thermal_conductivity_WmK", "log10_elec_conductivity",
            "elongation_at_break_pct", "dielectric_constant",
            "water_absorption_pct", "oxygen_permeability_barrer"]
for c in new_cols:
    print(f"  {c:<38} range: {df[c].min():.3g} – {df[c].max():.3g}")

df.to_csv(RAW_CSV, index=False)
print(f"\n✅  Saved expanded dataset → {RAW_CSV}")
print(f"   Shape: {df.shape}")
