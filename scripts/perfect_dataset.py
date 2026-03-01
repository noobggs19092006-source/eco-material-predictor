import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")
rng = np.random.RandomState(42)

def add_noise(val, pct=0.04):
    """Add realistic measurement noise (approx ~5 - 7%)."""
    actual_pct = pct + rng.uniform(0.01, 0.03)
    noise = rng.normal(0, max(abs(val) * actual_pct, 0.02))
    return val + noise

print("[*] Generating 4000+ Synthetic Polymers and 4000+ Synthetic Generic Metals...")
records = []

COMMON_POLYMERS = [
    "ABS", "PET", "LDPE", "HDPE", "PP", "PVC", "PS", "PC", "PMMA", "PA6",
    "PA66", "POM", "PTFE", "PEEK", "PI", "PBT", "PLA", "PHA", "PU", "EVA",
    "Epoxy", "Phenolic", "Polyester", "Silicone", "PEI", "PPS", "PSU", "PES", "LCP", "PFA"
]

COMMON_METALS = [
    "Steel 1018", "Steel 4140", "Steel 4340", "Stainless Steel 304", "Stainless Steel 316",
    "Stainless Steel 410", "Aluminum 1100", "Aluminum 2024", "Aluminum 5052", "Aluminum 6061",
    "Aluminum 7075", "Titanium Grade 2", "Titanium Ti-6Al-4V", "Brass (C36000)", "Brass (C26000)",
    "Bronze (C52100)", "Bronze (C93200)", "Copper (C10100)", "Copper (C11000)", "Beryllium Copper (C17200)",
    "Magnesium AZ31B", "Magnesium AZ91D", "Zinc Alloy (Zamak 3)", "Nickel Alloy 200", "Monel 400",
    "Inconel 600", "Inconel 625", "Inconel 718", "Stellite 6", "Tungsten"
]

# --- 1. GENERATE 4000 POLYMERS ---
for i in range(4000):
    mw = rng.uniform(20.0, 300.0)
    flex = rng.uniform(0.0, 1.0)
    polar = rng.randint(0, 4)
    hbond = rng.randint(0, 6)
    aro = rng.uniform(0.0, 1.0)
    cryst = rng.uniform(0.0, 1.0)
    eco = rng.uniform(0.1, 1.0)
    
    # Target Formulas
    tg = 40 + (aro * 150) + (hbond * 10) + (polar * 15) - (flex * 60)
    tens = 30 + (aro * 80) + (cryst * 60) + (hbond * 15) + (polar * 10) - (flex * 30)
    young = 0.3 + (aro * 4.0) + (cryst * 3.0) + (hbond * 0.8) + (polar * 0.4) - (flex * 1.5)
    dens = 0.80 + 0.35*(polar/3) + 0.30*aro + 0.25*cryst + 0.15*flex + 0.10*(hbond/5) + min(0.10, mw/3000)
    therm = 0.5 + 2.0*cryst + 1.5*aro - 0.2*hbond
    elec = -14.0 + 6.0*aro + 1.5*(polar/3) - 0.8*flex + 0.5*hbond
    elong = 10.0 + 60*flex - 30*cryst - 40*aro - 5*hbond + 8*(polar/3)
    diel = 2.2 + 0.8*polar + 0.3*hbond + 0.5*aro
    water = 0.02 + 0.8*(hbond/5) + 0.5*(polar/3) - 0.5*cryst - 0.3*aro
    o2 = 15.0 + 50.0*flex - 30.0*cryst - 10.0*(polar/3) - 10.0*aro
    
    records.append({
        "material_name": COMMON_POLYMERS[i % len(COMMON_POLYMERS)],
        "material_class": "polymer",
        "repeat_unit_MW": round(mw, 2),
        "backbone_flexibility": round(flex, 3),
        "polarity_index": polar,
        "hydrogen_bond_capacity": hbond,
        "aromatic_content": round(aro, 3),
        "crystallinity_tendency": round(cryst, 3),
        "eco_score": round(eco, 3),
        
        "Tg_celsius": round(add_noise(tg), 1),
        "tensile_strength_MPa": round(add_noise(tens), 1),
        "youngs_modulus_GPa": max(0.01, round(add_noise(young), 2)),
        "density_gcm3": max(0.80, round(add_noise(dens), 3)),
        "thermal_conductivity_WmK": max(0.10, round(add_noise(therm), 3)),
        "log10_elec_conductivity": round(np.clip(add_noise(elec), -16.0, 2.0), 2),
        "elongation_at_break_pct": max(1.0, round(add_noise(elong), 1)),
        "dielectric_constant": round(np.clip(add_noise(diel), 1.9, 12.0), 2),
        "water_absorption_pct": max(0.001, round(add_noise(water), 3)),
        "oxygen_permeability_barrer": max(0.001, round(add_noise(o2), 3))
    })

# --- 2. GENERATE 4000 GENERIC PHYSICAL METALS ---
for i in range(4000):
    r_diff = rng.uniform(0.0, 15.0)  # Atomic Radius Difference (%)
    enthalpy = rng.uniform(-40.0, 10.0) # Mixing Enthalpy (kJ/mol)
    valence = rng.uniform(3.0, 12.0)  # Valence electrons
    en_diff = rng.uniform(0.0, 0.5)   # Electronegativity Difference
    shear = rng.uniform(20.0, 100.0)  # Shear Modulus (GPa)
    melt = rng.uniform(600.0, 3000.0) # Melting Temp (C)
    eco = rng.uniform(0.0, 0.8)       # Realistically worse than bio-plastics
    
    # Target Formulas mapping these 7 inputs directly to the 10 outputs symmetrically
    tg = melt * 0.85 # Melting point acts as functional Tg proxy
    tens = shear * 10.0 + abs(enthalpy) * 5.0 + valence * 20.0
    young = shear * 2.5 + valence * 1.5
    dens = 2.0 + valence * 0.4 + (melt / 600.0)
    therm = 200.0 - (r_diff * 10.0) - (en_diff * 100.0)
    elec = 7.0 - (r_diff * 0.2) - (en_diff * 2.0)
    elong = 40.0 - (shear * 0.2) - abs(enthalpy) * 0.5
    diel = 1.5 + (en_diff * 5.0)  # Variance based on electronegativity
    water = 0.001 + (r_diff * 0.02) # Micro-porosity variance
    o2 = 0.001 + (r_diff * 0.05)    # Micro-porosity variance
    
    records.append({
        "material_name": COMMON_METALS[i % len(COMMON_METALS)],
        "material_class": "metal",
        "atomic_radius_difference": round(r_diff, 2),
        "mixing_enthalpy": round(enthalpy, 2),
        "valence_electrons": round(valence, 2),
        "electronegativity_diff": round(en_diff, 3),
        "shear_modulus": round(shear, 2),
        "melting_temp": round(melt, 1),
        "eco_score": round(eco, 3),
        
        "Tg_celsius": round(add_noise(tg), 1),
        "tensile_strength_MPa": round(add_noise(tens), 1),
        "youngs_modulus_GPa": max(1.0, round(add_noise(young), 2)),
        "density_gcm3": max(1.0, round(add_noise(dens), 3)),
        "thermal_conductivity_WmK": max(5.0, round(add_noise(therm), 2)),
        "log10_elec_conductivity": round(np.clip(add_noise(elec), 0.0, 10.0), 2),
        "elongation_at_break_pct": max(0.5, round(add_noise(elong), 1)),
        "dielectric_constant": round(np.clip(add_noise(diel, pct=0.03), 1.0, 5.0), 3),
        "water_absorption_pct": max(0.000, round(add_noise(water, pct=0.05), 4)),
        "oxygen_permeability_barrer": max(0.000, round(add_noise(o2, pct=0.05), 4))
    })

df = pd.DataFrame(records)
df.to_csv(RAW_CSV, index=False)

print(f"✅ Generated dual-pipeline QSPR dataset ({len(df)} total rows).")
print("   - 4000 Unified Polymer Profiles (7 features)")
print("   - 4000 Universal Metal Profiles (7 features)")
