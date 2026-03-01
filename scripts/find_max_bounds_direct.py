import numpy as np
import pandas as pd
from src.predict import predict

def get_random_polymer():
    return {
        "repeat_unit_MW": np.random.uniform(10, 600),
        "backbone_flexibility": np.random.uniform(0, 1),
        "polarity_index": np.random.uniform(0, 3),
        "hydrogen_bond_capacity": np.random.uniform(0, 5),
        "aromatic_content": np.random.uniform(0, 1),
        "crystallinity_tendency": np.random.uniform(0, 1),
        "eco_score": np.random.uniform(0, 1),
        "is_alloy": -1.0
    }

def get_random_alloy():
    hbond = np.random.uniform(0, 1) * 3.5
    return {
        "repeat_unit_MW": np.random.uniform(10, 300),
        "backbone_flexibility": np.random.uniform(0, 1),
        "polarity_index": np.random.uniform(0, 3),
        "hydrogen_bond_capacity": hbond,
        "aromatic_content": np.random.uniform(0, 1), 
        "crystallinity_tendency": hbond / 3.5,
        "eco_score": np.random.uniform(0, 1),
        "is_alloy": -1.0
    }

def get_random_metal():
    return {
        "atomic_radius_difference": np.random.uniform(0, 15),
        "mixing_enthalpy": np.random.uniform(-50, 20),
        "valence_electrons": np.random.uniform(3, 12),
        "electronegativity_diff": np.random.uniform(0, 0.6),
        "shear_modulus": np.random.uniform(10, 150),
        "melting_temp": np.random.uniform(400, 3500),
        "eco_score": np.random.uniform(0, 1),
        "is_alloy": 1.0,
        "mw_flexibility": 0,
        "polar_hbond": 0,
        "repeat_unit_MW": 0,
        "backbone_flexibility": 0,
        "polarity_index": 0,
        "hydrogen_bond_capacity": 0,
        "aromatic_content": 0,
        "crystallinity_tendency": 0
    }

def probe():
    maxes = {
        "polymer": {"tensile_strength_MPa": 0, "Tg_celsius": 0, "youngs_modulus_GPa": 0, "density_gcm3": 0, "thermal_conductivity_WmK": 0, "elongation_at_break_pct": 0},
        "alloy": {"tensile_strength_MPa": 0, "Tg_celsius": 0, "youngs_modulus_GPa": 0, "density_gcm3": 0, "thermal_conductivity_WmK": 0, "elongation_at_break_pct": 0},
        "metal": {"tensile_strength_MPa": 0, "Tg_celsius": 0, "youngs_modulus_GPa": 0, "density_gcm3": 0, "thermal_conductivity_WmK": 0, "elongation_at_break_pct": 0}
    }
    
    for i in range(15):
        # Polymer
        p = get_random_polymer()
        p["mw_flexibility"] = p["repeat_unit_MW"] * p["backbone_flexibility"]
        p["polar_hbond"] = p["polarity_index"] * p["hydrogen_bond_capacity"]
        res = predict(p)["predictions"]
        for k in maxes["polymer"]:
            if k in res:
                maxes["polymer"][k] = max(maxes["polymer"][k], float(res[k]))
                
        # Alloy
        a = get_random_alloy()
        a["mw_flexibility"] = a["repeat_unit_MW"] * a["backbone_flexibility"]
        a["polar_hbond"] = a["polarity_index"] * a["hydrogen_bond_capacity"]
        res = predict(a)["predictions"]
        for k in maxes["alloy"]:
            if k in res:
                maxes["alloy"][k] = max(maxes["alloy"][k], float(res[k]))
                
        # Metal
        m = get_random_metal()
        res = predict(m)["predictions"]
        for k in maxes["metal"]:
            if k in res:
                maxes["metal"][k] = max(maxes["metal"][k], float(res[k]))
                
    print("MAX OBSERVED VALUES ACROSS 15 PERMUTATIONS:")
    import pprint
    pprint.pprint(maxes)

if __name__ == "__main__":
    probe()
