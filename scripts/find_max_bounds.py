import requests
import numpy as np

URL = "http://localhost:8000/predict"

def get_random_polymer():
    return {
        "inputs": {
            "repeat_unit_MW": np.random.uniform(10, 600),
            "backbone_flexibility": np.random.uniform(0, 1),
            "polarity_index": np.random.uniform(0, 3),
            "hydrogen_bond_capacity": np.random.uniform(0, 5),
            "aromatic_content": np.random.uniform(0, 1),
            "crystallinity_tendency": np.random.uniform(0, 1),
            "eco_score": np.random.uniform(0, 1),
        },
        "mode": "polymer"
    }

def get_random_alloy():
    return {
        "inputs": {
            "repeat_unit_MW": np.random.uniform(10, 300),
            "backbone_flexibility": np.random.uniform(0, 1),
            "polarity_index": np.random.uniform(0, 3),
            "aromatic_content": np.random.uniform(0, 1), # alloying content
            "crystallinity_tendency": np.random.uniform(0, 1),
            "eco_score": np.random.uniform(0, 1),
        },
        "mode": "alloy"
    }

def get_random_metal():
    return {
        "inputs": {
            "atomic_radius_difference": np.random.uniform(0, 15),
            "mixing_enthalpy": np.random.uniform(-50, 20),
            "valence_electrons": np.random.uniform(3, 12),
            "electronegativity_diff": np.random.uniform(0, 0.6),
            "shear_modulus": np.random.uniform(10, 150),
            "melting_temp": np.random.uniform(400, 3500),
            "eco_score": np.random.uniform(0, 1),
        },
        "mode": "metal"
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
        p["inputs"]["mw_flexibility"] = p["inputs"]["repeat_unit_MW"] * p["inputs"]["backbone_flexibility"]
        p["inputs"]["polar_hbond"] = p["inputs"]["polarity_index"] * p["inputs"]["hydrogen_bond_capacity"]
        p["inputs"]["is_alloy"] = -1
        res = requests.post(URL, json=p).json()
        for k in maxes["polymer"]:
            if k in res["predictions"]:
                maxes["polymer"][k] = max(maxes["polymer"][k], res["predictions"][k])
                
        # Alloy
        a = get_random_alloy()
        hbond = a["inputs"]["crystallinity_tendency"] * 3.5
        a["inputs"]["mw_flexibility"] = a["inputs"]["repeat_unit_MW"] * a["inputs"]["backbone_flexibility"]
        a["inputs"]["polar_hbond"] = a["inputs"]["polarity_index"] * hbond
        a["inputs"]["hydrogen_bond_capacity"] = hbond
        a["inputs"]["is_alloy"] = -1
        res = requests.post(URL, json=a).json()
        for k in maxes["alloy"]:
            if k in res["predictions"]:
                maxes["alloy"][k] = max(maxes["alloy"][k], res["predictions"][k])
                
        # Metal
        m = get_random_metal()
        m["inputs"]["is_alloy"] = 1
        res = requests.post(URL, json=m).json()
        for k in maxes["metal"]:
            if k in res["predictions"]:
                maxes["metal"][k] = max(maxes["metal"][k], res["predictions"][k])
                
    print("MAX OBSERVED VALUES ACROSS 150 PERMUTATIONS:")
    import pprint
    pprint.pprint(maxes)

if __name__ == "__main__":
    probe()
