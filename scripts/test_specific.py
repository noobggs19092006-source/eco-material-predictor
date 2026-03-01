import numpy as np
import pandas as pd
from src.predict import predict

p = {
    "repeat_unit_MW": 229,
    "backbone_flexibility": 0.49,
    "polarity_index": 2.6,
    "aromatic_content": 0.35,
    "crystallinity_tendency": 0.71,
    "eco_score": 0.36,
    "hydrogen_bond_capacity": 0.71 * 3.5,
    "is_alloy": -1.0
}
p["mw_flexibility"] = p["repeat_unit_MW"] * p["backbone_flexibility"]
p["polar_hbond"] = p["polarity_index"] * p["hydrogen_bond_capacity"]

print(predict(p)["predictions"])
