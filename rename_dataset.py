import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/materials_dataset.csv")

dirty_polymers = ["ABS", "HDPE", "LDPE", "PET", "PVC", "PS", "PP", "PC", "PMMA", "PA (Nylon)", "PTFE (Teflon)", "PEEK", "PU (Polyurethane)", "Epoxy Resin", "Phenolic Resin"]
clean_polymers = ["PLA", "PHA", "PBAT", "PCL", "PBS", "Bio-PE", "Bio-PET", "Cellulose Acetate", "Starch Blend", "Chitosan", "Bio-Polycarbonate", "Alginate", "Thermoplastic Starch"]

dirty_metals = ["Steel (Carbon)", "Steel (Stainless)", "Aluminum 6061", "Aluminum 7075", "Titanium Grade 2", "Titanium Grade 5", "Brass C360", "Bronze", "Copper C110", "Cast Iron", "Inconel 718", "Magnesium AZ31", "Lead Alloy", "Zinc Die-Cast", "Cobalt Chrome"]
clean_metals = ["Recycled Steel", "Scrap-based Aluminum", "High Entropy Alloy (CoCrFeNi)", "Bio-compatible Titanium", "Amorphous Metal", "Green Brass", "Secondary Copper", "Recycled Magnesium", "Eco-Nickel", "Recycled Zinc", "Bio-degradable Iron", "Scrap-based Cobalt", "Recycled Bronze"]

np.random.seed(42)

for i, row in df.iterrows():
    if row["material_class"] == "polymer":
        if row["eco_score"] < 0.6:
            df.at[i, "material_name"] = np.random.choice(dirty_polymers)
        else:
            df.at[i, "material_name"] = np.random.choice(clean_polymers)
    else:
        if row["eco_score"] < 0.6:
            df.at[i, "material_name"] = np.random.choice(dirty_metals)
        else:
            df.at[i, "material_name"] = np.random.choice(clean_metals)

# Optional: Add a small numerical ID to completely duplicate rows if we want to ensure uniqueness (or just leave them as non-unique. The API will just grab the first match). 
# Actually, the user asked for them NOT to be written like polymer-0. So we leave them as just the base names!

df.to_csv("data/raw/materials_dataset.csv", index=False)
print("Renamed materials in the dataset.")
