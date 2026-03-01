import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/materials_dataset.csv")

dirty_polymers = [
    "ABS (Acrylonitrile Butadiene Styrene)", "HDPE (High-Density Polyethylene)",
    "LDPE (Low-Density Polyethylene)", "PET (Polyethylene Terephthalate)",
    "PVC (Polyvinyl Chloride)", "PS (Polystyrene)", "PP (Polypropylene)",
    "PC (Polycarbonate)", "PMMA (Acrylic)", "PA6 (Nylon 6)", "PA66 (Nylon 66)",
    "PTFE (Teflon)", "PEEK (Polyether Ether Ketone)", "PU (Polyurethane)",
    "Epoxy Resin", "Phenolic Resin (Bakelite)", "Melamine Formaldehyde",
    "Urea Formaldehyde", "Vinyl Acetate", "Polyimide (Kapton)",
    "PPS (Polyphenylene Sulfide)", "PEI (Polyetherimide)", "PES (Polyethersulfone)",
    "POM (Polyoxymethylene / Delrin)", "PPO (Polyphenylene Oxide)", "SAN (Styrene Acrylonitrile)",
    "ASA (Acrylic Styrene Acrylonitrile)", "PBT (Polybutylene Terephthalate)", "PETG (Glycol-modified PET)",
    "CPE (Chlorinated Polyethylene)", "PC-ABS Blend", "SBR (Styrene-Butadiene Rubber)",
    "NBR (Nitrile Rubber)", "EPDM Rubber", "FKM (Viton)", "Silicone Rubber",
    "Neoprene", "Butyl Rubber", "Polyisoprene", "Polybutadiene",
    "TPE (Thermoplastic Elastomer)", "TPU (Thermoplastic Polyurethane)",
    "EVA (Ethylene Vinyl Acetate)", "Polyaryletherketone (PAEK)",
    "Polysulfone (PSU)", "PVDF (Polyvinylidene Fluoride)", "ECTFE (Halar)",
    "FEP (Fluorinated Ethylene Propylene)", "PFA (Perfluoroalkoxy)",
    "PCTFE (Polychlorotrifluoroethylene)", "Polyacrylamide", "Polyacrylic Acid",
    "Polyvinyl Alcohol (PVA)", "Polyvinyl Butyral (PVB)"
] # 54

clean_polymers = [
    "PLA (Polylactic Acid)", "PHA (Polyhydroxyalkanoate)", "PCL (Polycaprolactone)",
    "PBS (Polybutylene Succinate)", "Bio-PE (Bio-Polyethylene)", "Bio-PET",
    "Cellulose Acetate", "Thermoplastic Starch (TPS)", "Chitosan-based Plastic",
    "Bio-Polycarbonate", "Alginate-based Film", "Bio-Polyamide (Nylon 11)",
    "Hemp-infused PLA", "Soy-based Polyurethane", "Lignin-based Polymer",
    "Mycelium Composite", "PHB (Polyhydroxybutyrate)", "PGA (Polyglycolide)",
    "PTT-bio (Polytrimethylene Terephthalate)", "PLA/PHA Blend",
    "PLA/PBS Blend", "Cellulose Nanocrystal Composite", "Starch-grafted Polymer",
    "PHV (Polyhydroxyvalerate)", "PHB-PHV Copolymer", "Soy Protein Isolate Plastic",
    "Zein-based Plastic (Corn)", "Gluten-based Plastic", "Casein-based Plastic",
    "Whey Protein Plastic", "Gelatin-based Film", "Seaweed-based Bioplastic",
    "Agar-based Bioplastic", "Carrageenan Film", "Pectin-based Composite",
    "Bacterial Cellulose Film", "Vegetable Oil-based Epoxy", "Castor Oil-based PU",
    "Bio-based PP", "PLA-PCL Blend", "Cellulose Propionate", "Cellulose Acetate Butyrate",
    "Bio-based Epoxidized Soybean Oil", "Polyglycerol Sebacate (PGS)",
    "Poly(butylene adipate-co-terephthalate) (PBAT)", "Polyhydroxyhexanoate (PHHx)"
] # 46

dirty_metals = [
    "Steel 1018 (Carbon)", "Steel 1045 (Medium Carbon)", "Steel 4140 (Alloy)",
    "Steel 4340 (High-Speed)", "Steel O1 Tooling", "Steel D2 Tooling",
    "Steel A2 Tooling", "Steel H13 Tooling", "Steel W1 Tooling",
    "Steel 304L (Stainless)", "Steel 316L (Stainless)", "Steel 410 (Stainless)",
    "Steel 420 (Stainless)", "Steel 430 (Stainless)", "Steel 440C (Stainless)",
    "Steel 17-4 PH", "Galvanized Steel", "Cast Iron (Gray)", "Cast Iron (Ductile)",
    "Cast Iron (Malleable)", "Cast Iron (White)", "Aluminum 1050", "Aluminum 1100",
    "Aluminum 2014", "Aluminum 2024", "Aluminum 3003", "Aluminum 5052",
    "Aluminum 5083", "Aluminum 6061", "Aluminum 6063", "Aluminum 7050",
    "Aluminum 7075", "Aluminum 356.0 Cast", "Aluminum 380.0 Cast",
    "Copper C101", "Copper C110", "Brass C260", "Brass C360", "Brass C464",
    "Bronze C510", "Bronze C521", "Bronze C544", "Aluminum Bronze C63000",
    "Silicon Bronze C65500", "Beryllium Copper C17200", "Cupronickel 70/30",
    "Titanium Grade 1", "Titanium Grade 2", "Titanium Grade 5 (Ti-6Al-4V)",
    "Titanium Grade 9", "Nickel 200", "Nickel 201", "Monel 400", "Monel K-500",
    "Inconel 600", "Inconel 625", "Inconel 718", "Invar 36", "Hastelloy C276",
    "Magnesium AZ31B", "Magnesium AZ91D", "Zinc Zamak 3", "Zinc Zamak 5",
    "Tungsten Carbide", "Cobalt Chrome (Stellite 6)", "Lead-Antimony Alloy",
    "Tin-Lead Solder", "Babbitt Metal", "Zirconium Alloy"
] # 69

clean_metals = [
    "Recycled Steel", "Scrap-based Aluminum 6061", "High Entropy Alloy (CoCrFeNi)",
    "Bio-compatible Titanium G5", "Amorphous Metal (Bulk Metallic Glass)",
    "Green Brass (Lead-Free)", "Secondary Copper C110", "Recycled Magnesium AZ31",
    "Eco-Nickel 200", "Recycled Zinc Zamak", "Bio-degradable Iron",
    "Scrap-based Cobalt", "Recycled Bronze", "Low-Energy Aluminum",
    "Scrap-Stainless Steel 316L", "Green Titanium Alloy", "Upcycled Monel 400",
    "Advanced High-Strength Steel (AHSS) - Recycled", "Recycled AL 7075",
    "Scrap Cast Iron", "Bio-compatible Co-Cr", "Lead-Free Solder (Sn-Ag-Cu)",
    "Eco-Alloy 3106", "Eco-Alloy 880", "High Entropy Alloy (AlCoCrFeNi)",
    "High Entropy Alloy (MoNbTaVW)", "Liquid Metal Alloy (Zr-based)",
    "Biodegradable Magnesium (Mg-Zn-Ca)", "Biodegradable Zinc (Zn-Mg)",
    "Green Copper Wire", "Recycled Tungsten", "Scrap-based Titanium Ti-6Al-4V",
    "Eco-friendly Galvanized Steel", "Low-Carbon Stainless Steel",
    "Secondary Aluminum 356", "Secondary Aluminum 380", "Recycled Tool Steel",
    "Scrap-based Inconel 718", "Eco-friendly Brass C27400", "Lead-Free Bronze C510",
    "Recycled Silicon Bronze"
] # 41

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

df.to_csv("data/raw/materials_dataset.csv", index=False)
print(f"Renamed materials. Expanded DB uniquely to {len(dirty_polymers)+len(clean_polymers)} polymers and {len(dirty_metals)+len(clean_metals)} metals.")
