import os
import sys
import pandas as pd
import numpy as np
import joblib
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "material_predictor.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
PDB_SCRIPT = os.path.join(ROOT, "src", "generate_pdb.py")

console = Console()
STYLE = questionary.Style([
    ("qmark", "fg:#2ecc71 bold"),
    ("question", "bold"),
    ("choice", "fg:#ffffff"),
    ("selected", "fg:#2ecc71 bold"),
    ("instruction", "fg:#888888 italic"),
])

def ask_float(prompt, min_val, max_val, default_str):
    """Robust float prompt."""
    while True:
        val = Prompt.ask(f"  {prompt} ({min_val}–{max_val})", default=default_str)
        try:
            f = float(val)
            if min_val <= f <= max_val:
                return f
            console.print(f"  [red]Please enter a value between {min_val} and {max_val}.[/red]")
        except ValueError:
            console.print("  [red]Invalid number.[/red]")

def collect_features(is_alloy):
    """Prompt user for 7 physical parameters based on class."""
    console.print()
    if not is_alloy:
        console.print("[dim]  — 1. Molecular Structure [/dim]")
        mw = ask_float("Repeat unit MW [g/mol]", 10.0, 600.0, "72.0 PLA")
        flex = ask_float("Backbone flexibility (0=rigid, 1=flexible)", 0.0, 1.0, "0.4")

        console.print("[dim]  — 2. Intermolecular Forces [/dim]")
        polarity = ask_float("Polarity index (0=non-polar, 3=highly polar)", 0.0, 3.0, "2.0")
        hbond = ask_float("H-bond capacity (0-5)", 0.0, 5.0, "2.0")
        aromatic = ask_float("Aromatic content (0-1)", 0.0, 1.0, "0.0 PLA")
        cryst = ask_float("Crystallinity tendency (0=amorphous, 1=crystalline)", 0.0, 1.0, "0.35 PLA")

        console.print("[dim]  — 3. Eco-Score [/dim]")
        eco = ask_float("Eco/sustainability score (0=petroleum, 1=bio-based)", 0.0, 1.0, "1.0 PLA")

        return {
            "inputs": {
                "repeat_unit_MW": float(mw),
                "backbone_flexibility": float(flex),
                "polarity_index": float(polarity),
                "hydrogen_bond_capacity": float(hbond),
                "aromatic_content": float(aromatic),
                "crystallinity_tendency": float(cryst),
                "eco_score": float(eco),
            },
            "mode": "polymer"
        }
    else:
        console.print("[dim]  — 1. Structural Physics [/dim]")
        shear = ask_float("Shear Modulus [GPa]", 0.1, 200.0, "80.0 generic steel")
        melt = ask_float("Melting Temperature [°C]", 300.0, 4000.0, "1500.0 Fe")

        console.print("[dim]  — 2. Atomic Level Interactions [/dim]")
        r_diff = ask_float("Atomic Radius Difference [%]", 0.0, 15.0, "5.0")
        enthalpy = ask_float("Mixing Enthalpy [kJ/mol]", -50.0, 20.0, "-10.0")
        valence = ask_float("Valence Electrons (avg per atom)", 3.0, 12.0, "7.5")
        en_diff = ask_float("Electronegativity Difference", 0.0, 1.0, "0.2")

        console.print("[dim]  — 3. Environmental Impact [/dim]")
        eco = ask_float("Recyclability / Eco Score (0=toxic/complex, 1=green/pure)", 0.0, 1.0, "0.5")

        return {
            "inputs": {
                "atomic_radius_difference": float(r_diff),
                "mixing_enthalpy": float(enthalpy),
                "valence_electrons": float(valence),
                "electronegativity_diff": float(en_diff),
                "shear_modulus": float(shear),
                "melting_temp": float(melt),
                "eco_score": float(eco),
            },
            "mode": "metal"
        }

def guess_material_key(inputs):
    """Hack for the 3D visualizer based on MW proxy."""
    mw = inputs.get("repeat_unit_MW", 0)
    if mw == 0: return None
    if mw < 35: return "pe"
    if mw < 45: return "pp"
    if 45 <= mw < 60: return "pvc"
    if 60 <= mw < 80: return "pla"
    if 80 <= mw < 150: return "ps"
    return "pet"

def context_bar(target, val):
    m = {
        "Tg_celsius": ("°C", "[104°C PS, 150°C PC, 250°C PEEK | Metals: 500-3000°C]"),
        "tensile_strength_MPa": ("MPa", "[30 ABS, 70 PET, 100+ PEEK | Metals: 200-2000]"),
        "youngs_modulus_GPa": ("GPa", "[2.0 PP, 3.5 PLA, 5.0+ PAI | Metals: 50-250]"),
        "density_gcm3": ("g/cm³", "[0.9 PP, 1.2 PLA, 1.4 PVC | Metals: 2.0-15.0]"),
        "thermal_conductivity_WmK": ("W/m·K", "[0.1-0.5 Plastics | Metals: 15-400]"),
        "log10_elec_conductivity": ("log₁₀ S/m", "[-14 Plastics (Insulator) | Metals: 5-8 (Conductor)]"),
        "elongation_at_break_pct": ("%", "[2% PS, 50% PET, 500% LDPE | Metals: 1-50%]"),
        "dielectric_constant": ("—", "[2-3 Teflon/PE, 4-8 Polar | Metals: N/A]"),
        "water_absorption_pct": ("%", "[<0.1% PE, 2-5% Nylon | Metals: 0.0]"),
        "oxygen_permeability_barrer": ("Barrers", "[0.1 PVDC, 5.0 PET, 100+ PE | Metals: 0.0]"),
    }
    unit, hint = m.get(target, ("", ""))
    return f"[bold white]{val:>8.2f}[/bold white] [dim]{unit:<9} {hint}[/dim]"

def predict(payload):
    """Unified inference engine for the 7-parameter generic inputs."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not compiled.")

    import warnings
    warnings.filterwarnings("ignore")

    bundle = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    mode = payload["mode"]
    inputs = payload["inputs"]

    # Fill un-provided physical features with 0.0 to match the scaler
    processed_inputs = {col: inputs.get(col, 0.0) for col in bundle["feature_cols"]}
    
    X = pd.DataFrame([processed_inputs])
    X = X[bundle["feature_cols"]] # strict ordering
    X_scaled = scaler.transform(X)

    model_type = "alloy" if mode == "metal" else "polymer"
    preds, confs = {}, {}

    for target in bundle["target_cols"]:
        m = bundle[model_type][target]
        base = np.column_stack([m["rf"].predict(X_scaled), m["xgb"].predict(X_scaled)])
        meta = m["meta"].predict(base)[0]
        
        if "density" in target or "water" in target or "oxygen" in target:
            meta = max(0.001, meta)
            
        preds[target] = float(meta)
        confs[target] = float(np.std([t.predict(X_scaled)[0] for t in m["rf"].estimators_]))

    return {"predictions": preds, "confidence": confs}

def get_app_recommendation(preds, mode):
    if mode == "polymer":
        tg = preds.get("Tg_celsius", 0)
        tens = preds.get("tensile_strength_MPa", 0)
        if tens > 80 and tg > 150: return "Aerospace & Automotive inner parts (High strength, thermal resistant)"
        if tens > 40 and tg > 60: return "Rigid packaging & Consumer electronics (Durable, moderate strength)"
        if tens < 20 and tg < 30: return "Flexible films & Eco-friendly bags (Soft, high elongation)"
        return "General purpose commercial plastics and molding"
    else:
        tens = preds.get("tensile_strength_MPa", 0)
        dens = preds.get("density_gcm3", 5.0)
        if tens > 1000 and dens < 5.0: return "Aerospace structures & Advanced lightweight vehicle frames"
        if tens > 1000: return "Heavy-duty construction, Industrial tooling, Structural supports"
        if tens < 500 and dens < 3.0: return "Lightweight consumer electronics, Bike frames, Aviation panels"
        return "General structural, marine, and architectural metal applications"

def render_results(result, payload):
    console.print("\n[bold cyan]  🧪 Prediction Results[/bold cyan]")
    app_text = get_app_recommendation(result["predictions"], payload["mode"])
    console.print(f"  [bold yellow]💡 Application:[/bold yellow] [italic]{app_text}[/italic]\n")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Target", style="green", justify="right")
    table.add_column("Value & Context", justify="left")
    
    # Render all 10 natively
    order = [
        "tensile_strength_MPa", "Tg_celsius", "youngs_modulus_GPa", "density_gcm3",
        "thermal_conductivity_WmK", "log10_elec_conductivity", "elongation_at_break_pct",
        "dielectric_constant", "water_absorption_pct", "oxygen_permeability_barrer"
    ]
    for k in order:
        val = result["predictions"].get(k, 0)
        table.add_row(k.replace("_", " ").title(), context_bar(k, val))

    console.print(Panel(table, border_style="cyan"))

def offer_pdb(payload):
    mode = payload["mode"]
    if mode == "metal":
        key = "METAL_FCC"
        desc = "Generate 3D crystal lattice structure (PDB) for this generic metal?"
    else:
        key = guess_material_key(payload["inputs"])
        if key is None: return
        desc = f"Generate 3D molecular structure (PDB) for closest heuristic match ([bold]{key.upper()}[/bold])?"

    console.print()
    if Confirm.ask(f"  [dim]{desc}[/dim]", default=False):
        import subprocess
        out_name = f"heuristic_molecule_{key}.pdb" if mode != "metal" else "generic_metal_lattice.pdb"
        try:
            subprocess.run([sys.executable, PDB_SCRIPT, key, out_name], check=True)
            console.print(f"  [bold green]✓[/bold green] Wrote [cyan]results/{out_name}[/cyan]\n")
        except Exception as e:
            console.print(f"  [red]PDB generation failed: {e}[/red]")

def green_mode():
    console.print(Rule("[bold green]🌱 Green Recommender[/bold green]"))
    console.print("[dim]Select a petroleum/toxic material to find its eco-alternatives.[/dim]\n")
    
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from recommend import find_green_alternatives, list_petroleum_materials
    
    cat_choice = questionary.select(
        "  Select Category:",
        choices=["1. Polymer Alternative", "2. Metal Alternative"],
        style=STYLE,
    ).ask()
    if not cat_choice: return
    
    mat_class = "polymer" if "Polymer" in cat_choice else "metal"
    choices = list_petroleum_materials(mat_class)
    
    query = questionary.select(
        f"  Target {cat_choice.split('.')[1].strip().split(' ')[0]}:",
        choices=choices,
        style=STYLE,
    ).ask()
    
    if not query:
        return
    
    with console.status("[green]Scanning vector database…[/green]", spinner="dots"):
        try:
            result = find_green_alternatives(query, top_n=3)
        except ValueError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            return
            
    target = result["target"]
    console.print(f"\n[bold white]Target:[/bold white] {target['material_name']} "
                  f"[dim](Eco: {target['eco_score']} | Tensile: {target['tensile_strength_MPa']:.1f} MPa)[/dim]")
    
    table = Table(
        title="[bold green]Top Green Alternatives[/bold green]",
        box=None,
        border_style="green",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank",             style="bold white",  justify="center", width=6)
    table.add_column("Material",         style="green",        justify="left",   min_width=30)
    table.add_column("Eco-Score",        style="bold green",   justify="center", width=10)
    table.add_column("Tensile (MPa)",    justify="right",      width=14)
    table.add_column("Tg (°C)",         justify="right",      width=9)
    table.add_column("Density (g/cm³)", justify="right",      width=14)
    table.add_column("Match %",         style="bold yellow",  justify="center", width=9)
    table.add_column("Elongation (%)",  justify="right",      width=15)

    medals = ["🥇", "🥈", "🥉"]
    for i, alt in enumerate(result["alternatives"]):
        tensile_diff = alt["tensile_strength_MPa"] - target["tensile_strength_MPa"]
        tensile_str = f"{alt['tensile_strength_MPa']:.1f}"
        if abs(tensile_diff) < 10:
            tensile_str = f"[green]{tensile_str}[/green]"
        elif tensile_diff > 0:
            tensile_str = f"[cyan]{tensile_str} ▲[/cyan]"
        table.add_row(
            medals[i],
            alt["material_name"],
            str(alt["eco_score"]),
            tensile_str,
            f"{alt['Tg_celsius']:.1f}",
            f"{alt['density_gcm3']:.2f}",
            f"{alt.get('match_pct', alt.get('performance_match_pct', 0))}%",
            f"{alt['elongation_at_break_pct']:.1f}",
        )
    console.print(table)
    console.print("\n[bold yellow]💡 Recommended Applications for Alternatives:[/bold yellow]")
    for i, alt in enumerate(result["alternatives"]):
        guess_mode = alt.get("material_class", "polymer")
        app_text = get_app_recommendation(alt, guess_mode)
        console.print(f"  {medals[i]} [bold]{alt['material_name']}:[/bold] [italic]{app_text}[/italic]")

    console.print("\n[dim]  Match % = structural/mechanical performance similarity (higher = closer match)[/dim]")
    console.print("[dim]  Eco-Score 0 = petroleum/toxic · 1 = fully bio-based/green[/dim]\n")

def main():
    console.print()
    console.print(Panel(
        "[bold green]🌿 Eco-Material Innovation Engine[/bold green]\n"
        "[dim]Predict material properties · Find green alternatives[/dim]",
        border_style="green",
    ))

    mode = questionary.select(
        "  Choose a mode:",
        choices=[
            "🔬  Property Predictor  (invent a new material)",
            "🌱  Green Recommender   (find eco alternatives to a petroleum/toxic material)",
        ],
        style=STYLE,
    ).ask()

    if mode is None:
        return

    if "Green" in mode:
        try:
            green_mode()
        except (KeyboardInterrupt, EOFError):
            pass
        console.print("\n[bold green]🌿  Thank you for using Eco-Material Predictor.[/bold green]\n")
        return

    while True:
        console.print(Rule("[bold green]🌿 New Prediction[/bold green]"))
        
        material_class = questionary.select(
            "  Material class:",
            choices=["🧪 Polymer (eco-plastic)", "⚙️  All Metals"],
            style=STYLE,
        ).ask()
            
        if material_class is None:
            break
            
        is_alloy = "Metals" in material_class
        payload = collect_features(is_alloy)
        
        if payload is None:
            break

        with console.status("[green]Running ensemble inference…[/green]", spinner="dots"):
            try:
                result = predict(payload)
            except FileNotFoundError as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                console.print("[dim]  Run [cyan]make train[/cyan] first to generate the model.[/dim]")
                break

        render_results(result, payload)
        offer_pdb(payload)

        try:
            if Confirm.ask("  [dim]Export prediction to CSV?[/dim]", default=False):
                out = os.path.join(ROOT, "results", "prediction_export.csv")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                
                export_dict = payload["inputs"].copy()
                export_dict.update(result["predictions"])
                
                df_out = pd.DataFrame([export_dict])
                df_out.to_csv(out, index=False)
                console.print(f"  [bold green]✓[/bold green] Exported to [cyan]{out}[/cyan]\n")
        except KeyboardInterrupt:
            break

        if not Confirm.ask("  [bold]Run another prediction?[/bold]", default=True):
            break

    console.print("\n[bold green]🌿  Thank you for using Eco-Material Predictor.[/bold green]\n")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold green]🌿  Thank you for using Eco-Material Predictor.[/bold green]\n")
