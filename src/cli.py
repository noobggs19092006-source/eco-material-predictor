"""
cli.py — Interactive terminal predictor for 10 material properties + PDB export.
"""
import os, sys, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich import box
import questionary
from questionary import Style as QStyle

from predict import predict, format_value
from data_prep import TARGET_COLS, TARGET_META
from generate_pdb import guess_material_key, save_pdb, list_available
from recommend import find_green_alternatives, list_petroleum_materials

console = Console()

STYLE = QStyle([
    ("qmark",       "fg:#2ecc71 bold"),
    ("question",    "fg:#ecf0f1 bold"),
    ("answer",      "fg:#3498db bold"),
    ("pointer",     "fg:#2ecc71 bold"),
    ("highlighted", "fg:#2ecc71 bold"),
    ("selected",    "fg:#2ecc71"),
    ("instruction", "fg:#586069"),
])

# Reference ranges for context bars
RANGES = {
    "Tg_celsius":                 (-150, 400),
    "tensile_strength_MPa":       (0.1,  3000),
    "youngs_modulus_GPa":         (0.001, 600),
    "density_gcm3":               (0.85, 15),
    "thermal_conductivity_WmK":   (0.10, 250),
    "log10_elec_conductivity":    (-16, 8),
    "elongation_at_break_pct":    (0, 900),
    "dielectric_constant":        (-1, 12),
    "water_absorption_pct":       (0, 35),
    "oxygen_permeability_barrer": (0, 500),
}

# Medium tags
MEDIUM = {
    "Tg_celsius":                 "🧱 Thermal",
    "tensile_strength_MPa":       "🧱 Solid",
    "youngs_modulus_GPa":         "🧱 Solid",
    "density_gcm3":               "🧱 Bulk",
    "thermal_conductivity_WmK":   "♨️  Heat",
    "log10_elec_conductivity":    "⚡ Electric",
    "elongation_at_break_pct":    "🧱 Solid",
    "dielectric_constant":        "⚡ Electric",
    "water_absorption_pct":       "💧 Water",
    "oxygen_permeability_barrer": "💨 Air/Gas",
}


def banner():
    console.print()
    console.print(Panel.fit(
        "[bold green]🌿  Eco-Material Property Predictor  v2[/bold green]\n"
        "[dim]Predicts 10 properties across Thermal · Structural · Electrical · Environmental media[/dim]\n"
        "[dim]Powered by RF + XGBoost Stacked Ensemble[/dim]",
        border_style="green", padding=(1, 4),
    ))
    console.print()


def ask_float(q, lo, hi, ex=""):
    console.print(f"  [dim](range {lo}–{hi}{', e.g. '+ex if ex else ''})[/dim]")
    while True:
        v = Prompt.ask(f"  [cyan]{q}[/cyan]")
        try:
            val = float(v)
            if lo <= val <= hi:
                return val
            console.print(f"  [red]✗[/red] Enter a value between {lo} and {hi}")
        except ValueError:
            console.print("  [red]✗[/red] Invalid — enter a number")


def ask_int(q, lo, hi):
    console.print(f"  [dim](integer {lo}–{hi})[/dim]")
    while True:
        v = Prompt.ask(f"  [cyan]{q}[/cyan]")
        try:
            val = int(v)
            if lo <= val <= hi:
                return val
            console.print(f"  [red]✗[/red] Enter {lo}–{hi}")
        except ValueError:
            console.print("  [red]✗[/red] Enter a whole number")


def collect_features():
    console.print(Rule("[bold green]Step 1 — Material Class[/bold green]"))
    mat_class = questionary.select(
        "Material class:", style=STYLE,
        choices=["Polymer / Biopolymer", "Metal Alloy"]
    ).ask()
    is_alloy = 1 if "Alloy" in mat_class else 0

    console.print()
    console.print(Rule("[bold green]Step 2 — Molecular Properties[/bold green]"))

    if is_alloy == 0:
        mw = ask_float("Repeat unit MW (g/mol)", 28, 1000, "72 for PLA")
    else:
        mw = 0.0

    flex = ask_float("Backbone flexibility (0=rigid, 1=flexible)", 0.0, 1.0,
                     "0.4 PLA, 1.0 metals")

    if is_alloy == 0:
        polarity  = ask_int("Polarity index (0=nonpolar, 3=highly polar)", 0, 3)
        hbond     = ask_int("H-bond capacity (0=none, 5=strong)", 0, 5)
        aromatic  = ask_float("Aromatic content (0–1)", 0.0, 1.0, "0.5 polystyrene")
        cryst     = ask_float("Crystallinity tendency (0=amorphous, 1=crystalline)",
                              0.0, 1.0, "0.35 PLA")
    else:
        polarity, hbond, aromatic, cryst = 0, 0, 0.0, 1.0
        console.print("  [dim]Polarity/H-bond/Aromatic → defaults for metals[/dim]")

    eco = ask_float("Eco/sustainability score (0=petroleum, 1=bio-based)",
                    0.0, 1.0, "1.0 PLA, 0.0 conventional PE")

    return {
        "repeat_unit_MW":         mw,
        "backbone_flexibility":   flex,
        "polarity_index":         polarity,
        "hydrogen_bond_capacity": hbond,
        "aromatic_content":       aromatic,
        "crystallinity_tendency": cryst,
        "eco_score":              eco,
        "is_alloy":               is_alloy,
    }


def context_bar(target, val):
    lo, hi = RANGES[target]
    if hi == lo:
        return "─" * 10
    norm   = (val - lo) / (hi - lo)
    norm   = max(0.0, min(1.0, norm))
    filled = int(norm * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    if norm < 0.33:   color = "cyan"
    elif norm < 0.66: color = "yellow"
    else:             color = "red"
    return f"[{color}]{bar}[/{color}]"


def render_results(result):
    console.print()
    console.print(Rule("[bold green]Step 3 — Prediction Results[/bold green]"))
    console.print()

    table = Table(show_header=True, header_style="bold green",
                  box=box.ROUNDED, border_style="dim green", padding=(0, 1))
    table.add_column("Ic",  width=3,  justify="center")
    table.add_column("Property",      width=28, style="bold")
    table.add_column("Medium",        width=13)
    table.add_column("Predicted",     width=22, justify="right")
    table.add_column("±Confidence",   width=16, justify="right", style="dim")
    table.add_column("Context",       width=13, justify="center")

    for target in TARGET_COLS:
        name, unit, icon = TARGET_META[target]
        val = result["predictions"][target]
        ci  = result["confidence"][target]

        display = format_value(target, val)
        ci_str  = f"±{ci:.2f} {unit}" if target != "log10_elec_conductivity" else f"±{ci:.2f}"

        table.add_row(
            icon,
            name,
            MEDIUM[target],
            f"[bold]{display} {unit}[/bold]" if target != "log10_elec_conductivity" else f"[bold]{display}[/bold]",
            ci_str,
            context_bar(target, val),
        )

    console.print(table)

    # ── Confidence summary ───────────────────────────────────────────────────
    console.print(
        "  [dim]ℹ  Model accuracy (polymer test R²): "
        "Tg 0.98 · Tensile 0.93 · Young's 0.92 · Density 0.84 · "
        "Thermal 0.97 · Elec 0.97 · Elongation 0.95 · Dielec 0.95 · "
        "Water 0.98 · O₂ 0.97[/dim]"
    )
    console.print()

    # Tg interpretation
    tg = result["predictions"]["Tg_celsius"]
    wabs = result["predictions"]["water_absorption_pct"]
    o2   = result["predictions"]["oxygen_permeability_barrer"]
    ec   = result["predictions"]["log10_elec_conductivity"]

    notes = []
    if tg < 0:        notes.append("[cyan]Tg < 0°C → rubbery at room temperature (elastomeric)[/cyan]")
    elif tg < 80:     notes.append("[green]Tg 0–80°C → soft/semi-rigid at room temp (packaging grade)[/green]")
    elif tg < 150:    notes.append("[yellow]Tg 80–150°C → engineering plastic (structural use)[/yellow]")
    else:             notes.append("[red]Tg > 150°C → high-performance / aerospace plastic[/red]")

    if wabs < 0.5:    notes.append("[cyan]Water absorption < 0.5% → hydrophobic, moisture-resistant[/cyan]")
    elif wabs < 3.0:  notes.append("[yellow]Water absorption 0.5–3% → moderate moisture sensitivity[/yellow]")
    else:             notes.append("[red]Water absorption > 3% → hydrophilic, dimensionally unstable in water[/red]")

    if o2 < 5:        notes.append("[green]O₂ permeability < 5 Barrers → excellent gas barrier (food packaging suitable)[/green]")
    elif o2 < 50:     notes.append("[yellow]O₂ permeability 5–50 Barrers → moderate barrier[/yellow]")
    else:             notes.append("[red]O₂ permeability > 50 Barrers → poor barrier, not suitable for gas-sensitive packaging[/red]")

    if ec < -10:      notes.append("[cyan]Electrical: excellent insulator (electronics, wire coating)[/cyan]")
    elif ec < 0:      notes.append("[yellow]Electrical: semi-insulating / antistatic range[/yellow]")
    else:             notes.append("[red]Electrical: semi-conductor or conductor range[/red]")

    console.print(Panel(
        "\n".join(f"  {n}" for n in notes),
        title="[bold]Material Interpretation[/bold]",
        border_style="dim green", padding=(0, 2)
    ))
    console.print()


def offer_pdb(features):
    if features.get("is_alloy", 0):
        console.print("  [dim]PDB export: not available for metal alloys in this version.[/dim]")
        return

    key = guess_material_key(features)
    if key is None:
        return

    console.print(f"  [dim]Closest polymer in PDB library: [bold]{key}[/bold][/dim]")
    available = list_available()
    choice = questionary.select(
        "Export PDB for VMD? Choose polymer or skip:",
        style=STYLE,
        choices=available + ["─── Skip ───"],
    ).ask()

    if "Skip" in choice:
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "results", f"molecule_{choice}_{ts}.pdb")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    save_pdb(choice, out)
    console.print(f"\n  [bold green]✅  PDB saved → {out}[/bold green]")
    console.print(f"  [dim]Open in VMD: [cyan]vmd {out}[/cyan][/dim]")
    console.print()


def export_csv(result):
    import csv
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "results", f"prediction_{ts}.csv")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["property", "unit", "medium", "predicted", "confidence_pm"])
        w.writeheader()
        for target in TARGET_COLS:
            name, unit, _ = TARGET_META[target]
            w.writerow({
                "property": name,
                "unit": unit,
                "medium": MEDIUM[target].split(" ", 1)[-1].strip(),   # drop emoji, keep text only
                "predicted": round(result["predictions"][target], 4),
                "confidence_pm": round(result["confidence"][target], 4),
            })
    console.print(f"  [green]✅  Saved → {out}[/green]")


def _predict_mode():
    """Original Property Predictor loop."""
    banner()
    while True:
        try:
            features = collect_features()
        except (KeyboardInterrupt, EOFError):
            break

        console.print()
        with console.status("[bold green]Running ensemble prediction...[/bold green]", spinner="dots"):
            try:
                result = predict(features)
            except FileNotFoundError as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                console.print("[dim]  Run [cyan]make train[/cyan] first to generate the model.[/dim]")
                break

        render_results(result)
        offer_pdb(features)

        try:
            if Confirm.ask("  [dim]Export prediction to CSV?[/dim]", default=False):
                export_csv(result)
            if not Confirm.ask("  [dim]Run another prediction?[/dim]", default=True):
                break
        except (KeyboardInterrupt, EOFError):
            break

def green_mode():
    """Interactive Green Alternative Recommender."""
    console.print()
    console.print(Rule("[bold green]🌿  Green Alternative Recommender[/bold green]"))
    console.print("[dim]  Find eco-friendly materials that match the performance of petroleum plastics.[/dim]\n")

    options = list_petroleum_materials()
    if not options:
        console.print("[red]No petroleum materials found in dataset.[/red]")
        return

    console.print("[dim]Available petroleum materials:[/dim]")
    for i, name in enumerate(options, 1):
        console.print(f"  [dim]{i:>2}.[/dim] {name}")
    console.print()
    chosen = Prompt.ask("  [bold]Enter material name[/bold] [dim](or part of it, e.g. 'ABS')[/dim]")

    result = find_green_alternatives(chosen, top_n=3)

    if result["error"]:
        console.print(f"\n[bold red]Error:[/bold red] {result['error']}")
        return

    target = result["target"]
    console.print()
    console.print(Panel(
        f"[bold white]{target['material_name']}[/bold white]\n"
        f"[dim]Eco-Score:[/dim] [red]{target['eco_score']:.2f}[/red]  "
        f"Tensile: [yellow]{target['tensile_strength_MPa']:.1f} MPa[/yellow]  "
        f"Tg: {target['Tg_celsius']:.1f}°C  "
        f"Density: {target['density_gcm3']:.2f} g/cm³",
        title="[bold red]Petroleum Target[/bold red]",
        border_style="red",
    ))

    console.print()
    table = Table(
        title="[bold green]🌱 Top 3 Bio-Based Alternatives[/bold green]",
        box=box.ROUNDED,
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
            f"{alt['performance_match_pct']}%",
            f"{alt['elongation_at_break_pct']:.1f}",
        )

    console.print(table)
    console.print("\n[dim]  Match % = structural/mechanical performance similarity (higher = closer match)[/dim]")
    console.print("[dim]  Eco-Score 0 = petroleum · 1 = fully bio-based[/dim]\n")


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
            "🌱  Green Recommender   (find eco alternatives to a petroleum plastic)",
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

    # ---- Property Predictor (original loop) ---
    while True:
        console.print(Rule("[bold green]🌿 New Prediction[/bold green]"))
        material_class = questionary.select(
            "  Material class:",
            choices=["🧪 Polymer (eco-plastic)", "⚙️  Metal Alloy"],
            style=STYLE,
        ).ask()
        if material_class is None:
            break
        is_alloy = "Alloy" in material_class

        features = collect_features(is_alloy)
        if features is None:
            break

        with console.status("[green]Running ensemble inference…[/green]", spinner="dots"):
            try:
                result = predict(features)
            except FileNotFoundError as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                console.print("[dim]  Run [cyan]make train[/cyan] first to generate the model.[/dim]")
                break

        render_results(result)
        offer_pdb(features)

        try:
            if Confirm.ask("  [dim]Export prediction to CSV?[/dim]", default=False):
                export_csv(result)
            if not Confirm.ask("  [dim]Run another prediction?[/dim]", default=True):
                break
        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[bold green]🌿  Thank you for using Eco-Material Predictor.[/bold green]\n")


if __name__ == "__main__":
    main()
