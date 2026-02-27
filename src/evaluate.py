"""
evaluate.py — Full evaluation with 5 publication-quality graphs.
Graph 1: Actual vs Predicted (2×5 grid for all 10 targets)
Graph 2: Feature Importance Heatmap (10 targets × 10 features)
Graph 3: Property Correlation Matrix
Graph 4: Eco-Score vs Key Properties (sustainability trade-offs)
Graph 5: Residual Distributions (10-panel)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR  = os.path.join(ROOT, "models")
PROC_DIR    = os.path.join(ROOT, "data", "processed")
RAW_CSV     = os.path.join(ROOT, "data", "raw", "materials_dataset.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from data_prep import FEATURE_COLS, TARGET_COLS, TARGET_META

sns.set_theme(style="darkgrid", font_scale=1.0)
PALETTE = sns.color_palette("tab10", n_colors=10)


def load_artifacts():
    bundle   = joblib.load(os.path.join(MODELS_DIR, "material_predictor.pkl"))
    test_df  = pd.read_csv(os.path.join(PROC_DIR, "features_test.csv"))
    train_df = pd.read_csv(os.path.join(PROC_DIR, "features_train.csv"))
    val_df   = pd.read_csv(os.path.join(PROC_DIR, "features_val.csv"))
    X_test   = test_df[FEATURE_COLS]
    y_test   = test_df[TARGET_COLS]
    X_train  = train_df[FEATURE_COLS]
    y_train  = train_df[TARGET_COLS]
    X_val    = val_df[FEATURE_COLS]
    y_val    = val_df[TARGET_COLS]
    return bundle, X_test, y_test, X_train, y_train, X_val, y_val


def stack_predict(bundle, target, X_test):
    """Predict using the right model based on 'is_alloy' flag."""
    preds = np.zeros(len(X_test))
    
    # Polymer predictions (scaled is_alloy is negative)
    poly_mask = (X_test["is_alloy"] < 0).values
    if poly_mask.any():
        X_poly = X_test[poly_mask]
        m = bundle["polymer"][target]
        preds[poly_mask] = m["meta"].predict(
            np.column_stack([m["rf"].predict(X_poly), m["xgb"].predict(X_poly)])
        )
        
    # Alloy predictions (scaled is_alloy is positive)
    alloy_mask = (X_test["is_alloy"] > 0).values
    if alloy_mask.any():
        X_alloy = X_test[alloy_mask]
        m = bundle["alloy"][target]
        preds[alloy_mask] = m["meta"].predict(
            np.column_stack([m["rf"].predict(X_alloy), m["xgb"].predict(X_alloy)])
        )
        
    return preds


# ─── GRAPH 1: Actual vs Predicted (2×5) ───────────────────────────────────
def plot_actual_vs_predicted(models, X_test, y_test, name_prefix, title_suffix):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f"Actual vs Predicted — All 10 Material Properties ({title_suffix})",
                 fontsize=15, fontweight="bold", y=1.02)
    axes = axes.flatten()
    results = {}
    for i, (target, color) in enumerate(zip(TARGET_COLS, PALETTE)):
        ax     = axes[i]
        preds  = stack_predict(models, target, X_test)
        y      = y_test[target].values
        results[target] = {"preds": preds, "y": y,
                           "mae": mean_absolute_error(y, preds),
                           "rmse": np.sqrt(mean_squared_error(y, preds)),
                           "r2":   r2_score(y, preds)}

        lmin = min(y.min(), preds.min())
        lmax = max(y.max(), preds.max())
        margin = (lmax - lmin) * 0.05
        lims = [lmin - margin, lmax + margin]

        ax.scatter(y, preds, alpha=0.75, s=35, color=color,
                   edgecolors="white", linewidth=0.4)
        ax.plot(lims, lims, "k--", lw=1.0, alpha=0.55, label="Perfect")
        ax.fill_between(lims, [v*0.85 for v in lims], [v*1.15 for v in lims],
                        alpha=0.06, color=color)
        name, unit, _ = TARGET_META[target]
        ax.set_title(f"{name}\nR²={results[target]['r2']:.3f}  MAE={results[target]['mae']:.2f}{unit}",
                     fontsize=8.5)
        ax.set_xlabel(f"Actual ({unit})", fontsize=7)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=7)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"01_actual_vs_predicted_{name_prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Graph 1 ({name_prefix}) saved → {path}")
    return results


# ─── GRAPH 2: Feature Importance Heatmap ──────────────────────────────────
def plot_feature_importance_heatmap(bundle):
    for class_label in ["polymer", "alloy"]:
        models = bundle[class_label]
        fi_data = {}
        for target in TARGET_COLS:
            fi_data[target] = models[target]["rf_fi"]

        fi_df = pd.DataFrame(fi_data, index=FEATURE_COLS).T
        fi_df.index = [TARGET_META[t][0] for t in TARGET_COLS]
        fi_df.columns = [c.replace("_", " ").title() for c in FEATURE_COLS]

        fig, ax = plt.subplots(figsize=(13, 7))
        sns.heatmap(fi_df, annot=True, fmt=".2f", cmap="YlGn",
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size": 7.5}, ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_title(f"Feature Importance Heatmap — {class_label.upper()}",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Input Feature", fontsize=10)
        ax.set_ylabel("Predicted Property", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"02_feature_importance_heatmap_{class_label}s.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[evaluate] Graph 2 saved → {path}")


# ─── GRAPH 3: Property Correlation Matrix ─────────────────────────────────
def plot_correlation_matrix():
    df  = pd.read_csv(RAW_CSV)
    
    for class_label in ["polymer", "alloy"]:
        sub = df[df["material_class"] == class_label][TARGET_COLS].copy()
        sub.columns = [TARGET_META[t][0] for t in TARGET_COLS]

        corr = sub.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 7.5}, ax=ax,
                    cbar_kws={"shrink": 0.75, "label": "Pearson r"})
        ax.set_title(f"Property Correlation Matrix — {class_label.upper()}",
                     fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"03_property_correlation_matrix_{class_label}s.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[evaluate] Graph 3 saved → {path}")


# ─── GRAPH 4: Eco-Score vs Key Properties ─────────────────────────────────
def plot_eco_score_tradeoffs():
    df  = pd.read_csv(RAW_CSV)
    targets_to_plot = [
        ("Tg_celsius",                "Glass Transition Temp (°C)"),
        ("tensile_strength_MPa",      "Tensile Strength (MPa)"),
        ("thermal_conductivity_WmK",  "Thermal Conductivity (W/m·K)"),
        ("water_absorption_pct",      "Water Absorption (%)"),
        ("log10_elec_conductivity",   "Elec. Conductivity (log₁₀ S/m)"),
        ("oxygen_permeability_barrer","O₂ Permeability (Barrers)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Eco / Sustainability Score vs Key Material Properties",
                 fontsize=14, fontweight="bold", y=1.02)
    axes = axes.flatten()
    colors = {"polymer": "#2ecc71", "alloy": "#3498db"}

    for ax, (target, label) in zip(axes, targets_to_plot):
        for cls, cmap_c in colors.items():
            mask = df["material_class"] == cls
            ax.scatter(df.loc[mask, "eco_score"], df.loc[mask, target],
                       alpha=0.65, s=30, color=cmap_c, label=cls.capitalize(),
                       edgecolors="white", linewidth=0.3)

        # Trend line (all points)
        z = np.polyfit(df["eco_score"], df[target], 1)
        p = np.poly1d(z)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, p(xs), "r--", lw=1.2, alpha=0.7, label="Trend")

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Eco Score (0=petroleum → 1=bio-based)", fontsize=7.5)
        ax.set_ylabel(label, fontsize=7.5)
        ax.legend(fontsize=6.5)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_eco_score_vs_properties.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Graph 4 saved → {path}")


# ─── GRAPH 5: Residual Distributions ──────────────────────────────────────
def plot_residuals(results, name_prefix, title_suffix):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle(f"Residual Distributions — All 10 Targets ({title_suffix})",
                 fontsize=14, fontweight="bold", y=1.02)
    axes = axes.flatten()

    for i, (target, color) in enumerate(zip(TARGET_COLS, PALETTE)):
        ax  = axes[i]
        res = results[target]["y"] - results[target]["preds"]
        sns.histplot(res, kde=True, ax=ax, color=color, bins=12,
                     edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", ls="--", lw=1.2)
        name, unit, _ = TARGET_META[target]
        ax.set_title(f"{name}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel(f"Residual ({unit})", fontsize=7.5)
        ax.set_ylabel("Count", fontsize=7.5)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"05_residual_distributions_{name_prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Graph 5 ({name_prefix}) saved → {path}")


# ─── Text Report ───────────────────────────────────────────────────────────
def print_report(results, name_prefix, title_suffix):
    lines = [
        "═" * 80,
        f"  🌿  Eco-Material Property Predictor — Evaluation Report ({title_suffix})",
        "═" * 80,
        f"  Test samples : {len(next(iter(results.values()))['y'])}",
        "",
        f"  {'Property':<32}  {'Unit':<10}  {'MAE':>10}  {'RMSE':>10}  {'R²':>8}",
        "  " + "─" * 76,
    ]
    for target, res in results.items():
        name, unit, icon = TARGET_META[target]
        lines.append(
            f"  {icon} {name:<30}  {unit:<10}  "
            f"{res['mae']:>10.3f}  {res['rmse']:>10.3f}  {res['r2']:>8.4f}"
        )
    lines += ["", "═" * 72]
    text = "\n".join(lines)
    print(text)
    path = os.path.join(RESULTS_DIR, f"evaluation_report_{name_prefix}.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n[evaluate] Report saved → {path}")


def run():
    print("[evaluate] Loading model & test data...")
    bundle, X_test, y_test, X_train, y_train, X_val, y_val = load_artifacts()

    # ── Separate polymer and alloy for distinct reports ────────────────
    poly_mask = (X_test["is_alloy"] < 0).values
    X_poly = X_test[poly_mask].reset_index(drop=True)
    y_poly = y_test[poly_mask].reset_index(drop=True)
    
    alloy_mask = (X_test["is_alloy"] > 0).values
    X_alloy = X_test[alloy_mask].reset_index(drop=True)
    y_alloy = y_test[alloy_mask].reset_index(drop=True)

    print(f"\\n[evaluate] --- POLYMER EVALUATION ({len(X_poly)} samples) ---")
    poly_results = plot_actual_vs_predicted(bundle, X_poly, y_poly, "polymers", "Polymers Only")
    plot_residuals(poly_results, "polymers", "Polymers Only")
    print_report(poly_results, "polymers", "Polymers")

    if len(X_alloy) > 0:
        print(f"\\n[evaluate] --- ALLOY EVALUATION ({len(X_alloy)} samples) ---")
        alloy_results = plot_actual_vs_predicted(bundle, X_alloy, y_alloy, "alloys", "Alloys Only")
        plot_residuals(alloy_results, "alloys", "Alloys Only")
        print_report(alloy_results, "alloys", "Alloys")

    print("\\n[evaluate] Generating Graph 2: Feature Importance Heatmap...")
    plot_feature_importance_heatmap(bundle)

    print("[evaluate] Generating Graph 3: Property Correlation Matrix...")
    plot_correlation_matrix()

    print("[evaluate] Generating Graph 4: Eco-Score vs Properties...")
    plot_eco_score_tradeoffs()

    print("\\n[evaluate] ✅  Evaluation graphs and reports saved to results/")


if __name__ == "__main__":
    run()
