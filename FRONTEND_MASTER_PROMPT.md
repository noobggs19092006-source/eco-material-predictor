# 🌿 MASTER FRONTEND PROMPT: Eco-Material Innovation Engine
## For use with: v0.dev, Bolt.new, Lovable, or any AI frontend builder

---

## PROJECT OVERVIEW

You are building the frontend for the **Eco-Material Innovation Engine** — a hackathon-winning web application powered by a Python machine-learning backend (FastAPI). The app predicts the physical properties of eco-friendly materials and recommends green alternatives to petroleum plastics. This is for a hackathon, so the design must be **jaw-dropping, premium, and unlike anything judges have ever seen**.

The backend already exists and is fully functional at `http://localhost:8000`. You are ONLY building the frontend. Everything in this document is the source of truth.

---

## VISUAL DESIGN MANDATE

**Theme: "Liquid Glass Biopunk"**

- **Background:** Near-black (`#050810`). Animated glowing orbs — one large, slow-moving, soft green/teal glow (`#00ff87`) and one smaller blue/purple glow (`#6c63ff`) drifting softly in the background. They should be CSS radial gradients animated with a slow 20-second keyframe drift. These orbs sit BEHIND everything.
- **Cards/Panels:** True glassmorphism. `background: rgba(255, 255, 255, 0.03)`, `backdrop-filter: blur(24px)`, `border: 1px solid rgba(255, 255, 255, 0.12)`, `border-radius: 24px`, `box-shadow: 0 8px 40px rgba(0,0,0,0.5)`. The panels must be visibly transparent — the orbs glow through them.
- **Color Palette:**
  - Primary Green (eco): `#2ecc71` / `#00ff87`
  - Accent Blue: `#3498db`
  - Warning Yellow: `#f1c40f`
  - Purple: `#9b59b6`
  - Text Primary: `#f0f0f0`
  - Text Muted: `rgba(255,255,255,0.45)`
- **Typography:** Use [Inter](https://fonts.google.com/specimen/Inter) from Google Fonts. Headings: `font-weight: 800`. All caps labels: `letter-spacing: 0.15em`.
- **Micro-animations:** All number values must animate (CountUp) when they appear or update. All cards must have a subtle `transition: all 0.3s ease` with a `translateY(-4px)` on hover. Sliders should have a glowing thumb. The Radar Chart must animate its polygon drawing in on first render.
- **Dock:** A floating, macOS-style icon dock at the bottom of the screen with magnification on hover. Contains: Home, Predictor, Green Recommender, About icons. Icons are from any SVG icon set (Lucide, Heroicons, etc.).

---

## APPLICATION STRUCTURE

Two main pages/sections:

### 1. Hero / Landing Section
- Full viewport height, dark background with orbs behind it.
- Big animated gradient title: **"Eco-Material Innovation Engine"** (gradient: green → yellow → orange, animating slowly with `background-position`).
- Subtitle: `"Predicting Next-Generation Sustainable Materials with AI Ensembles"`
- Two pill-shaped call-to-action buttons:
  - `🔬 Predict Properties` (green border, glassmorphism fill)
  - `🌱 Find Green Alternatives` (blue border, glassmorphism fill)
- Subtle floating particle animation in the background (simple CSS, small white dots drifting upward).

### 2. Property Predictor Section

**Layout:** Two-column grid on desktop (`lg:grid-cols-2`), stacked on mobile.

#### LEFT COLUMN: "Material Composition" Panel
Glass card with the heading "Material Composition" and a mode toggle button in the top-right corner:
- If **Polymer Mode (default):** Button is green and says `🌿 Polymer Mode`
- If **Alloy Mode:** Button is blue and says `✦ Alloy Mode`

**Below the toggle, render these sliders:**

For POLYMER MODE (always show):
| Label | Min | Max | Default | Step |
|---|---|---|---|---|
| Repeat Unit MW | 10 | 600 | 72 | 1 |
| Backbone Flexibility | 0 | 1 | 0.4 | 0.01 |
| Polarity Index | 0 | 3 | 2 | 0.1 |
| H-Bond Capacity | 0 | 5 | 2 | 0.1 |
| Aromatic Content | 0 | 1 | 0 | 0.01 |
| Crystallinity | 0 | 1 | 0.35 | 0.01 |
| Bio-based Eco Score | 0 | 1 | 1.0 | 0.01 |

For ALLOY MODE (replace polymer-specific fields):
| Label | Min | Max | Default | Step |
|---|---|---|---|---|
| Average Atomic Weight | 10 | 300 | 27 | 1 |
| Backbone Flexibility | 0 | 1 | 0.7 | 0.01 |
| Polarity Index | 0 | 3 | 2 | 0.1 |
| Alloying Content | 0 | 1 | 0.5 | 0.01 |
| Crystallinity | 0 | 1 | 0.9 | 0.01 |
| Recycled Content Score | 0 | 1 | 0.6 | 0.01 |

**For each slider:**
- Show the label on the left, the current numerical value on the right (updates live as you drag)
- A custom-styled range input:
  - Track: `background: rgba(255,255,255,0.1)`, `height: 6px`, `border-radius: 99px`
  - Filled portion (up to thumb): green (`#2ecc71`) for polymer, blue (`#3498db`) for alloy
  - Thumb: white circle with a soft glow shadow, `width: 18px`, `height: 18px`
- After the slider finishes moving (debounce 300ms), automatically fire an API call to the backend

**"Predict" Button:**
- Full width, below the sliders.
- Green gradient, bold text "Run Ensemble Prediction", with a loading spinner that appears during the API call.

#### RIGHT COLUMN: Two stacked cards

**TOP: "Property Radar Profile" Card**
- A `RadarChart` (use `recharts` library) that visualises 6 predicted output properties:
  - Tensile Strength, Glass Transition Temp, Young's Modulus, Density, Thermal Conductivity, Elongation at Break
- All values are normalised 0–100 for the chart axes.
- Chart is dark: `backgroundColor: transparent`, polar grid lines: `rgba(255,255,255,0.15)`, axis labels: `rgba(255,255,255,0.7)`, filled polygon: green (#2ecc71) at 60% opacity for polymer, blue (#3498db) for alloy.
- Before first prediction, show a pulsing placeholder text: `"Adjust sliders to compute prediction..."`
- After a prediction, animate the polygon morphing from its previous shape to its new shape.

**BOTTOM: "Key ML Predictions" Card**
- A 2×2 grid of metric cards. Each card is a glassmorphism sub-card.
- Show these 4 values with CountUp animation:
  1. `Tensile Strength` → green text, unit: MPa
  2. `Glass Transition (Tg)` → blue text, unit: °C
  3. `Young's Modulus` → yellow text, unit: GPa
  4. `Density` → purple text, unit: g/cm³
- Each card should also show a confidence interval: `± X.XX` in muted text below the main number
- Only render this section after the first API call completes. Before that, show a placeholder: `"Run a prediction to see results"`

---

### 3. Green Alternative Recommender Section

**Layout:** Single column, max width 900px, centred.

**Top: Material Search Panel (Glass Card)**
- Heading: `🌱 Green Alternative Recommender`
- Subheading: `"Find bio-based materials that match the performance of petroleum plastics"`
- A **searchable dropdown** (autocomplete combobox) listing all 26 petroleum-based materials. The user types a name like "ABS" and a filtered dropdown appears.
- Full-width `"Find Green Alternatives"` button (green gradient). On click, fires the API call.

**The 26 petroleum materials to pre-load in the dropdown:**
```
ABS (conventional), HDPE (conventional), LCP (Liquid Crystal Polymer),
LDPE (conventional), PAI (Polyamide-imide), PBAT (Polybutylene adipate terephthalate),
PEEK (bio-grade), PEI (bio-compatible), PEI Ultem, PES (Polyethersulfone),
PET (conventional), POM (Polyoxymethylene), PP (conventional),
PPO (Polyphenylene oxide), PSU (Polysulfone), PTFE (Teflon),
PVC (conventional), Poly(caprolactone) PCL, Poly(ether ether ketone) bio-PEEK,
Polyamide 6 (conventional), Polyamide 66 (conventional),
Polycarbonate (conventional), Polyimide (Kapton),
Polyphenylene sulfide PPS, Polystyrene (conventional), SAN (conventional)
```

**Below the search: "Petroleum Target" Panel (Red-bordered glass card)**
Shows the selected material's real properties:
- Name in large bold white text
- Eco-Score shown as a red badge: `0.00`
- Property badges: Tensile, Tg, Density, Elongation

**Below that: 3 Alternative Cards side-by-side (or stacked on mobile)**
Each alternative card is a glass card with:
- A rank medal in the top left: 🥇 🥈 🥉
- Material name in green
- `Eco-Score` shown as large green badge (e.g. `1.00`)
- Property comparison:
  - For each of: Tensile, Tg, Density, Elongation
  - Show the alternative's value
  - Show a small `▲ +X%` or `▼ -X%` badge relative to the petroleum target
- A `Performance Match` score shown as a green circular progress ring (e.g. `98.7%`)
- Border: `1px solid rgba(46, 204, 113, 0.3)`, glowing on hover

---

## API ENDPOINTS

All API calls go to `http://localhost:8000`.

### POST `/predict`
**Headers:** `Content-Type: application/json`
**Body (Polymer example):**
```json
{
  "repeat_unit_MW": 72.0,
  "backbone_flexibility": 0.4,
  "polarity_index": 2.0,
  "hydrogen_bond_capacity": 2.0,
  "aromatic_content": 0.0,
  "crystallinity_tendency": 0.35,
  "eco_score": 1.0,
  "is_alloy": 0,
  "mw_flexibility": 28.8,
  "polar_hbond": 4.0
}
```

> IMPORTANT: `mw_flexibility = repeat_unit_MW × backbone_flexibility` and `polar_hbond = polarity_index × hydrogen_bond_capacity`. Compute these in the frontend before sending.

**Response:**
```json
{
  "predictions": {
    "Tg_celsius": 66.7,
    "tensile_strength_MPa": 85.5,
    "youngs_modulus_GPa": 3.15,
    "density_gcm3": 1.238,
    "thermal_conductivity_WmK": 0.795,
    "log10_elec_conductivity": -12.8,
    "elongation_at_break_pct": 19.5,
    "dielectric_constant": 4.54,
    "water_absorption_pct": 0.505,
    "oxygen_permeability_barrer": 17.79
  },
  "confidence": {
    "Tg_celsius": 2.1,
    "tensile_strength_MPa": 3.4,
    ...
  }
}
```

### GET `/materials/alternatives/{material_name}`
**Example:** `GET /materials/alternatives/ABS%20(conventional)`
**Response:**
```json
[
  {
    "material_name": "Poly(isosorbide adipate)",
    "eco_score": 1.0,
    "tensile_strength_MPa": 83.7,
    "Tg_celsius": 67.1,
    "youngs_modulus_GPa": 2.86,
    "density_gcm3": 1.27,
    "elongation_at_break_pct": 20.9
  },
  ...
]
```

### GET `/health`
Simple health check. Returns `{ "status": "ok", "model_loaded": true }`.

---

## TECHNOLOGY STACK

- **Framework:** React (with Vite)
- **Styling:** Plain CSS or CSS Modules (no Tailwind required, but allowed)
- **Charts:** `recharts` library for the Radar Chart
- **HTTP:** `axios`
- **Animations:** `framer-motion` for CountUp, card entrances, and radar chart morphing
- **Icons:** `lucide-react`
- **No backend code** — only consume the API above.

---

## EXTRA POLISH DETAILS

1. **Connection status indicator:** Small pill in the top-right corner. Green dot = `API Connected`, Red dot = `API Offline`. Polls `/health` every 5 seconds.
2. **Toast notifications:** When the API call succeeds, show a brief green toast: `"✅ Stacked Ensemble computed in 0.3s"`. On error, red toast: `"❌ Backend offline — run make train first"`.
3. **Eco Score visual:** Anywhere the eco_score is shown, display it as a coloured badge:
   - `0.0 – 0.3` = Red badge (Petroleum)
   - `0.3 – 0.7` = Yellow badge (Mixed)
   - `0.7 – 1.0` = Green badge (Bio-based)
4. **About / Info tab:** A glass panel explaining the project. Copy: *"This AI engine uses a two-layer Stacked Ensemble (Random Forest + XGBoost → Ridge meta-learner) trained on 285 materials with scientifically-grounded QSPR formulas. Tested on a completely held-out 20% blind dataset. R² > 0.90 across all 10 material properties."*
5. **Mobile responsive:** The two-column layout collapses to single column below 768px. The dock icons shrink. The radar chart remains full width.
