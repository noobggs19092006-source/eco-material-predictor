import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, Tooltip, ResponsiveContainer, PolarRadiusAxis } from 'recharts';
import Tilt from 'react-parallax-tilt';
import Orb from './components/Orb';
import PropTypes from 'prop-types';
import './index.css';

const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API = import.meta.env.VITE_API_URL || (isLocal ? 'http://127.0.0.1:8000' : '');

const PETROLEUM_MATERIALS = [
  'ABS (conventional)', 'HDPE (conventional)', 'LCP (Liquid Crystal Polymer)',
  'LDPE (conventional)', 'PAI (Polyamide-imide)',
  'PBAT (Polybutylene adipate terephthalate)', 'PEEK (bio-grade)',
  'PEI (bio-compatible)', 'PEI Ultem', 'PES (Polyethersulfone)',
  'PET (conventional)', 'POM (Polyoxymethylene)', 'PP (conventional)',
  'PPO (Polyphenylene oxide)', 'PSU (Polysulfone)', 'PTFE (Teflon)',
  'PVC (conventional)', 'Poly(caprolactone) PCL',
  'Poly(ether ether ketone) bio-PEEK', 'Polyamide 6 (conventional)',
  'Polyamide 66 (conventional)', 'Polycarbonate (conventional)',
  'Polyimide (Kapton)', 'Polyphenylene sulfide PPS',
  'Polystyrene (conventional)', 'SAN (conventional)',
];

const POLYMER_SLIDERS = [
  { key: 'repeat_unit_MW', label: 'Repeat Unit MW', min: 10, max: 600, default: 72, step: 1 },
  { key: 'backbone_flexibility', label: 'Backbone Flexibility', min: 0, max: 1, default: 0.4, step: 0.01 },
  { key: 'polarity_index', label: 'Polarity Index', min: 0, max: 3, default: 2, step: 0.1 },
  { key: 'hydrogen_bond_capacity', label: 'H-Bond Capacity', min: 0, max: 5, default: 2, step: 0.1 },
  { key: 'aromatic_content', label: 'Aromatic Content', min: 0, max: 1, default: 0, step: 0.01 },
  { key: 'crystallinity_tendency', label: 'Crystallinity', min: 0, max: 1, default: 0.35, step: 0.01 },
  { key: 'eco_score', label: 'Bio-based Eco Score', min: 0, max: 1, default: 1.0, step: 0.01 },
];

const ALLOY_SLIDERS = [
  { key: 'repeat_unit_MW', label: 'Avg Atomic Weight', min: 10, max: 300, default: 27, step: 1 },
  { key: 'backbone_flexibility', label: 'Backbone Flexibility', min: 0, max: 1, default: 0.7, step: 0.01 },
  { key: 'polarity_index', label: 'Polarity Index', min: 0, max: 3, default: 2, step: 0.1 },
  { key: 'aromatic_content', label: 'Alloying Content', min: 0, max: 1, default: 0.5, step: 0.01 },
  { key: 'crystallinity_tendency', label: 'Crystallinity', min: 0, max: 1, default: 0.9, step: 0.01 },
  { key: 'eco_score', label: 'Recycled Content', min: 0, max: 1, default: 0.6, step: 0.01 },
];

const METAL_SLIDERS = [
  { key: 'atomic_radius_difference', label: 'Atomic Radius Diff (%)', min: 0, max: 15.0, default: 5.0, step: 0.1 },
  { key: 'mixing_enthalpy', label: 'Mixing Enthalpy (kJ/mol)', min: -50.0, max: 20.0, default: -10.0, step: 0.5 },
  { key: 'valence_electrons', label: 'Valence Electrons', min: 3.0, max: 12.0, default: 7.5, step: 0.1 },
  { key: 'electronegativity_diff', label: 'Electronegativity Diff', min: 0, max: 0.6, default: 0.2, step: 0.01 },
  { key: 'shear_modulus', label: 'Shear Modulus (GPa)', min: 10, max: 150, default: 80, step: 1 },
  { key: 'melting_temp', label: 'Melting Temp (°C)', min: 400, max: 3500, default: 1500, step: 10 },
  { key: 'eco_score', label: 'Eco Recyclability Score', min: 0, max: 1, default: 0.5, step: 0.01 },
];

const POLYMER_RADAR_NORMS = {
  tensile_strength_MPa: { max: 260, label: 'Tensile' },
  Tg_celsius: { max: 280, min: -50, label: 'Tg' },
  youngs_modulus_GPa: { max: 15, label: "Young's" },
  density_gcm3: { max: 2.5, min: 0.5, label: 'Density' },
  thermal_conductivity_WmK: { max: 5.0, label: 'Thermal' },
  elongation_at_break_pct: { max: 80, label: 'Flex (%)' },
};

const ALLOY_RADAR_NORMS = {
  tensile_strength_MPa: { max: 260, label: 'Tensile' },
  Tg_celsius: { max: 280, min: -50, label: 'Tg' },
  youngs_modulus_GPa: { max: 15, label: "Young's" },
  density_gcm3: { max: 2.5, min: 0.5, label: 'Density' },
  thermal_conductivity_WmK: { max: 5.0, label: 'Thermal' },
  elongation_at_break_pct: { max: 80, label: 'Flex (%)' },
};

const METAL_RADAR_NORMS = {
  tensile_strength_MPa: { max: 1800, label: 'Tensile' },
  Tg_celsius: { max: 3000, label: 'Tg/Melt' },
  youngs_modulus_GPa: { max: 350, label: "Young's" },
  density_gcm3: { max: 15, label: 'Density' },
  thermal_conductivity_WmK: { max: 250, label: 'Thermal' },
  elongation_at_break_pct: { max: 50, label: 'Flex (%)' },
};

function normalise(key, val, mode) {
  let norms = POLYMER_RADAR_NORMS;
  if (mode === 'alloy') norms = ALLOY_RADAR_NORMS;
  if (mode === 'metal') norms = METAL_RADAR_NORMS;

  const n = norms[key] || { max: 100 };
  const mn = n.min ?? 0;
  return Math.min(100, Math.max(0, ((val - mn) / (n.max - mn)) * 100));
}

// Sci-fi text scramble hook
function useScramble(targetStr, duration = 800) {
  const [val, setVal] = useState(targetStr);
  const prev = useRef(targetStr);
  useEffect(() => {
    if (targetStr === prev.current) return;
    const chars = '0123456789!@#$%^&*<>[]{}';
    const start = Date.now();
    let frame;
    const tick = () => {
      const p = (Date.now() - start) / duration;
      if (p >= 1) {
        setVal(targetStr);
        prev.current = targetStr;
        return;
      }
      let s = '';
      for (let i = 0; i < targetStr.length; i++) {
        if (targetStr[i] === '.' || targetStr[i] === '-') s += targetStr[i];
        else if (p > i / targetStr.length) s += targetStr[i];
        else s += chars[Math.floor(Math.random() * chars.length)];
      }
      setVal(s);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [targetStr, duration]);
  return val;
}

function MetricCard({ label, value, unit, confidence, colorClass }) {
  const precision = colorClass === 'purple' ? 2 : 1;
  const targetStr = (value ?? 0).toFixed(precision);
  const animatedStr = useScramble(targetStr);
  return (
    <div className={`metric-card metric-card--${colorClass}`}
      style={{
        borderColor: colorClass === 'green' ? 'rgba(46,204,113,0.25)'
          : colorClass === 'blue' ? 'rgba(52,152,219,0.25)'
            : colorClass === 'yellow' ? 'rgba(241,196,15,0.25)'
              : 'rgba(155,89,182,0.25)'
      }}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        <span className={`c-${colorClass}`} style={{ fontVariantNumeric: 'tabular-nums' }}>{animatedStr}</span>
        <span className="metric-unit">{unit}</span>
      </div>
      {confidence != null && (
        <div className="metric-conf">± {confidence.toFixed(2)}</div>
      )}
    </div>
  );
}

function CustomTooltip({ active, payload }) {
  if (active && payload?.length) {
    const d = payload[0].payload;
    return (
      <div style={{
        background: 'rgba(5,8,16,0.95)', border: '1px solid rgba(255,255,255,0.15)',
        borderRadius: 10, padding: '10px 14px', fontSize: '0.84rem'
      }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.subject}</div>
        <div style={{ color: '#aaa' }}>Score: <b style={{ color: '#2ecc71' }}>{d.value.toFixed(1)}/100</b></div>
        <div style={{ color: '#aaa' }}>Actual: <b style={{ color: '#fff' }}>{d.raw}</b></div>
      </div>
    );
  }
  return null;
}

function MatchRing({ pct }) {
  const r = 30, circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  const isHighMatch = pct >= 95;
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" className={isHighMatch ? 'match-ring-pulse' : ''}>
      <circle cx="40" cy="40" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="7" />
      <circle cx="40" cy="40" r={r} fill="none" stroke="#00f0ff" strokeWidth="7"
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round" transform="rotate(-90 40 40)"
        style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
      <text x="40" y="44" textAnchor="middle" fill="#00f0ff" fontSize="13" fontWeight="800"
        style={{ textShadow: isHighMatch ? '0 0 8px rgba(0,240,255,0.5)' : 'none' }}>
        {pct}%
      </text>
    </svg>
  );
}

function Toast({ msg, type, onDone }) {
  useEffect(() => { const t = setTimeout(onDone, 2800); return () => clearTimeout(t); }, [onDone]);
  return <div className={`toast toast-${type}`}>{msg}</div>;
}

MetricCard.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.number,
  unit: PropTypes.string.isRequired,
  confidence: PropTypes.number,
  colorClass: PropTypes.string.isRequired,
};

CustomTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.array,
};

MatchRing.propTypes = {
  pct: PropTypes.oneOfType([PropTypes.number, PropTypes.string]).isRequired,
};

Toast.propTypes = {
  msg: PropTypes.string.isRequired,
  type: PropTypes.string.isRequired,
  onDone: PropTypes.func.isRequired,
};

export default function App() {
  const [section, setSection] = useState('hero');
  const [mode, setMode] = useState('polymer'); // 'polymer', 'alloy', 'metal'
  const [values, setValues] = useState(() =>
    Object.fromEntries(POLYMER_SLIDERS.map(s => [s.key, s.default]))
  );
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiOk, setApiOk] = useState(null);
  const [toast, setToast] = useState(null);

  // Green Recommender state
  const [query, setQuery] = useState('');
  const [showSug, setShowSug] = useState(false);
  const [target, setTarget] = useState(null);
  const [alts, setAlts] = useState([]);
  const [recLoading, setRecLoading] = useState(false);
  const [petroleums, setPetroleums] = useState([]);

  useEffect(() => {
    axios.get(`${API}/materials/petroleum`)
      .then(res => setPetroleums(res.data))
      .catch(e => console.error("Failed to load petroleums", e));
  }, []);

  // Poll API health
  useEffect(() => {
    const check = async () => {
      try { await axios.get(`${API}/health`); setApiOk(true); }
      catch { setApiOk(false); }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  // Initialise slider values when mode changes
  useEffect(() => {
    let sliders = POLYMER_SLIDERS;
    if (mode === 'alloy') sliders = ALLOY_SLIDERS;
    if (mode === 'metal') sliders = METAL_SLIDERS;
    setValues(Object.fromEntries(sliders.map(s => [s.key, s.default])));
    setResult(null);
  }, [mode]);

  const buildPayload = useCallback(() => {
    let payload = {};
    let sliders = POLYMER_SLIDERS;
    if (mode === 'alloy') sliders = ALLOY_SLIDERS;
    if (mode === 'metal') sliders = METAL_SLIDERS;

    sliders.forEach(s => {
      payload[s.key] = values[s.key] ?? s.default;
    });

    // Uniform API dispatch: Metal predictions are handled inherently by the backend flag now
    // Actually the backend `/predict` endpoint expects `eco_score` etc.
    // If it's a generic metal, we still send the 7 inputs to `/predict`. Wait, we just need to send `{ inputs: {...}, mode: mode }`
    if (mode === 'metal') {
      payload.is_alloy = 1; // Generic backend flag indicating this is the alloy pipeline
    } else {
      payload.is_alloy = -1; // General polymers
    }
    const mw = parseFloat(values.repeat_unit_MW ?? 72);
    const flex = parseFloat(values.backbone_flexibility ?? 0.4);
    const polar = parseFloat(values.polarity_index ?? 2);
    const hbond = (mode === 'alloy') ? parseFloat(values.crystallinity_tendency ?? 0) * 3.5 : parseFloat(values.hydrogen_bond_capacity ?? 2);

    // Add derived features for polymer/alloy modes
    if (mode !== 'metal') {
      payload.mw_flexibility = mw * flex;
      payload.polar_hbond = polar * hbond;
    }

    return { inputs: payload, mode: mode };
  }, [values, mode]);

  const runPredict = useCallback(async () => {
    setLoading(true);
    const t0 = Date.now();
    try {
      const endpoint = `${API}/predict`;
      const res = await axios.post(endpoint, buildPayload());
      setResult(res.data);
      setToast({ msg: `✅ Ensemble computed in ${Date.now() - t0}ms`, type: 'success' });
    } catch (err) {
      if (err.response && err.response.status === 500 && err.response.data.detail.includes("Model not loaded")) {
        setToast({ msg: '❌ Backend offline — run: make train', type: 'error' });
      } else {
        setToast({ msg: `❌ Prediction failed: ${err.message}`, type: 'error' });
        console.error("Prediction Error:", err);
      }
    } finally { setLoading(false); }
  }, [buildPayload]);

  const sliders = mode === 'metal' ? METAL_SLIDERS : (mode === 'alloy' ? ALLOY_SLIDERS : POLYMER_SLIDERS);

  let radarData = [];
  if (result && result.predictions) {
    if (result.predictions['tensile_strength_MPa'] !== undefined) {
      const norms = mode === 'metal' ? METAL_RADAR_NORMS : (mode === 'alloy' ? ALLOY_RADAR_NORMS : POLYMER_RADAR_NORMS);

      const rawPcts = Object.keys(norms).map(k => {
        const val = result.predictions[k] || 0;
        const n = norms[k] || { max: 100 };
        const mn = n.min ?? 0;
        return Math.max(0, ((val - mn) / (n.max - mn)) * 100);
      });

      const maxPct = Math.max(100, ...rawPcts);

      radarData = Object.keys(norms).map((k, i) => {
        return {
          subject: norms[k].label,
          value: (rawPcts[i] / maxPct) * 100, // Dynamically scale down if Max > 100%
          raw: `${(result.predictions[k] || 0).toFixed(1)} ${k.includes('MPa') ? 'MPa' : k.includes('GPa') ? 'GPa' : ''}`
        };
      });
    }
  }

  const findAlts = async () => {
    if (!query) return;
    setRecLoading(true);
    try {
      const name = PETROLEUM_MATERIALS.find(m => m.toLowerCase() === query.toLowerCase()) ?? query;
      const res = await axios.get(`${API}/materials/alternatives/${encodeURIComponent(name)}`);
      setAlts(res.data);
      const t2 = await axios.get(`${API}/materials/petroleum`);
      const match = t2.data.find(m => m.material_name.toLowerCase() === name.toLowerCase());
      setTarget(match ?? { material_name: name, eco_score: 0 });
      setToast({ msg: `✅ Found ${res.data.length} green alternatives`, type: 'success' });
    } catch {
      setToast({ msg: '❌ Material not found or backend offline', type: 'error' });
    } finally { setRecLoading(false); }
  };

  return (
    <div className="app">
      {/* Orb Background */}
      <div className="fixed-orb-bg">
        <Orb hoverIntensity={2} rotateOnHover hue={0} backgroundColor="#050810" />
      </div>

      {/* Status pill */}
      <div className="status-pill">
        <div className={`dot ${apiOk ? 'dot-green' : 'dot-red'}`} />
        {apiOk == null ? 'Checking…' : apiOk ? 'API Connected' : 'API Offline'}
      </div>

      {/* Nav */}
      <div className="section-nav">
        {['hero', 'predict', 'recommend', 'about'].map(s => (
          <button key={s} className={`nav-pill ${section === s ? 'active' : ''}`}
            onClick={() => setSection(s)}>
            {s === 'hero' ? 'Home' : s === 'predict' ? 'Predictor' :
              s === 'recommend' ? '🌱 Green' : 'About'}
          </button>
        ))}
      </div>

      {/* Toast */}
      {toast && <Toast msg={toast.msg} type={toast.type} onDone={() => setToast(null)} />}

      {/* ── HERO ── */}
      {section === 'hero' && (
        <div className="hero">
          <div style={{ fontSize: '3rem', marginBottom: '-12px' }}>🌿</div>
          <h1 className="hero-title">Eco-Material<br />Innovation Engine</h1>
          <p className="hero-sub">
            Predicting Next-Generation Sustainable Materials with AI Ensembles.<br />
            R² &gt; 0.90 across 10 properties. 285 materials. Zero lab tests required.
          </p>
          <div className="hero-btns">
            <button className="btn btn-green" onClick={() => setSection('predict')}>🔬 Predict Properties</button>
            <button className="btn btn-blue" onClick={() => setSection('recommend')}>🌱 Find Green Alternatives</button>
          </div>
          <div style={{ marginTop: 48, display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, maxWidth: 560 }}>
            {[['285', 'Materials'], ['10', 'Properties Predicted'], ['0.90+', 'Average R²']].map(([n, l]) => (
              <Tilt key={l} glareEnable={true} glareMaxOpacity={0.15} glareColor="#2ecc71" glarePosition="all" tiltMaxAngleX={10} tiltMaxAngleY={10} scale={1.05} transitionSpeed={2000}>
                <div className="card" style={{ textAlign: 'center', padding: '24px 16px', height: '100%' }}>
                  <div style={{ fontSize: '2rem', fontWeight: 900, color: '#2ecc71' }}>{n}</div>
                  <div style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.45)', marginTop: 4 }}>{l}</div>
                </div>
              </Tilt>
            ))}
          </div>
        </div>
      )}

      {/* ── PREDICTOR ── */}
      {section === 'predict' && (
        <div className="section" style={{ paddingTop: 100 }}>
          <h2 className="section-title">🔬 Property Predictor</h2>
          <p className="section-sub">Adjust the sliders to define a material — the AI stacked ensemble predicts 10 properties instantly.</p>

          <div className="predictor-grid">
            {/* LEFT */}
            <Tilt glareEnable={true} glareMaxOpacity={0.1} glarePosition="all" tiltMaxAngleX={3} tiltMaxAngleY={3} transitionSpeed={2500}>
              <div className="card" style={{ height: '100%' }}>
                <div className="mode-header">
                  <h3 style={{ fontSize: '1rem', fontWeight: 800 }}>Material Definition</h3>
                  <div style={{ display: 'flex', gap: '6px', marginTop: 12 }}>
                    <button className={`mode-btn ${mode === 'polymer' ? 'mode-btn-green' : ''}`}
                      onClick={() => { setMode('polymer'); setResult(null); }} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                      🌿 Polymer
                    </button>
                    <button className={`mode-btn ${mode === 'alloy' ? 'mode-btn-blue' : ''}`}
                      onClick={() => { setMode('alloy'); setResult(null); }} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                      ✦ Synthetic Alloy
                    </button>
                    <button className={`mode-btn ${mode === 'metal' ? 'mode-btn-orange' : ''}`}
                      onClick={() => { setMode('metal'); setResult(null); }} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                      ⚙️ All Metals
                    </button>
                  </div>
                </div>

                {sliders.map(s => (
                  <div className="slider-row" key={s.key}>
                    <div className="slider-label">
                      <span>{s.label}</span>
                      <span className="slider-val">
                        {s.step < 1 ? (values[s.key] ?? s.default).toFixed(2)
                          : Math.round(values[s.key] ?? s.default)}
                      </span>
                    </div>
                    <input type="range" min={s.min} max={s.max} step={s.step}
                      className={mode === 'metal' ? 'orange-slider' : (mode === 'alloy' ? 'blue-slider' : 'green-slider')}
                      value={values[s.key] ?? s.default}
                      style={{
                        background: `linear-gradient(to right, ${mode === 'metal' ? '#e67e22' : (mode === 'alloy' ? '#3498db' : '#2ecc71')} ${(((values[s.key] ?? s.default) - s.min) / (s.max - s.min)) * 100
                          }%, rgba(255,255,255,0.1) 0)`
                      }}
                      onChange={e => setValues(v => ({ ...v, [s.key]: parseFloat(e.target.value) }))} />
                  </div>
                ))}

                <button className="predict-btn" onClick={runPredict} disabled={loading}>
                  {loading ? <><div className="spinner" />&nbsp;Computing…</> : '⚡ Run Ensemble Prediction'}
                </button>
              </div>
            </Tilt>

            {/* RIGHT */}
            <div className="right-col">

              {/* ── Radar chart card ── */}
              <Tilt glareEnable={true} glareMaxOpacity={0.08} tiltMaxAngleX={4} tiltMaxAngleY={4} transitionSpeed={2500}>
                <div className="card">
                  <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: 16 }}>
                    🕸 Material DNA — Radar Profile
                  </h3>
                  {result ? (
                    <>
                      <ResponsiveContainer width="100%" height={280}>
                        <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                          <PolarGrid stroke="rgba(255,255,255,0.12)" />
                          <PolarAngleAxis dataKey="subject"
                            tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11, fontWeight: 600 }} />
                          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
                          <Tooltip content={<CustomTooltip />} />
                          <Radar name="Material" dataKey="value" dot
                            stroke={mode === 'metal' ? '#e67e22' : (mode === 'alloy' ? '#3498db' : '#2ecc71')}
                            fill={mode === 'metal' ? '#e67e22' : (mode === 'alloy' ? '#3498db' : '#2ecc71')}
                            fillOpacity={0.55} />
                        </RadarChart>
                      </ResponsiveContainer>
                      <div style={{ textAlign: 'center', marginTop: 8, fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)' }}>
                        <span style={{ color: '#2ecc71', fontWeight: 'bold' }}>Score:</span> Normalized 0-100 UI Scale &nbsp;|&nbsp; <span style={{ color: '#fff', fontWeight: 'bold' }}>Actual:</span> Real Scientific Value
                      </div>
                    </>
                  ) : (
                    <div className="chart-placeholder">
                      Adjust sliders and hit<br /><b>⚡ Run Ensemble Prediction</b><br />to visualise the Material DNA
                    </div>
                  )}
                </div>
              </Tilt>

              {/* ── Key metric cards ── */}
              {result ? (
                <div className="card">
                  <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: 16 }}>📊 Key ML Predictions</h3>
                  <div style={{ marginBottom: 16, padding: '12px 16px', background: 'rgba(255, 235, 59, 0.1)', borderLeft: '4px solid #f1c40f', borderRadius: 4 }}>
                    <div style={{ fontSize: '0.8rem', color: '#f1c40f', fontWeight: 800, textTransform: 'uppercase', marginBottom: 4 }}>💡 Ideal Application</div>
                    <div style={{ fontSize: '0.95rem', color: '#fff', fontStyle: 'italic' }}>
                      {(() => {
                        const { tensile_strength_MPa: tens = 0, Tg_celsius: tg = 0, density_gcm3: dens = 5.0 } = result.predictions;
                        if (mode === "polymer") {
                          if (tens > 80 && tg > 150) return "Aerospace & Automotive inner parts (High strength, thermal resistant)";
                          if (tens > 40 && tg > 60) return "Rigid packaging & Consumer electronics (Durable, moderate strength)";
                          if (tens < 20 && tg < 30) return "Flexible films & Eco-friendly bags (Soft, high elongation)";
                          return "General purpose commercial plastics and molding";
                        } else {
                          if (tens > 1000 && dens < 5.0) return "Aerospace structures & Advanced lightweight vehicle frames";
                          if (tens > 1000) return "Heavy-duty construction, Industrial tooling, Structural supports";
                          if (tens < 500 && dens < 3.0) return "Lightweight consumer electronics, Bike frames, Aviation panels";
                          return "General structural, marine, and architectural metal applications";
                        }
                      })()}
                    </div>
                  </div>
                  <div className="metrics-grid">
                    <>
                      <MetricCard label="Tensile Strength" colorClass="green"
                        value={result.predictions.tensile_strength_MPa}
                        confidence={result.confidence.tensile_strength_MPa}
                        unit="MPa" />
                      <MetricCard label="Glass Transition Tg" colorClass="blue"
                        value={result.predictions.Tg_celsius}
                        confidence={result.confidence.Tg_celsius}
                        unit="°C" />
                      <MetricCard label="Young's Modulus" colorClass="yellow"
                        value={result.predictions.youngs_modulus_GPa}
                        confidence={result.confidence.youngs_modulus_GPa}
                        unit="GPa" />
                      <MetricCard label="Density" colorClass="purple"
                        value={result.predictions.density_gcm3}
                        confidence={result.confidence.density_gcm3}
                        unit="g/cm³" />
                    </>
                  </div>
                </div>
              ) : (
                <div className="card" style={{
                  display: 'flex', alignItems: 'center',
                  justifyContent: 'center', minHeight: 140, color: 'rgba(255,255,255,0.3)',
                  fontStyle: 'italic', fontSize: '0.9rem'
                }}>
                  Run a prediction to see results
                </div>
              )}

              {/* ── Full properties table ── */}
              {result && (
                <div className="card">
                  <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: 4 }}>📋 All 10 Properties</h3>
                  <table className="props-table">
                    <thead>
                      <tr><th>Property</th><th>Value</th><th>Unit</th><th>±Conf</th></tr>
                    </thead>
                    <tbody>
                      {[
                        ['Tensile Strength', 'tensile_strength_MPa', 'MPa'],
                        ['Glass Transition Tg', 'Tg_celsius', '°C'],
                        ["Young's Modulus", 'youngs_modulus_GPa', 'GPa'],
                        ['Density', 'density_gcm3', 'g/cm³'],
                        ['Thermal Conductivity', 'thermal_conductivity_WmK', 'W/m·K'],
                        ['Elec. Conductivity', 'log10_elec_conductivity', 'log₁₀ S/m'],
                        ['Elongation at Break', 'elongation_at_break_pct', '%'],
                        ['Dielectric Constant', 'dielectric_constant', '—'],
                        ['Water Absorption', 'water_absorption_pct', '%'],
                        ['O₂ Permeability', 'oxygen_permeability_barrer', 'Barrers'],
                      ].map(([name, key, unit]) => (
                        <tr key={key}>
                          <td style={{ color: 'rgba(255,255,255,0.7)' }}>{name}</td>
                          <td style={{ fontWeight: 700 }}>{result.predictions[key] !== undefined ? result.predictions[key].toFixed(2) : '-'}</td>
                          <td style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.78rem' }}>{unit}</td>
                          <td style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.78rem' }}>
                            ±{result.confidence && result.confidence[key] !== undefined ? result.confidence[key].toFixed(2) : '0.00'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

            </div>
          </div>
        </div>
      )}

      {/* ── GREEN RECOMMENDER ── */}
      {section === 'recommend' && (
        <div className="recommender-section" style={{ paddingTop: 100 }}>
          <h2 className="section-title">🌱 Green Alternative Recommender</h2>
          <p className="section-sub">
            Find bio-based materials that match the structural performance of petroleum plastics — with a 5× better eco-score.
          </p>

          <div className="card">
            <div className="search-row">
              <div className="search-wrap" style={{ position: 'relative' }}>
                <input className="search-input" value={query}
                  placeholder="Select or type a petroleum/toxic material to replace..."
                  onChange={e => { setQuery(e.target.value); setShowSug(true); }}
                  onFocus={() => setShowSug(true)}
                  onBlur={() => setTimeout(() => setShowSug(false), 200)}
                  style={{ backgroundColor: '#111', color: 'white', padding: '12px', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '6px', fontSize: '1rem', width: '100%', outline: 'none' }} />
                <div style={{ position: 'absolute', right: 16, top: 16, pointerEvents: 'none', color: 'rgba(255,255,255,0.5)', fontSize: '0.8rem' }}>▼</div>
                {showSug && (
                  <div className="search-suggestions" style={{ position: 'absolute', top: '100%', left: 0, right: 0, maxHeight: '300px', overflowY: 'auto', backgroundColor: '#111', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '6px', zIndex: 50, marginTop: '4px', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }}>
                    {(petroleums.length > 0 ? petroleums : [{ material_name: 'Steel (Conventional)', eco_score: 0.2 }, ...PETROLEUM_MATERIALS.map(name => ({ material_name: name, eco_score: 0.0 }))])
                      .filter(m => m.material_name.toLowerCase().includes(query.toLowerCase()))
                      .map(m => (
                        <div key={m.material_name}
                          style={{ padding: '12px 16px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                          onMouseDown={() => { setQuery(m.material_name); setShowSug(false); }}
                          onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)'}
                          onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                          <span style={{ color: 'white' }}>{m.material_name}</span>
                          <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.85rem' }}>Eco: {m.eco_score.toFixed(2)}</span>
                        </div>
                      ))}
                  </div>
                )}
              </div>
              <button className="find-btn" onClick={findAlts} disabled={recLoading || !query}>
                {recLoading ? 'Searching…' : '🌱 Find Alternatives'}
              </button>
            </div>
          </div>

          {target && (
            <div className="target-panel">
              <div className="target-name">{target.material_name}</div>
              <div className="props-badges">
                <span className="badge badge-red">⚠ Eco-Score: {(target.eco_score ?? 0).toFixed(2)}</span>
                {target.tensile_strength_MPa && <span className="badge badge-gray">Tensile: {target.tensile_strength_MPa.toFixed(1)} MPa</span>}
                {target.Tg_celsius && <span className="badge badge-gray">Tg: {target.Tg_celsius.toFixed(1)}°C</span>}
              </div>
            </div>
          )}

          {alts.length > 0 && (
            <div className="alts-grid">
              {alts.map((alt, i) => {
                const baseTensile = target?.tensile_strength_MPa ?? alt.tensile_strength_MPa;
                const diff = ((alt.tensile_strength_MPa - baseTensile) / baseTensile * 100).toFixed(1);
                const medals = ['🥇', '🥈', '🥉'];
                const matchPct = alt.match_pct ?? 0;
                return (
                  <div className="alt-card" key={alt.material_name}>
                    <div className="alt-rank">{medals[i]}</div>
                    <div className="match-ring-wrap">
                      <MatchRing pct={parseFloat(matchPct)} />
                    </div>
                    <div className="alt-props">
                      <div className="alt-prop">
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 16 }}>
                          <div style={{ fontSize: '1.2rem' }}>💎</div>
                          <div>
                            <div style={{ color: '#00f0ff', fontWeight: 800, fontSize: '1.1rem', marginBottom: 4 }}>
                              {alt.material_name}
                            </div>
                            <div style={{ fontSize: '0.85rem', color: '#f1c40f', fontStyle: 'italic', marginBottom: 8 }}>
                              💡 {(() => {
                                const isMetal = alt.material_class === "metal" || alt.material_class === "alloy";
                                const tens = alt.tensile_strength_MPa || 0;
                                const tg = alt.Tg_celsius || 0;
                                const dens = alt.density_gcm3 || 5.0;
                                if (!isMetal) {
                                  if (tens > 80 && tg > 150) return "Aerospace & Automotive inner parts (High strength, thermal resistant)";
                                  if (tens > 40 && tg > 60) return "Rigid packaging & Consumer electronics (Durable, moderate strength)";
                                  if (tens < 20 && tg < 30) return "Flexible films & Eco-friendly bags (Soft, high elongation)";
                                  return "General purpose commercial plastics and molding";
                                } else {
                                  if (tens > 1000 && dens < 5.0) return "Lightweight Aerospace structures & Advanced vehicle frames";
                                  if (tens > 1000) return "Heavy-duty construction, Industrial tooling, Structural supports";
                                  if (tens < 500 && dens < 3.0) return "Lightweight consumer electronics, Bike frames, Aviation panels";
                                  return "General structural, marine, and architectural metal applications";
                                }
                              })()}
                            </div>
                            <div style={{ fontSize: '0.8rem', color: '#aaa', display: 'flex', flexDirection: 'column', gap: 4 }}>
                              <span style={{ color: '#00f0ff' }}>Eco-Score: <b>{alt.eco_score.toFixed(2)}</b></span>
                              <span>Tensile: <b>{alt.tensile_strength_MPa.toFixed(1)}</b> MPa <small style={{ color: parseFloat(diff) > 0 ? '#e74c3c' : '#00f0ff' }}>{parseFloat(diff) > 0 ? '▼' : '▲'} {Math.abs(parseFloat(diff)).toFixed(1)}%</small></span>
                              <span>Tg: <b>{alt.Tg_celsius.toFixed(1)}</b> °C</span>
                              <span>Density: <b>{alt.density_gcm3.toFixed(2)}</b> g/cm³</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ── ABOUT ── */}
      {section === 'about' && (
        <div className="about-section">
          <h2 className="section-title" style={{ marginBottom: 24 }}>About This Project</h2>
          <div className="card" style={{ marginBottom: 24 }}>
            <p>
              This AI engine uses a <strong>two-layer Stacked Ensemble</strong> — Random Forest + XGBoost as base learners,
              then a Ridge Regression meta-learner — trained on <strong>285 materials</strong> generated from
              scientifically-grounded QSPR (Quantitative Structure-Property Relationship) formulas with 2% realistic lab noise.
            </p>
            <p>
              Accuracy is never faked: <strong>20% of all materials were locked away</strong> in a blind test vault the AI was
              never allowed to see during training. Every R² score comes from those held-out samples.
            </p>
            <p>
              Two separate ensemble brains are maintained: one for <strong>Polymers</strong> and one for <strong>Metal Alloys</strong>,
              automatically routed based on material class.
            </p>
          </div>
          <div className="stat-grid">
            {[['R² > 0.90', 'Across all 10 properties'], ['285', 'Curated materials'], ['27/27', 'Tests passing']].map(([n, l]) => (
              <div key={l} className="card stat-card">
                <div className="stat-num">{n}</div>
                <div className="stat-label">{l}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
