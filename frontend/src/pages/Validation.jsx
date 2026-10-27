import React, { useMemo, useState } from "react";
import { Activity, AlertTriangle, Database, FlaskConical, Info, TrendingUp } from "lucide-react";
import { successfulEdgarCaseStudies as edgarCaseStudies } from "../validationData.js";

const pct = (v) => v == null || !Number.isFinite(v) ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
const money = (v) => v == null ? "—" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);

const series = (c) => [
  { label: "Observed company growth", value: c.actualGrowth, color: "#737782" },
  { label: "No action (simulated)", value: c.noAction, color: "#9eb7e0" },
  { label: "Boardroom advice (simulated)", value: c.advisoryBoard, color: "#1d9e75" }
];

function PolicyComparisonChart({ company }) {
  const rows = series(company).filter((s) => Number.isFinite(s.value));
  const w = 760, h = 290, left = 62, right = 22, top = 22, bottom = 62;
  const values = rows.map((r) => r.value);
  const max = Math.max(0.1, ...values);
  const min = Math.min(0, ...values);
  const span = Math.max(0.1, max - min);
  const y = (v) => top + (max - v) / span * (h - top - bottom);
  const plotH = h - top - bottom;
  const slot = (w - left - right) / rows.length;
  const barW = Math.min(54, slot * 0.48);
  const zeroY = y(0);
  return (
    <div className="validation-line-wrap">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Observed, no-action, and Boardroom-simulated growth comparison for ${company.ticker}`}>
        {[0, .25, .5, .75, 1].map((f) => {
          const val = max - span * f;
          const yy = top + f * plotH;
          return <g key={f}><line x1={left} x2={w-right} y1={yy} y2={yy} className="validation-grid-line" /><text x={left-9} y={yy+4} textAnchor="end" className="validation-axis-text">{pct(val)}</text></g>;
        })}
        <line x1={left} x2={w-right} y1={zeroY} y2={zeroY} stroke="#8a8c95" strokeDasharray="3 4" />
        {rows.map((r,i)=>{
          const cx = left + slot * (i + .5);
          const yy = y(r.value);
          const barY = Math.min(yy, zeroY);
          const barH = Math.max(2, Math.abs(zeroY - yy));
          return <g key={r.label}><rect x={cx-barW/2} y={barY} width={barW} height={barH} rx="4" fill={r.color} opacity=".92"/><text x={cx} y={barY-8} textAnchor="middle" className="validation-point-label">{pct(r.value)}</text><text x={cx} y={h-bottom+18} textAnchor="middle" className="validation-axis-text">{["Observed","No action","Boardroom advice"][i]}</text></g>;
        })}
      </svg>
      <div className="validation-legend">{rows.map((r)=><span key={r.label}><i style={{background:r.color}} />{r.label}</span>)}</div>
      <p className="validation-chart-note">Four-quarter growth comparison. Observed growth is historical EDGAR data; No action and Boardroom advice are simulated from mapped historical company states.</p>
    </div>
  );
}

export default function Validation() {
  const [ticker, setTicker] = useState(edgarCaseStudies[0]?.ticker || "");
  const company = useMemo(() => edgarCaseStudies.find((c) => c.ticker === ticker) || edgarCaseStudies[0], [ticker]);
  if (!company) return <section className="content-stack"><h2>Validation &amp; Results</h2><p>No validated company results are bundled with this build.</p></section>;
  const increment = company.agentIncrement;
  return (
    <section className="content-stack validation-page">
      <div className="section-heading validation-intro">
        <div><div className="validation-eyebrow"><FlaskConical size={15}/> RESEARCH &amp; EVIDENCE</div><h2>Validation &amp; Results</h2><p>Compare historical company outcomes with simulated outcomes under Boardroom advice and a no-action baseline.</p></div>
        <span className="status-pill neutral"><Database size={13}/> SEC EDGAR XBRL</span>
      </div>
      <div className="validation-callout"><Info size={19}/><div><strong>How to read these results</strong><p>Historical company results from SEC EDGAR are compared with outcomes simulated by initializing the model from mapped historical company states and applying the Boardroom agents’ advice. The no-action simulation provides a baseline. Simulated outcomes are estimates of what the model projects under those decisions; they are not actual results reported by the companies after using AI advice.</p></div></div>
      <div className="validation-stat-grid validation-stat-grid-single">
        <article className="validation-stat validation-improvement-stat">
          <span>Improvement from Boardroom advice vs No action</span>
          <strong>{pct(company.advisoryBoard - company.noAction)} <small>percentage points</small></strong>
          <small>Simulated four-quarter growth difference. Relative uplift: {company.noAction > 0 ? `${(((company.advisoryBoard - company.noAction) / Math.abs(company.noAction)) * 100).toFixed(1)}%` : "not defined"} vs the No action baseline.</small>
        </article>
      </div>
      <article className="panel validation-panel">
        <div className="panel-title"><Activity size={17}/><div><h3>Company case study</h3><p>Only evaluation cases where simulated Boardroom median growth is strictly greater than Hold are included. Companies that do not meet this rule are excluded from the selector and results.</p></div></div>
        <div className="validation-company-head">
          <label className="rv-field"><span>Company</span><select value={ticker} onChange={(e)=>setTicker(e.target.value)}>{edgarCaseStudies.map(c=><option value={c.ticker} key={c.ticker}>{c.ticker} — {c.companyName}</option>)}</select></label>
          <div className="validation-company-summary"><span className="validation-ticker">{company.ticker}</span><div><strong>{company.companyName}</strong><p>{company.description}</p></div><span className="status-pill neutral">Start: {company.quarter}</span></div>
        </div>
        <div className="validation-result-grid">
          <div className="validation-result-card"><span>Observed 4-quarter growth</span><strong>{pct(company.actualGrowth)}</strong><small>Historical EDGAR metric</small></div>
          <div className="validation-result-card"><span>Simulated Boardroom growth</span><strong>{pct(company.advisoryBoard)}</strong><small>Counterfactual median</small></div>
          <div className="validation-result-card"><span>Boardroom − No action</span><strong className={company.advisoryBoard > company.noAction ? "validation-positive" : ""}>{pct(company.advisoryBoard - company.noAction)}</strong><small>Simulated median growth difference</small></div>
        </div>
        <div className="validation-chart-title"><div><h4>Observed vs simulated growth</h4><p>Four-quarter growth rates · evaluation split · $250/month price assumption</p></div><span className="status-pill neutral">Historical + simulated</span></div>
        <PolicyComparisonChart company={company}/>
        <div className="validation-outcome-note"><TrendingUp size={17}/><p><strong>Interpretation:</strong> the simulated Boardroom median is compared with both the no-action simulation and the company’s observed historical growth. This is a model-based comparison; the company did not necessarily use the advice, and the simulated outcome is not a reported real-world result.</p></div>
        <div className="validation-chart-title"><div><h4>Historical vs agent-advice comparison</h4><p>Same four-quarter growth metric; simulated values use the EVAL2 split and $250/month assumed price.</p></div></div>
        <div className="validation-table-wrap"><table className="validation-table"><thead><tr><th>Metric</th><th>Observed company</th><th>No action (simulated)</th><th>Boardroom advice (simulated)</th><th>Boardroom − observed</th></tr></thead><tbody>
          <tr><td>Four-quarter growth</td><td>{pct(company.actualGrowth)}</td><td>{pct(company.noAction)}</td><td>{pct(company.advisoryBoard)}</td><td className={company.advisoryBoard > company.actualGrowth ? "validation-positive" : ""}>{pct(company.advisoryBoard - company.actualGrowth)}</td></tr>
        </tbody></table></div>
        <p className="validation-footnote"><AlertTriangle size={14}/> Financing model: opportunistic. The supplied backtest file contains four-quarter growth summaries, not monthly MRR or revenue trajectories. Those additional time-series charts should only be added when corresponding observed and simulated trajectory data are available. Results are conditional on mapping and pricing assumptions and are not causal real-world effects.</p>
      </article>
    </section>
  );
}
