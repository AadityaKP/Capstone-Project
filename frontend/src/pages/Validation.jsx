import React, { useMemo, useState } from "react";
import { Activity, AlertTriangle, Database, FlaskConical, Info, TrendingUp } from "lucide-react";
import { successfulEdgarCaseStudies as edgarCaseStudies } from "../validationData.js";

const pct = (v) => v == null || !Number.isFinite(v) ? "â€”" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
const money = (v) => v == null ? "â€”" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);

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


function MonthlyMRRChart({ company }) {
  const [trajectories, setTrajectories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  React.useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(
          "/real_company_monthly_trajectories_eval2.csv"
        );
        if (!response.ok) {
          throw new Error(`Trajectory data request failed (${response.status}).`);
        }

        const csv = await response.text();
        const lines = csv.trim().split(/\r?\n/);
        if (lines.length < 2) {
          throw new Error("The trajectory file contains no data rows.");
        }

        const headers = lines[0].split(",");
        const index = Object.fromEntries(headers.map((h, i) => [h, i]));
        const required = [
          "ticker", "init_quarter", "price_assumed", "financing_model",
          "financing", "policy", "seed", "month", "simulated_mrr"
        ];

        for (const field of required) {
          if (index[field] === undefined) {
            throw new Error(`Missing trajectory column: ${field}`);
          }
        }

        const rows = lines.slice(1).map((line) => {
          const cells = line.split(",");
          return Object.fromEntries(headers.map((h, i) => [h, cells[i]]));
        });

        const matching = rows.filter((r) =>
          r.ticker === company.ticker &&
          r.init_quarter === company.quarter &&
          Number(r.price_assumed) === Number(company.priceAssumed) &&
          r.financing_model === "opportunistic" &&
          r.financing === "True" &&
          ["noop", "boardroom"].includes(r.policy)
        );

        if (!matching.length) {
          throw new Error("No trajectories match this company and evaluation configuration.");
        }

        if (!cancelled) setTrajectories(matching);
      } catch (err) {
        if (!cancelled) setError(err.message || "Unable to load trajectory data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [company.ticker, company.quarter, company.priceAssumed]);

  if (loading) return <p>Loading monthly simulated MRR trajectories?</p>;
  if (error) return <p className="validation-chart-note">{error}</p>;

  const policies = [
    { key: "noop", label: "No action (simulated)", color: "#9eb7e0" },
    { key: "boardroom", label: "Boardroom advice (simulated)", color: "#1d9e75" }
  ];

  const medians = policies.map((policy) => {
    const points = [];
    for (let month = 1; month <= 12; month++) {
      const values = trajectories
        .filter((r) => r.policy === policy.key && Number(r.month) === month)
        .reduce((acc, r) => {
          const value = Number(r.simulated_mrr);
          if (Number.isFinite(value)) acc.push(value);
          return acc;
        }, [])
        .sort((a, b) => a - b);

      if (values.length) {
        const mid = Math.floor(values.length / 2);
        points.push({
          month,
          value: values.length % 2
            ? values[mid]
            : (values[mid - 1] + values[mid]) / 2
        });
      }
    }
    return { ...policy, points };
  });

  const allValues = medians.flatMap((p) => p.points.map((v) => v.value));
  if (!allValues.length) return <p>No revenue or simulated MRR values are available for this case.</p>;

  const w = 760, h = 310, left = 92, right = 24, top = 24, bottom = 48;
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const span = Math.max(1, max - min);
  const x = (month) => left + ((month - 1) / 11) * (w - left - right);
  const y = (value) => top + ((max - value) / span) * (h - top - bottom);
  const moneyShort = (value) => {
    const abs = Math.abs(value);
    if (abs >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
    return `$${value.toFixed(0)}`;
  };

  return (
    <div className="validation-line-wrap">
      <svg viewBox={`0 0 ${w} ${h}`} role="img"
        aria-label={`Historical quarterly revenue and simulated monthly MRR for ${company.ticker}, starting ${company.quarter}`}>
        {[0, .25, .5, .75, 1].map((f) => {
          const value = max - span * f;
          const yy = top + f * (h - top - bottom);
          return <g key={f}>
            <line x1={left} x2={w - right} y1={yy} y2={yy}
              className="validation-grid-line" />
            <text x={left - 10} y={yy + 4} textAnchor="end"
              className="validation-axis-text">{moneyShort(value)}</text>
          </g>;
        })}
         {[1, 3, 5, 7, 9, 11, 12].map((month) =>
          <text key={month} x={x(month)} y={h - 18} textAnchor="middle"
            className="validation-axis-text">{month}</text>
        )}
        {medians.map((policy) => {
          const path = policy.points.map((point, i) =>
            `${i === 0 ? "M" : "L"} ${x(point.month)} ${y(point.value)}`
          ).join(" ");
          return <g key={policy.key}>
            <path d={path} fill="none" stroke={policy.color}
              strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
            {policy.points.map((point) =>
              <circle key={point.month} cx={x(point.month)} cy={y(point.value)}
                r="3.2" fill={policy.color} />
            )}
          </g>;
        })}
      </svg>
      <div className="validation-legend">
         {policies.map((p) => <span key={p.key}>
          <i style={{ background: p.color }} />{p.label}
        </span>)}
      </div>
      <p className="validation-chart-note">
        Simulated monthly MRR in USD, shown as the median across available seeds.
        No action and Boardroom advice are simulated policy outcomes,
        not actual reported company revenue.
      </p>
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
      <div className="validation-callout"><Info size={19}/><div><strong>How to read these results</strong><p>We compare historical company growth reported to SEC EDGAR with simulated outcomes based on mapped historical company data. Boardroom advice and no action are simulated scenarios, not actual results reported by the companies.</p></div></div>
      <div className="validation-stat-grid validation-stat-grid-single">
        <article className="validation-stat validation-improvement-stat">
          <span>Improvement from Boardroom advice vs No action</span>
          <strong>{pct(company.advisoryBoard - company.noAction)} <small>percentage points</small></strong>
          <small>Simulated four-quarter growth difference. Relative uplift: {company.noAction > 0 ? `${(((company.advisoryBoard - company.noAction) / Math.abs(company.noAction)) * 100).toFixed(1)}%` : "not defined"} vs the No action baseline.</small>
        </article>
      </div>
      <article className="panel validation-panel">
        <div className="panel-title"><Activity size={17}/><div><h3>Company case study</h3><p>Only evaluation cases where simulated Boardroom median growth is strictly greater than the No action baseline are included. Companies that do not meet this rule are excluded from the selector and results.</p></div></div>
        <div className="validation-company-head">
          <label className="rv-field"><span>Company</span><select value={ticker} onChange={(e)=>setTicker(e.target.value)}>{edgarCaseStudies.map(c=><option value={c.ticker} key={c.ticker}>{c.ticker} - {c.companyName}</option>)}</select></label>
          <div className="validation-company-summary"><span className="validation-ticker">{company.ticker}</span><div><strong>{company.companyName}</strong><p>{company.description}</p></div><span className="status-pill neutral">Start: {company.quarter}</span></div>
        </div>
        <div className="validation-result-grid">
          <div className="validation-result-card"><span>Observed 4-quarter growth</span><strong>{pct(company.actualGrowth)}</strong><small>Historical EDGAR metric</small></div>
          <div className="validation-result-card"><span>Simulated Boardroom growth</span><strong>{pct(company.advisoryBoard)}</strong><small>Counterfactual median</small></div>
          <div className="validation-result-card"><span>Boardroom - No action</span><strong className={company.advisoryBoard > company.noAction ? "validation-positive" : ""}>{pct(company.advisoryBoard - company.noAction)}</strong><small>Simulated median growth difference</small></div>
        </div>
        <div className="validation-chart-title"><div><h4>Observed vs simulated growth</h4><p>Four-quarter growth rates Â· evaluation split Â· $250/month price assumption</p></div><span className="status-pill neutral">Historical + simulated</span></div>
        <PolicyComparisonChart company={company}/>
        <div className="validation-chart-title"><div><h4>Monthly simulated MRR trajectories</h4><p>12-month simulation ? same company, quarter, price, and financing configuration</p></div><span className="status-pill neutral">Simulated MRR</span></div>
        <MonthlyMRRChart company={company} />
        <div className="validation-outcome-note"><TrendingUp size={17}/><p><strong>Interpretation:</strong> the simulated Boardroom is compared with the no-action simulation. This is a model-based comparison; the company did not necessarily use the advice, and the simulated outcome is not a reported real-world result.</p></div>
        <div className="validation-chart-title"><div><h4>Historical vs agent-advice comparison</h4><p>Same four-quarter growth metric; simulated values use the EVAL2 split and $250/month assumed price.</p></div></div>
        <div className="validation-table-wrap"><table className="validation-table"><thead><tr><th>Metric</th><th>Observed company</th><th>No action (simulated)</th><th>Boardroom - Observed</th></tr></thead><tbody>
          <tr><td>Four-quarter growth</td><td>{pct(company.actualGrowth)}</td><td>{pct(company.noAction)}</td><td>{pct(company.advisoryBoard)}</td><td className={company.advisoryBoard > company.actualGrowth ? "validation-positive" : ""}>{pct(company.advisoryBoard - company.actualGrowth)}</td></tr>
        </tbody></table></div>
        <p className="validation-footnote"><AlertTriangle size={14}/> Financing model: opportunistic. Historical EDGAR data supplies four-quarter revenue growth; monthly MRR trajectories shown above are simulated counterfactuals. They must not be interpreted as observed company revenue or causal real-world effects. Results are conditional on mapping and pricing assumptions and are not causal real-world effects.</p>
      </article>
    </section>
  );
}
