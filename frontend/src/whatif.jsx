// S? What-if — twelve-month projection under three policies (defect D5).
//
// Until this existed, nothing in the product showed what *happens* if the
// founder takes the board's plan: the advise path calls Boardroom.decide() once
// and never steps the environment. This panel is the missing half.
//
// It renders three policies over the same horizon, the same seeds and the same
// shock tape, as a median line with a 25–75 interquartile band. The caveat sits
// directly under the chart, not in a footnote, because the whole panel is a
// simulated counterfactual and a reader who misses that misreads everything.

import React, { useState } from "react";
import { Activity, AlertTriangle, Loader2, Zap } from "lucide-react";
import { money, pct } from "./derive.js";

const POLICY_ORDER = ["recommended", "hold", "rule_based"];

// Recommended is the emphasis colour; the other two are deliberately quieter so
// the comparison reads at a glance without implying the baselines are bad.
const POLICY_STYLE = {
  recommended: { color: "var(--purple)", band: "rgba(60, 52, 137, 0.16)" },
  hold: { color: "var(--muted)", band: "rgba(118, 119, 131, 0.13)" },
  rule_based: { color: "var(--blue)", band: "rgba(47, 127, 213, 0.13)" }
};

const PANELS = [
  { key: "mrr", label: "Monthly revenue", format: money },
  { key: "cash", label: "Cash", format: money },
  { key: "churn", label: "Churn", format: (v) => pct(v) },
  { key: "rule_of_40", label: "Rule of 40", format: (v) => v.toFixed(0) }
];

// One panel: three median lines, each inside its own IQR band, on shared axes.
function FanChart({ title, series, format, shockMonth }) {
  const w = 320, h = 130, padX = 6, padTop = 8, padBottom = 18;

  const all = POLICY_ORDER.flatMap((p) => {
    const s = series[p];
    return s ? [...s.p25, ...s.p75, ...s.median] : [];
  });
  if (!all.length) return null;

  const min = Math.min(...all), max = Math.max(...all);
  const span = max - min || 1;
  const months = series[POLICY_ORDER[0]].median.length;

  const x = (i) => padX + (i / Math.max(months - 1, 1)) * (w - padX * 2);
  const y = (v) => h - padBottom - ((v - min) / span) * (h - padTop - padBottom);

  const line = (pts) => pts.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  // Band = upper edge left-to-right, then lower edge right-to-left, closed.
  const band = (lo, hi) =>
    [...hi.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`),
     ...lo.map((v, i) => `${x(lo.length - 1 - i).toFixed(1)},${y(lo[lo.length - 1 - i]).toFixed(1)}`)
    ].join(" ");

  const zeroY = min < 0 && max > 0 ? y(0) : null;

  return (
    <div className="wi-chart">
      <span className="wi-chart-title">{title}</span>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${title} projection`}>
        {zeroY !== null && (
          <line x1={padX} x2={w - padX} y1={zeroY} y2={zeroY} className="wi-zero" />
        )}
        {shockMonth != null && shockMonth < months && (
          <line
            x1={x(shockMonth)} x2={x(shockMonth)} y1={padTop} y2={h - padBottom}
            className="wi-shock-line"
          />
        )}
        {POLICY_ORDER.map((p) => series[p] && (
          <polygon key={`b-${p}`} points={band(series[p].p25, series[p].p75)}
                   fill={POLICY_STYLE[p].band} stroke="none" />
        ))}
        {POLICY_ORDER.map((p) => series[p] && (
          <polyline key={`l-${p}`} points={line(series[p].median)}
                    fill="none" stroke={POLICY_STYLE[p].color} strokeWidth="2"
                    strokeLinejoin="round" strokeLinecap="round" />
        ))}
        <text x={padX} y={h - 5} className="wi-axis">now</text>
        <text x={w - padX} y={h - 5} className="wi-axis" textAnchor="end">
          {months} mo
        </text>
      </svg>
      <span className="wi-chart-end">
        {POLICY_ORDER.filter((p) => series[p]).map((p) => (
          <em key={p} style={{ color: POLICY_STYLE[p].color }}>
            {format(series[p].median[series[p].median.length - 1])}
          </em>
        ))}
      </span>
    </div>
  );
}

export default function WhatIfPanel({ result, loading, error, onRun, shockMode, onToggleShock }) {
  const [showAssumptions, setShowAssumptions] = useState(false);

  if (loading) {
    return (
      <article className="panel wi-panel">
        <div className="wi-loading">
          <Loader2 size={16} className="wi-spin" /> Projecting {12} months…
        </div>
      </article>
    );
  }

  if (error) {
    return (
      <article className="panel wi-panel">
        <div className="wi-error">
          <AlertTriangle size={15} /> {error}
          <button type="button" className="link-button" onClick={onRun}>Try again</button>
        </div>
      </article>
    );
  }

  if (!result) {
    return (
      <article className="panel wi-panel">
        <div className="panel-title-row">
          <h3><Activity size={16} /> What happens if you follow this plan</h3>
        </div>
        <p className="subtle">
          Roll your current numbers forward {12} months under the board's plan, doing
          nothing, and a standard playbook — same simulated conditions for all three.
        </p>
        <button type="button" className="primary-button" onClick={onRun}>
          Run the projection
        </button>
      </article>
    );
  }

  const { policies, horizon_months: horizon, n_seeds: seeds, shock, assumptions } = result;
  const shockMonth = shock ? shock.month : null;

  return (
    <article className="panel wi-panel">
      <div className="panel-title-row">
        <h3><Activity size={16} /> What happens if you follow this plan</h3>
        <button
          type="button"
          className={`wi-shock-toggle ${shockMode ? "on" : ""}`}
          onClick={onToggleShock}
        >
          <Zap size={13} /> {shockMode ? "Competitor shock on" : "Add a competitor shock"}
        </button>
      </div>

      <div className="wi-legend">
        {POLICY_ORDER.filter((p) => policies[p]).map((p) => (
          <span key={p}>
            <i style={{ background: POLICY_STYLE[p].color }} /> {policies[p].label}
          </span>
        ))}
      </div>

      {shockMode && shock && (
        <p className="wi-shock-note">
          <Zap size={13} /> A competitor surge hits all three plans at month {shock.month}:{" "}
          {shock.description}.
        </p>
      )}

      <div className="wi-grid">
        {PANELS.map((panel) => (
          <FanChart
            key={panel.key}
            title={panel.label}
            format={panel.format}
            shockMonth={shockMonth}
            series={Object.fromEntries(
              POLICY_ORDER.filter((p) => policies[p]).map((p) => [p, policies[p].series[panel.key]])
            )}
          />
        ))}
      </div>

      {/* The caveat is rendered here, immediately beneath the charts, on purpose. */}
      <p className="wi-caveat">{result.caveat}</p>

      <div className="wi-table-wrap">
        <table className="wi-table">
          <thead>
            <tr>
              <th>Plan</th>
              <th>Revenue in {horizon} mo</th>
              <th>Cash in {horizon} mo</th>
              <th>Survives</th>
              <th>Rule of 40</th>
              {shockMode && <th>Shock cost</th>}
              {shockMode && <th>Recovery</th>}
            </tr>
          </thead>
          <tbody>
            {POLICY_ORDER.filter((p) => policies[p]).map((p) => {
              const s = policies[p].summary;
              return (
                <tr key={p} className={p === "recommended" ? "wi-row-primary" : ""}>
                  <td><i style={{ background: POLICY_STYLE[p].color }} /> {policies[p].label}</td>
                  <td>{money(s.median_terminal_mrr)}</td>
                  <td>{money(s.median_terminal_cash)}</td>
                  <td>{Math.round(s.survival_rate * 100)}%</td>
                  <td>{s.mean_rule_of_40.toFixed(0)}</td>
                  {shockMode && (
                    <td>{s.shock_cost_pct == null ? "—" : `${s.shock_cost_pct.toFixed(1)}%`}</td>
                  )}
                  {shockMode && (
                    <td>
                      {s.months_to_recover != null
                        ? `${s.months_to_recover} mo`
                        : s.drawdown_fraction === 0
                          ? "no drop"
                          : "—"}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="wi-meta">
        Median of {seeds} simulated runs per plan, same {seeds} starting conditions for each,
        so the plan is the only thing that differs.
        {result.shock_tape_shared === false && (
          <strong> Conditions diverged between plans in this run — compare with care.</strong>
        )}
        {shockMode && POLICY_ORDER.every((p) => policies[p]?.summary.drawdown_fraction === 0) && (
          <> Revenue never fell below its pre-shock level under any plan, so there is no
          recovery time to report.</>
        )}
      </p>

      <button type="button" className="link-button" onClick={() => setShowAssumptions(!showAssumptions)}>
        {showAssumptions ? "Hide" : "Show"} what this projection assumes ({assumptions.length})
      </button>
      {showAssumptions && (
        <ul className="wi-assumptions">
          {assumptions.map((a) => (
            <li key={a.field}>
              <strong>{a.field}:</strong> {a.value}
              <span className={`prov-chip ${a.basis === "derived" ? "simulated" : ""}`}>{a.basis}</span>
              <span className="wi-assumption-detail">{a.detail}</span>
              {a.source && <span className="wi-assumption-source">{a.source}</span>}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
