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
//
// Lines stop where companies stop. The server sends ragged series — a run that
// ran out of cash contributes nothing after the month it died — plus
// `alive_fraction`, the share of runs still solvent in each month. A line is
// solid while every run is alive, dashed once some have died, and ends at a
// marker when none are left. Without that, "died in month 1" and "stagnated for
// a year" were the same flat line.

import React, { useState } from "react";
import { Activity, AlertTriangle, ChevronDown, ChevronRight, Loader2, Skull, Zap } from "lucide-react";
import { money, pct } from "./derive.js";

// The founder-facing comparison is the board's plan against doing nothing.
// The API also returns `rule_based`, the heuristic C-suite from
// agents/baseline_agents - a research comparator rather than a plan anyone is
// offering, and one more line than this question needs. Add it back here to
// render it; nothing on the server changed.
const POLICY_ORDER = ["recommended", "hold"];

// Recommended is the emphasis colour; the other two are deliberately quieter so
// the comparison reads at a glance without implying the baselines are bad.
const POLICY_STYLE = {
  recommended: { color: "var(--purple)", band: "rgba(60, 52, 137, 0.16)" },
  hold: { color: "var(--muted)", band: "rgba(118, 119, 131, 0.13)" },
  rule_based: { color: "var(--blue)", band: "rgba(47, 127, 213, 0.13)" }
};

// The fourth panel is whichever efficiency measure means something at this
// company's size. The server decides and says so in result.display, so the
// $1M-ARR floor is not hardcoded here as a second copy.
function panelsFor(display) {
  const efficiencyPanel = display?.efficiency_panel_series === "rule_of_40"
    ? { key: "rule_of_40", label: "Rule of 40", format: (v) => v.toFixed(0) }
    : {
        key: "spend_ratio",
        label: display?.efficiency_panel_label || "Spend per $1 of revenue",
        format: (v) => `$${v.toFixed(2)}`
      };
  return [
    { key: "mrr", label: "Monthly revenue", format: money },
    { key: "cash", label: "Cash", format: money },
    { key: "churn", label: "Customers lost", format: (v) => `1 in ${Math.round(100 / Math.max(v, 0.01))}` },
    efficiencyPanel
  ];
}

// Where a policy's line can be drawn, and where it has to change character.
//   last  — final month with any surviving run; the line ends here
//   full  — final month where every run is still alive; solid up to here
function lifespan(median, alive) {
  let last = -1;
  for (let i = 0; i < median.length; i += 1) if (median[i] != null) last = i;
  let full = -1;
  for (let i = 0; i <= last; i += 1) {
    if ((alive?.[i] ?? 1) >= 1) full = i;
    else break;
  }
  return { last, full: full < 0 ? 0 : full };
}

// One panel: median lines, each inside its own IQR band, on shared axes.
// Exported so the multi-month Plan page can draw its trajectory with the same
// grammar. Two optional overlays, both inert by default:
//   shockMarkers [{month, type}] labelled vertical lines at scheduled shocks
//   refLines     [{from, to, value, color}] horizontal segments
//   shadeFrom    month index from which everything to the right is the model's
//                projection: a light background band labelled "projected".
//                Dashed keeps meaning "some runs died" (oefa_loop_plan.md 5.2).
//   markers      [{month, value, label}] filled points, e.g. a closed month's
//                actual numbers
//   xEndLabel    the right-hand axis label (defaults to "{n} mo")
export function FanChart({
  title, series, alive, format, shockMonth,
  policies = POLICY_ORDER, styles = POLICY_STYLE,
  shockMarkers = null, refLines = null, xStartLabel = "now",
  shadeFrom = null, markers = null, xEndLabel = null
}) {
  const w = 320, h = 130, padX = 6, padTop = 8, padBottom = 18;

  const all = policies.flatMap((p) => {
    const s = series[p];
    if (!s) return [];
    return [...s.p25, ...s.p75, ...s.median].filter((v) => v != null);
  });
  if (!all.length) return null;
  (refLines || []).forEach((r) => all.push(r.value));
  (markers || []).forEach((m) => { if (m.value != null) all.push(m.value); });

  const min = Math.min(...all), max = Math.max(...all);
  const span = max - min || 1;
  const months = series[policies.find((p) => series[p])].median.length;

  const x = (i) => padX + (i / Math.max(months - 1, 1)) * (w - padX * 2);
  const y = (v) => h - padBottom - ((v - min) / span) * (h - padTop - padBottom);

  // Points between two month indexes, skipping months with no survivors.
  const pointsBetween = (values, from, to) => {
    const out = [];
    for (let i = from; i <= to; i += 1) {
      if (values[i] == null) continue;
      out.push(`${x(i).toFixed(1)},${y(values[i]).toFixed(1)}`);
    }
    return out.join(" ");
  };

  // Band = upper edge left-to-right, then lower edge right-to-left, closed.
  const bandPoints = (lo, hi, last) => {
    const up = [];
    const down = [];
    for (let i = 0; i <= last; i += 1) {
      if (hi[i] == null || lo[i] == null) continue;
      up.push(`${x(i).toFixed(1)},${y(hi[i]).toFixed(1)}`);
      down.unshift(`${x(i).toFixed(1)},${y(lo[i]).toFixed(1)}`);
    }
    return [...up, ...down].join(" ");
  };

  const zeroY = min < 0 && max > 0 ? y(0) : null;

  return (
    <div className="wi-chart">
      <span className="wi-chart-title">{title}</span>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${title} projection`}>
        {shadeFrom != null && shadeFrom < months && (
          <g className="wi-projected">
            <rect x={x(shadeFrom)} y={padTop - 4} width={w - padX - x(shadeFrom)} height={h - padTop - padBottom + 8} />
            <text x={w - padX - 2} y={padTop + 4} textAnchor="end">projected</text>
          </g>
        )}
        {zeroY !== null && (
          <line x1={padX} x2={w - padX} y1={zeroY} y2={zeroY} className="wi-zero" />
        )}
        {shockMonth != null && shockMonth < months && (
          <line
            x1={x(shockMonth)} x2={x(shockMonth)} y1={padTop} y2={h - padBottom}
            className="wi-shock-line"
          />
        )}
        {(shockMarkers || []).filter((s) => s.month < months).map((s, i) => (
          <g key={`s-${s.month}`}>
            <line
              x1={x(s.month)} x2={x(s.month)} y1={padTop} y2={h - padBottom}
              className="wi-shock-line"
            />
            <text
              x={x(s.month) + 2} y={padTop + 6 + (i % 2) * 8}
              className="wi-shock-label"
            >
              {s.type}
            </text>
          </g>
        ))}
        {(refLines || []).map((r, i) => (
          <line
            key={`r-${i}`}
            x1={x(Math.max(r.from, 0))} x2={x(Math.min(r.to, months - 1))}
            y1={y(r.value)} y2={y(r.value)}
            stroke={r.color} strokeWidth="1" strokeDasharray="2 3" opacity="0.85"
          />
        ))}
        {policies.map((p) => {
          const s = series[p];
          if (!s) return null;
          const { last } = lifespan(s.median, alive[p]);
          if (last < 0) return null;
          return (
            <polygon key={`b-${p}`} points={bandPoints(s.p25, s.p75, last)}
                     fill={styles[p].band} stroke="none" />
          );
        })}
        {policies.map((p) => {
          const s = series[p];
          if (!s) return null;
          const { last, full } = lifespan(s.median, alive[p]);
          if (last < 0) return null;
          const stroke = styles[p].color;
          return (
            <g key={`l-${p}`}>
              <polyline points={pointsBetween(s.median, 0, full)}
                        fill="none" stroke={stroke} strokeWidth="2"
                        strokeLinejoin="round" strokeLinecap="round" />
              {/* Dashed once some runs have died: the median behind it is over
                  survivors only, which is a different claim from the solid part. */}
              {last > full && (
                <polyline points={pointsBetween(s.median, full, last)}
                          fill="none" stroke={stroke} strokeWidth="2"
                          strokeDasharray="3 3"
                          strokeLinejoin="round" strokeLinecap="round" />
              )}
              {last < months - 1 && s.median[last] != null && (
                <g className="wi-death-mark">
                  <line x1={x(last) - 3} y1={y(s.median[last]) - 3}
                        x2={x(last) + 3} y2={y(s.median[last]) + 3} stroke={stroke} />
                  <line x1={x(last) - 3} y1={y(s.median[last]) + 3}
                        x2={x(last) + 3} y2={y(s.median[last]) - 3} stroke={stroke} />
                </g>
              )}
            </g>
          );
        })}
        {(markers || []).filter((m) => m.value != null && m.month < months).map((m) => (
          <g key={`m-${m.month}`} className="wi-marker">
            <circle cx={x(m.month)} cy={y(m.value)} r="3.6" />
            {m.label && <text x={x(m.month) + 6} y={y(m.value) - 5}>{m.label}</text>}
          </g>
        ))}
        <text x={padX} y={h - 5} className="wi-axis">{xStartLabel}</text>
        <text x={w - padX} y={h - 5} className="wi-axis" textAnchor="end">
          {xEndLabel || `${months} mo`}
        </text>
      </svg>
      <span className="wi-chart-end">
        {policies.filter((p) => series[p]).map((p) => {
          const s = series[p];
          const { last } = lifespan(s.median, alive[p]);
          return (
            <em key={p} style={{ color: styles[p].color }}>
              {last >= 0 && s.median[last] != null ? format(s.median[last]) : "—"}
            </em>
          );
        })}
      </span>
    </div>
  );
}

// "Survives 0%" told a founder nothing about whether the company lasted one
// month or eleven. The sentence comes from founder_view on the server; the
// percentage stays because the column is scanned, not read.
function survivalCell(policy) {
  const summary = policy.summary;
  const survived = Math.round(summary.survival_rate * 100);
  if (!summary.deaths) return <span className="wi-alive">{survived}%</span>;
  return (
    <span className={survived === 0 ? "wi-dead" : "wi-partial"}>
      {survived}%
      <em>{policy.display?.survival}</em>
    </span>
  );
}

// The panel leads with the verdict — the death note, the table and the
// server's caveat — and keeps the four charts, the legend, the shock toggle
// and the run metadata behind one expander. The assumptions the projection
// used are rendered by the page, in its Assumptions section, from
// `result.assumptions`, so they sit beside the analysis's own guesses.
export default function WhatIfPanel({ result, loading, error, onRun, shockMode, onToggleShock }) {
  const [showCharts, setShowCharts] = useState(false);

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
          <h3><Activity size={16} /> The plan against doing nothing</h3>
        </div>
        <p className="subtle">
          Roll your current numbers forward {12} months under the board's plan and under
          doing nothing — the same simulated conditions for both, so the plan is the only
          thing that differs.
        </p>
        <button type="button" className="primary-button" onClick={onRun}>
          Run the projection
        </button>
      </article>
    );
  }

  const { policies, horizon_months: horizon, n_seeds: seeds, shock } = result;
  const display = result.display || {};
  const panels = panelsFor(display);
  const shockMonth = shock ? shock.month : null;
  const alive = Object.fromEntries(
    POLICY_ORDER.filter((p) => policies[p]).map((p) => [p, policies[p].alive_fraction || []])
  );

  // The board's own plan running out of cash is the headline, not a table cell.
  const recommended = policies.recommended?.summary;
  const anyDeaths = POLICY_ORDER.some((p) => policies[p]?.summary.deaths > 0);

  return (
    <article className="panel wi-panel">
      <div className="panel-title-row">
        <h3><Activity size={16} /> The plan against doing nothing</h3>
        {shockMode && <span className="wi-shock-on"><Zap size={13} /> competitor shock on</span>}
      </div>

      {recommended?.deaths > 0 && (
        <p className="wi-death-note">
          <Skull size={14} />
          <span>
            Over {horizon} simulated months, the board's plan ran out of cash in{" "}
            {recommended.deaths === recommended.runs
              ? "every run"
              : `${recommended.deaths} of ${recommended.runs} runs`}
            {recommended.median_death_month != null &&
              `, typically around month ${recommended.median_death_month}`}
            {recommended.earliest_death_month != null &&
              recommended.earliest_death_month !== recommended.median_death_month &&
              ` (earliest month ${recommended.earliest_death_month})`}
            .
          </span>
        </p>
      )}

      <div className="wi-table-wrap">
        <table className="wi-table">
          <thead>
            <tr>
              <th>Plan</th>
              <th>Revenue in {horizon} mo</th>
              <th>Cash in {horizon} mo</th>
              <th>Survives (of simulated runs)</th>
              <th>{display.efficiency_panel_label || "Rule of 40"}</th>
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
                  <td>{policies[p].display?.revenue ?? money(s.median_terminal_mrr)}</td>
                  <td>{policies[p].display?.cash ?? money(s.median_terminal_cash)}</td>
                  <td>{survivalCell(policies[p])}</td>
                  <td>
                    {display.efficiency_panel_series === "rule_of_40"
                      ? s.mean_rule_of_40.toFixed(0)
                      : policies[p].display?.spend_ratio ?? "—"}
                  </td>
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

      {/* The server's caveat sits directly under the verdict, on purpose. */}
      <p className="wi-caveat">{result.caveat}</p>

      {display.rule_of_40_withheld_because && (
        <p className="subtle">{display.rule_of_40_withheld_because}</p>
      )}

      <button type="button" className="link-button" onClick={() => setShowCharts(!showCharts)}>
        {showCharts ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        {showCharts ? "Hide" : "Show"} the charts
      </button>

      {showCharts && (
        <div className="wi-charts">
          <div className="wi-legend-row">
            <div className="wi-legend">
              {POLICY_ORDER.filter((p) => policies[p]).map((p) => (
                <span key={p}>
                  <i style={{ background: POLICY_STYLE[p].color }} /> {policies[p].label}
                </span>
              ))}
            </div>
            <button
              type="button"
              className={`wi-shock-toggle ${shockMode ? "on" : ""}`}
              onClick={onToggleShock}
            >
              <Zap size={13} /> {shockMode ? "Competitor shock on" : "Add a competitor shock"}
            </button>
          </div>

          {shockMode && shock && (
            <p className="wi-shock-note">
              <Zap size={13} /> A competitor surge hits both plans at month {shock.month}:{" "}
              {shock.description}.
            </p>
          )}

          <div className="wi-grid">
            {panels.map((panel) => (
              <FanChart
                key={panel.key}
                title={panel.label}
                format={panel.format}
                shockMonth={shockMonth}
                alive={alive}
                series={Object.fromEntries(
                  POLICY_ORDER.filter((p) => policies[p]).map((p) => [p, policies[p].series[panel.key]])
                )}
              />
            ))}
          </div>

          {anyDeaths && (
            <p className="wi-survivor-note">
              A line goes dashed once some runs have run out of cash, and ends where none are
              left. Past that point the line averages only the companies still standing, so it
              can rise while most of them are gone.
            </p>
          )}

          <p className="wi-meta">
            Median of {seeds} simulated runs per plan, same {seeds} starting conditions for each,
            so the plan is the only thing that differs.
            {anyDeaths && (
              <> Revenue and cash for a run that ended early are its figures at the month it
              ended.</>
            )}
            {result.shock_tape_shared === false && (
              <strong> Conditions diverged between plans in this run — compare with care.</strong>
            )}
            {shockMode && POLICY_ORDER.every((p) => policies[p]?.summary.drawdown_fraction === 0) && (
              <> Revenue never fell below its pre-shock level under any plan, so there is no
              recovery time to report.</>
            )}
          </p>
        </div>
      )}
    </article>
  );
}

// The projection's own assumptions, rendered by the page beside the
// analysis's guesses rather than inside the panel.
export function WhatIfAssumptions({ assumptions }) {
  if (!assumptions?.length) return null;
  return (
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
  );
}
