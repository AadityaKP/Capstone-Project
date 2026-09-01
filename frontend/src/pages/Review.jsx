// Review demo — two policies, one seed, one shared deterministic world.
//
// Research surface, deliberately separate from the founder pages: policies and
// seeds appear here and nowhere else. Each arm is one POST /api/review/compare
// (simulation_runner.run_simulation, deterministic_rng on — the same regime as
// every paired validation result), fired fast-arm-first so the boardroom renders
// while an LLM arm (oracle_v3 on local Ollama) is still thinking.
//
// Chart grammar is the founder what-if panel's FanChart, reused with research
// overlays: the EDGAR QoQ-growth band on the growth chart, labelled markers at
// the scheduled shock months on every time chart, and each arm's pre-shock
// Rule-of-40 level as a dashed reference so the recovery gap is visible.
// All band numbers arrive from the server (validation/results/); nothing here
// types a benchmark value.

import React, { useEffect, useState } from "react";
import { AlertTriangle, FlaskConical, Loader2, Play } from "lucide-react";
import { reviewMeta, reviewCompare } from "../api.js";
import { FanChart } from "../whatif.jsx";
import { money } from "../derive.js";

const ARM_KEYS = ["a", "b"];
const ARM_STYLE = {
  a: { color: "var(--purple)", band: "rgba(60, 52, 137, 0.16)" },
  b: { color: "var(--blue)", band: "rgba(47, 127, 213, 0.13)" }
};

const growthPct = (v) => `${(v * 100).toFixed(1)}%`;
const intFmt = (v) => v.toFixed(0);

// A single-run trace dressed as FanChart series: the "median" is the trace and
// the IQR edges coincide with it, so the band collapses to nothing. Padded to
// the longer arm's length so an early bankruptcy doesn't truncate the x-axis.
function traceSeries(values, length) {
  const padded = values.length >= length
    ? values
    : [...values, ...new Array(length - values.length).fill(null)];
  return { median: padded, p25: padded, p75: padded };
}

// The QoQ growth points live at each complete quarter's last month; the rest of
// the monthly axis is null and FanChart connects across the gaps.
function growthSeries(arm, length) {
  const out = new Array(length).fill(null);
  for (const point of arm.growth) {
    if (point.month < length) out[point.month] = point.growth;
  }
  return out;
}

function bandTooltip(eg) {
  return [
    "EDGAR QoQ revenue growth",
    `p10 ${growthPct(eg.p10)}`,
    `p25 ${growthPct(eg.p25)}`,
    `median ${growthPct(eg.median)}`,
    `p75 ${growthPct(eg.p75)}`,
    `p90 ${growthPct(eg.p90)}`
  ].join(" · ");
}

function recoveryCell(event) {
  if (event.pre_shock_r40 == null) return "—";
  if (event.recovered) return `${event.months_to_recover} mo`;
  return event.censored || "—";
}

export default function Review() {
  const [meta, setMeta] = useState(null);
  const [metaError, setMetaError] = useState(null);
  const [picks, setPicks] = useState({ a: "boardroom", b: "oracle_v3" });
  const [seed, setSeed] = useState(0);
  const [arms, setArms] = useState({ a: null, b: null }); // {loading, data, error, policy}

  useEffect(() => {
    let alive = true;
    reviewMeta().then((res) => {
      if (!alive) return;
      if (res.ok) setMeta(res.data);
      else setMetaError(res.error || "Review data unavailable");
    });
    return () => { alive = false; };
  }, []);

  const runArm = (key, policy, seedValue) => {
    reviewCompare(policy, seedValue).then((res) => {
      setArms((prev) => ({
        ...prev,
        [key]: res.ok
          ? { policy, data: res.data }
          : { policy, error: res.error || "Simulation failed" }
      }));
    });
  };

  const onCompare = () => {
    const seedValue = Math.max(0, Math.floor(Number(seed) || 0));
    setSeed(seedValue);
    setArms({
      a: { policy: picks.a, loading: true },
      b: { policy: picks.b, loading: true }
    });
    // Fast arm first: the server serializes simulations, so whichever request
    // arrives first also finishes first. LLM arms go second on purpose.
    const llmArms = new Set(["oracle_v3"]);
    const order = llmArms.has(picks.a) && !llmArms.has(picks.b) ? ["b", "a"] : ["a", "b"];
    for (const key of order) runArm(key, picks[key], seedValue);
  };

  const loaded = ARM_KEYS.filter((k) => arms[k]?.data);
  const anyLoading = ARM_KEYS.some((k) => arms[k]?.loading);
  const maxLen = Math.max(0, ...loaded.map((k) => arms[k].data.months.length));

  const series = {};
  const alive = {};
  if (loaded.length) {
    for (const k of loaded) {
      const d = arms[k].data;
      series[k] = {
        mrr: traceSeries(d.mrr, maxLen),
        cash: traceSeries(d.cash, maxLen),
        growth: traceSeries(growthSeries(d, maxLen), maxLen),
        rule_of_40: traceSeries(d.rule_of_40, maxLen)
      };
      alive[k] = d.months.map(() => 1);
    }
  }

  const shockMarkers = loaded.length
    ? arms[loaded[0]].data.shocks.map((s) => ({ month: s.month, type: s.type }))
    : null;
  const shockMonths = shockMarkers ? shockMarkers.map((s) => s.month) : [];

  const r40RefLines = loaded.flatMap((k) =>
    arms[k].data.summary.shock_recoveries
      .filter((e) => e.pre_shock_r40 != null)
      .map((e) => ({
        from: e.shock_month,
        to: e.shock_month + 24,
        value: e.pre_shock_r40,
        color: ARM_STYLE[k].color
      }))
  );

  const eg = meta?.edgar_growth;
  const ds = meta?.dataset;
  const edgarLegend = eg && ds
    ? `EDGAR panel, ${ds.n_companies} cos / ${ds.n_complete_quarters.toLocaleString("en-US")} quarters`
    : null;
  const growthBand = eg
    ? { p10: eg.p10, p25: eg.p25, median: eg.median, p75: eg.p75, p90: eg.p90, tooltip: bandTooltip(eg) }
    : null;

  const charts = [
    { key: "mrr", label: "MRR", format: money },
    { key: "cash", label: "Cash", format: money },
    { key: "growth", label: "MRR growth (QoQ)", format: growthPct, band: growthBand },
    { key: "rule_of_40", label: "Rule of 40", format: intFmt, refLines: r40RefLines }
  ];

  const policies = meta?.compare_policies || ["noop", "heuristic", "boardroom", "oracle_v3", "random"];

  return (
    <section className="content-stack">
      <article className="panel rv-panel">
        <div className="panel-title-row">
          <h3><FlaskConical size={16} /> Policy comparison — same seed, same world</h3>
        </div>
        <p className="subtle">
          Two policies over 120 months with <code>deterministic_rng</code> on: the
          environment owns a private random stream, so at equal seed both arms face
          the identical macro path and shock tape — the regime every paired
          validation result used. Simulated counterfactual, not a forecast.
        </p>

        {metaError && (
          <p className="wi-error"><AlertTriangle size={15} /> {metaError}</p>
        )}

        <div className="rv-controls">
          {ARM_KEYS.map((k) => (
            <label key={k} className="rv-field">
              <span style={{ color: ARM_STYLE[k].color }}>Policy {k.toUpperCase()}</span>
              <select
                value={picks[k]}
                onChange={(e) => setPicks({ ...picks, [k]: e.target.value })}
                disabled={anyLoading}
              >
                {policies.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
          ))}
          <label className="rv-field">
            <span>Seed</span>
            <input
              type="number" min="0" step="1" value={seed}
              onChange={(e) => setSeed(e.target.value)}
              disabled={anyLoading}
            />
          </label>
          <button
            type="button" className="primary-button" onClick={onCompare}
            disabled={anyLoading || !meta}
          >
            {anyLoading ? <Loader2 size={15} className="wi-spin" /> : <Play size={15} />}
            {anyLoading ? "Running…" : "Compare"}
          </button>
        </div>

        {ARM_KEYS.filter((k) => arms[k]?.loading).map((k) => (
          <p key={k} className="rv-progress">
            <Loader2 size={14} className="wi-spin" />
            <span>
              <strong style={{ color: ARM_STYLE[k].color }}>{arms[k].policy}</strong>
              {" "}is running{arms[k].policy === "oracle_v3"
                ? " on the local LLM (Ollama, llama3.1:8b) — a 120-month episode takes minutes; the other arm renders as soon as it finishes."
                : "…"}
            </span>
          </p>
        ))}
        {ARM_KEYS.filter((k) => arms[k]?.error).map((k) => (
          <p key={k} className="wi-error">
            <AlertTriangle size={15} /> {arms[k].policy}: {arms[k].error}
          </p>
        ))}

        {loaded.length > 0 && (
          <>
            <div className="wi-legend">
              {loaded.map((k) => (
                <span key={k}>
                  <i style={{ background: ARM_STYLE[k].color }} /> {arms[k].policy} (seed {arms[k].data.seed})
                </span>
              ))}
              {edgarLegend && (
                <span title={growthBand?.tooltip}>
                  <i className="rv-edgar-swatch" /> {edgarLegend}
                </span>
              )}
            </div>

            <div className="wi-grid">
              {charts.map((c) => (
                <FanChart
                  key={c.key}
                  title={c.label}
                  format={c.format}
                  series={Object.fromEntries(loaded.map((k) => [k, series[k][c.key]]))}
                  alive={alive}
                  policies={loaded}
                  styles={ARM_STYLE}
                  shockMarkers={shockMarkers}
                  band={c.band || null}
                  refLines={c.refLines?.length ? c.refLines : null}
                  xStartLabel="month 0"
                />
              ))}
            </div>

            <p className="wi-caveat">
              Vertical amber lines mark the scheduled shocks; dashed horizontal lines on
              the Rule-of-40 chart are each arm's pre-shock level for the 24 months after
              a shock. "Recovery" everywhere on this screen is <strong>Rule-of-40
              recovery</strong> — the month Rule of 40 regains its pre-shock level — not
              revenue recovery. On the growth chart, simulated monthly MRR is aggregated
              to quarters exactly as the E1 test does (complete 3-month quarters,
              quarterly revenue ratio) before it is compared with the EDGAR band.
            </p>

            <div className="wi-table-wrap">
              <table className="wi-table">
                <thead>
                  <tr>
                    <th>Policy</th>
                    <th>Final MRR</th>
                    <th>Survived</th>
                    <th>Min cash</th>
                    {shockMonths.map((m) => (
                      <th key={m}>R40 recovery, shock @{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loaded.map((k) => {
                    const s = arms[k].data.summary;
                    const byMonth = Object.fromEntries(
                      s.shock_recoveries.map((e) => [e.shock_month, e])
                    );
                    return (
                      <tr key={k}>
                        <td><i style={{ background: ARM_STYLE[k].color }} /> {arms[k].policy}</td>
                        <td>{money(s.final_mrr)}</td>
                        <td className={s.survived ? "wi-alive" : "wi-dead"}>
                          {s.survived ? "yes" : `no — month ${s.months_survived}`}
                        </td>
                        <td>{money(s.min_cash)}</td>
                        {shockMonths.map((m) => (
                          <td key={m}>{byMonth[m] ? recoveryCell(byMonth[m]) : "—"}</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="wi-meta">
              Months to regain the pre-shock Rule of 40 (Rule-of-40 recovery), censored at
              24 months or episode death — the A8 definition. Research-scale defaults
              ($50k MRR, $1M cash), one episode per arm.
            </p>
          </>
        )}
      </article>
    </section>
  );
}
