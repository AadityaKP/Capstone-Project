// Run tab — all four policies from all five demo companies' C1-mapped states.
//
// One button, twenty runs. Each arm is one POST /api/review/backtest/run: the
// environment is initialized exactly as the C1 backtest did (scale-aware
// physics, company gross margin, real G&A burn, assumed price/churn/CAC from
// mapped_states.csv, scheduled shocks off, deterministic_rng on), then rolled
// through simulation_runner.run_simulation — verified in-session to reproduce
// C1's recorded medians to machine precision. The fifteen rule-arm runs return
// almost instantly and render per company; the five oracle_v3 runs (local LLM
// via Ollama) stream in behind per-company progress indicators. The hold arm
// and the companies' actual trajectories are deliberately absent: this screen
// compares policies inside the model, it does not retrodict.

import React, { useEffect, useState } from "react";
import { AlertTriangle, Building2, Loader2, Play } from "lucide-react";
import { reviewBacktestCompanies, reviewBacktestRun } from "../api.js";
import { FanChart } from "../whatif.jsx";
import { money } from "../derive.js";

// One fixed colour per policy, everywhere on this screen.
const POLICY_STYLE = {
  noop: { color: "var(--muted)", band: "rgba(118, 119, 131, 0.13)" },
  heuristic: { color: "var(--green)", band: "rgba(31, 138, 84, 0.13)" },
  boardroom: { color: "var(--purple)", band: "rgba(60, 52, 137, 0.16)" },
  oracle_v3: { color: "var(--blue)", band: "rgba(47, 127, 213, 0.13)" }
};
const FAST_POLICIES = ["noop", "heuristic", "boardroom"];
const ALL_POLICIES = [...FAST_POLICIES, "oracle_v3"];

const growthPct = (v) => `${(v * 100).toFixed(1)}%`;

function traceSeries(values, length) {
  const padded = values.length >= length
    ? values
    : [...values, ...new Array(length - values.length).fill(null)];
  return { median: padded, p25: padded, p75: padded };
}

function growthSeries(arm, length) {
  const out = new Array(length).fill(null);
  for (const point of arm.growth) {
    if (point.month < length) out[point.month] = point.growth;
  }
  return out;
}

function CompanyBlock({ company, arms }) {
  const loaded = ALL_POLICIES.filter((p) => arms?.[p]?.data);
  const maxLen = Math.max(0, ...loaded.map((p) => arms[p].data.months.length));

  const series = {};
  const alive = {};
  for (const p of loaded) {
    const d = arms[p].data;
    series[p] = {
      mrr: traceSeries(d.mrr, maxLen),
      cash: traceSeries(d.cash, maxLen),
      growth: traceSeries(growthSeries(d, maxLen), maxLen)
    };
    alive[p] = d.months.map(() => 1);
  }

  const charts = [
    { key: "mrr", label: "MRR", format: money },
    { key: "cash", label: "Cash", format: money },
    { key: "growth", label: "MRR growth (QoQ)", format: growthPct }
  ];

  return (
    <div className="rv-company-block">
      <div className="rv-company-head">
        <strong>{company.ticker}</strong>
        <span>
          from {company.init_quarter}: MRR {money(company.mrr)}, cash{" "}
          {money(company.cash)}, gross margin {(company.gross_margin * 100).toFixed(1)}%;
          assumed price ${Math.round(company.price_assumed)}/mo, churn{" "}
          {(company.churn_assumed * 100).toFixed(1)}%/mo, CAC {money(company.cac_assumed)}.
        </span>
      </div>

      {ALL_POLICIES.filter((p) => arms?.[p]?.loading).map((p) => (
        <p key={p} className="rv-progress">
          <Loader2 size={14} className="wi-spin" />
          <span>
            <strong style={{ color: POLICY_STYLE[p].color }}>{p}</strong>
            {p === "oracle_v3"
              ? " is running on the local LLM (Ollama, llama3.1:8b)…"
              : " is running…"}
          </span>
        </p>
      ))}
      {ALL_POLICIES.filter((p) => arms?.[p]?.error).map((p) => (
        <p key={p} className="wi-error">
          <AlertTriangle size={15} /> {p}: {arms[p].error}
        </p>
      ))}

      {loaded.length > 0 && (
        <>
          <div className="wi-grid">
            {charts.map((c) => (
              <FanChart
                key={c.key}
                title={c.label}
                format={c.format}
                series={Object.fromEntries(loaded.map((p) => [p, series[p][c.key]]))}
                alive={alive}
                policies={loaded}
                styles={POLICY_STYLE}
                xStartLabel="month 0"
              />
            ))}
          </div>
          <div className="wi-table-wrap">
            <table className="wi-table">
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Final MRR</th>
                  <th>Survived</th>
                  <th>Min cash</th>
                </tr>
              </thead>
              <tbody>
                {loaded.map((p) => {
                  const s = arms[p].data.summary;
                  return (
                    <tr key={p}>
                      <td><i style={{ background: POLICY_STYLE[p].color }} /> {p}</td>
                      <td>{money(s.final_mrr)}</td>
                      <td className={s.survived ? "wi-alive" : "wi-dead"}>
                        {s.survived ? "yes" : `no — month ${s.months_survived}`}
                      </td>
                      <td>{money(s.min_cash)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

export default function Run() {
  const [companies, setCompanies] = useState(null);
  const [metaError, setMetaError] = useState(null);
  const [seed, setSeed] = useState(0);
  const [horizon, setHorizon] = useState(12);
  const [runs, setRuns] = useState({}); // ticker -> policy -> {loading, data, error}

  useEffect(() => {
    let alive = true;
    reviewBacktestCompanies().then((res) => {
      if (!alive) return;
      if (res.ok) {
        setCompanies(res.data);
        setSeed(res.data.default_seed);
        setHorizon(res.data.default_horizon);
      } else {
        setMetaError(res.error || "Company states unavailable");
      }
    });
    return () => { alive = false; };
  }, []);

  const list = companies?.companies || [];
  const anyLoading = Object.values(runs).some((arms) =>
    Object.values(arms).some((a) => a.loading)
  );

  const onRun = () => {
    const seedValue = Math.max(0, Math.floor(Number(seed) || 0));
    const horizonValue = Math.min(120, Math.max(1, Math.floor(Number(horizon) || 12)));
    setSeed(seedValue);
    setHorizon(horizonValue);
    setRuns(Object.fromEntries(list.map((c) => [
      c.ticker,
      Object.fromEntries(ALL_POLICIES.map((p) => [p, { loading: true }]))
    ])));
    const dispatch = (ticker, policy) => {
      reviewBacktestRun(ticker, policy, seedValue, horizonValue).then((res) => {
        setRuns((prev) => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            [policy]: res.ok ? { data: res.data } : { error: res.error || "Run failed" }
          }
        }));
      });
    };
    // All fifteen rule-arm runs first, then the five LLM runs: the server's run
    // lock serves requests in arrival order, so every company's rule arms are
    // on screen before the first oracle occupies the lock.
    for (const c of list) for (const p of FAST_POLICIES) dispatch(c.ticker, p);
    for (const c of list) dispatch(c.ticker, "oracle_v3");
  };

  return (
    <section className="content-stack">
      <article className="panel rv-panel">
        <div className="panel-title-row">
          <h3><Building2 size={16} /> Run — all policies, all five demo companies</h3>
        </div>
        <p className="subtle">
          Each company's environment starts from its earliest complete EDGAR quarter,
          mapped exactly as the C1 backtest mapped it (price, churn and CAC are labelled
          assumptions), with <code>deterministic_rng</code> on and no scheduled shocks —
          so at equal seed all four policies face that company's identical world.
          Five companies from the 39-company panel; simulated counterfactuals, actual
          trajectories deliberately not shown.
        </p>

        {metaError && <p className="wi-error"><AlertTriangle size={15} /> {metaError}</p>}

        <div className="rv-controls">
          <label className="rv-field">
            <span>Seed</span>
            <input
              type="number" min="0" step="1" value={seed}
              onChange={(e) => setSeed(e.target.value)} disabled={anyLoading}
            />
          </label>
          <label className="rv-field">
            <span>Horizon (months)</span>
            <input
              type="number" min="1" max="120" step="1" value={horizon}
              onChange={(e) => setHorizon(e.target.value)} disabled={anyLoading}
            />
          </label>
          <button
            type="button" className="primary-button" onClick={onRun}
            disabled={anyLoading || !list.length}
          >
            {anyLoading ? <Loader2 size={15} className="wi-spin" /> : <Play size={15} />}
            {anyLoading ? "Running…" : "Run all policies"}
          </button>
        </div>

        <div className="wi-legend">
          {ALL_POLICIES.map((p) => (
            <span key={p}>
              <i style={{ background: POLICY_STYLE[p].color }} /> {p}
            </span>
          ))}
        </div>

        {list.map((c) => (
          <CompanyBlock key={c.ticker} company={c} arms={runs[c.ticker]} />
        ))}
      </article>
    </section>
  );
}
