// Plan — the 3–4 month view (docs/oefa_loop_plan.md section 5).
//
// Four month columns across the top, all visible at once, each filling in as
// its month lands. Month 1 is the decision the founder can act on this week;
// months 2–H are the model's physics compounded and carry a projection chip
// for that reason. One trajectory chart below with the seed band as the only
// uncertainty grammar, and the history rail running forward past "now" so
// that when a month closes the projected card is replaced in place by the
// actual one and the prediction error is the diff in the same slot.

import React from "react";
import { AlertTriangle, ChevronRight, LoaderCircle, RefreshCw, Workflow } from "lucide-react";
import {
  useStore, latestCycle, latestMonth, monthById, feedbackForCycleMonth
} from "../store.jsx";
import { useCycleRun } from "../cycleRun.jsx";
import { money, pct, signedPct, monthName } from "../derive.js";
import { runwayLabel } from "../founderView.js";
import { Banner, OefaStrip, RiskChip, SimulatedTag } from "../components.jsx";
import { FanChart } from "../whatif.jsx";
import { actionSummary, predictionSentences, scoreLine } from "../loopView.js";

const STYLE = { cycle: { color: "var(--purple)", band: "rgba(60, 52, 137, 0.16)" } };

function monthLabel(baseIso, offset) {
  const d = new Date(baseIso);
  d.setMonth(d.getMonth() + offset, 1);
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

function runwayText(months) {
  if (months === null || months === undefined) return "not burning";
  if (months <= 0) return "none";
  if (months >= 60) return "5 yr+";
  return `${Math.round(months)} mo`;
}

// ---- one month column ----

function MonthColumn({ index, month, baseIso, dominant, closed, onOpenAdvice }) {
  const label = monthLabel(baseIso, index);
  if (!month) {
    return (
      <article className="month-col pending">
        <div className="month-col-head"><span className="month-col-label">{label}</span></div>
        <div className="month-col-wait"><LoaderCircle size={16} className="spin" /> deliberating…</div>
      </article>
    );
  }
  const after = closed?.result?.actual_state || month.feedback?.state_after;
  const dead = month.feedback?.state_after?.survived === false;
  return (
    <article className={`month-col ${dominant ? "dominant" : ""} ${closed ? "closed" : ""}`}>
      <div className="month-col-head">
        <span className="month-col-label">{label}</span>
        {dominant ? <span className="chip live">this month</span>
          : closed ? <span className="chip actual">closed</span>
          : <span className="chip projection">projection</span>}
      </div>
      {month.execute.brief?.risk_level && <RiskChip level={month.execute.brief.risk_level} />}
      <ul className="month-action-lines">
        {actionSummary(month.execute.action).map((line) => <li key={line}>{line}</li>)}
      </ul>
      <div className="month-state-grid">
        <span><small>{closed ? "Revenue (yours)" : "Revenue after"}</small><strong>{money(after?.mrr)}</strong></span>
        <span><small>Cash</small><strong>{money(after?.cash)}</strong></span>
        <span><small>Cash lasts</small><strong>{runwayText(after?.runway_months)}</strong></span>
        <span><small>Churn</small><strong>{after?.churn_pct != null ? pct(after.churn_pct) : "—"}</strong></span>
      </div>
      {dead && <p className="month-dead"><AlertTriangle size={13} /> ran out of cash in simulation</p>}
      <OefaStrip month={month} closed={closed} compact />
      {dominant && onOpenAdvice && (
        <button className="link-button" type="button" onClick={onOpenAdvice}>
          Full advice <ChevronRight size={14} />
        </button>
      )}
    </article>
  );
}

// ---- trajectory chart across the H months ----

function TrajectoryChart({ months, opening }) {
  if (!months.length || !opening) return null;
  const series = {};
  const keys = [["mrr", "mrr"], ["cash", "cash"], ["churn_pct", "churn_pct"]];
  for (const [key, bandKey] of keys) {
    const median = [opening[key]];
    const p25 = [opening[key]];
    const p75 = [opening[key]];
    for (const m of months) {
      const after = m.feedback?.state_after;
      const band = m.feedback?.projection_band?.[bandKey];
      median.push(after ? after[key] : null);
      p25.push(band ? band.p25 : after ? after[key] : null);
      p75.push(band ? band.p75 : after ? after[key] : null);
    }
    series[key] = { median, p25, p75 };
  }
  const alive = [1, ...months.map((m) => (m.feedback?.state_after?.survived === false ? 0 : 1))];
  const panels = [
    { key: "mrr", label: "Monthly revenue", format: money },
    { key: "cash", label: "Cash", format: money },
    { key: "churn_pct", label: "Customers lost", format: (v) => `1 in ${Math.round(100 / Math.max(v, 0.01))}` }
  ];
  return (
    <article className="panel">
      <div className="panel-title-row"><h3>Where this plan takes you</h3><SimulatedTag /></div>
      <div className="wi-grid three">
        {panels.map((p) => (
          <FanChart
            key={p.key} title={p.label} format={p.format}
            series={{ cycle: series[p.key] }} alive={{ cycle: alive }}
            policies={["cycle"]} styles={STYLE} xStartLabel="now"
          />
        ))}
      </div>
      <p className="wi-caveat">
        The band is the spread across simulated worlds for each month's plan. Months 2 onward
        are the model's physics compounded month over month, not a forecast — the simulator
        scored a B against real companies, which is why your real numbers are checked against
        the plan when you close the month.
      </p>
    </article>
  );
}

// ---- the timeline: history, "now", then the plan ----

function ProjectedEntry({ index, month, baseIso, closed }) {
  const result = closed?.result;
  const after = result?.actual_state || month.feedback?.state_after;
  const before = month.observe?.state_before;
  const sentences = result
    ? predictionSentences({ before, actual: result.actual_state, expected: month.execute.expected_delta, error: result.prediction_error })
    : [];
  return (
    <li className={`timeline-entry ${result ? "actual" : "projected"}`}>
      <div className="timeline-rail"><i /></div>
      <div className="timeline-card">
        <div className="timeline-head static">
          <span className="timeline-month">{monthLabel(baseIso, index)}</span>
          <span className={`chip ${result ? "actual" : index === 0 ? "live" : "projection"}`}>
            {result ? "what actually happened" : index === 0 ? "this month's plan" : "projected"}
          </span>
        </div>
        <p className="timeline-numbers">
          MRR {money(after?.mrr)} · churn {after?.churn_pct != null ? pct(after.churn_pct) : "—"} · cash lasts {runwayText(after?.runway_months)}
        </p>
        <p className="timeline-plan">Plan: {actionSummary(month.execute.action).join(" · ").toLowerCase()}</p>
        {result && (
          <ul className="pe-diff">
            {sentences.map((s) => <li key={s.key} className={`pe-line ${s.tone}`}>{s.text}</li>)}
            <li className="muted">{scoreLine(result.prediction_error)}</li>
          </ul>
        )}
      </div>
    </li>
  );
}

function HistoryEntry({ month, prev }) {
  const v = month.values;
  const mrrPct = prev && prev.values.mrr > 0 ? ((v.mrr - prev.values.mrr) / prev.values.mrr) * 100 : null;
  return (
    <li className="timeline-entry">
      <div className="timeline-rail"><i /></div>
      <div className="timeline-card">
        <div className="timeline-head static"><span className="timeline-month">{monthName(month.enteredAt)}</span></div>
        <p className="timeline-numbers">
          MRR {money(v.mrr)}{mrrPct != null && <em> ({signedPct(mrrPct)})</em>}
          {" · "}churn {pct(v.churnMonthly)}
          {" · "}cash lasts {runwayLabel(v)}
        </p>
      </div>
    </li>
  );
}

// ---- the page ----

export default function Cycle({ navigate }) {
  const { state } = useStore();
  const { pollError, elapsed } = useCycleRun();
  const cycle = latestCycle(state);
  const baseMonth = cycle ? monthById(state, cycle.monthId) : latestMonth(state);
  const current = latestMonth(state);

  const running = cycle && !state.demo && ["queued", "running"].includes(cycle.status);
  const analysisForCycle = cycle ? state.analyses.find((a) => a.cycleId === cycle.id) : null;

  const months = cycle?.months || [];
  const horizon = cycle?.horizon || cycle?.summary?.horizon_months || 4;
  const opening = months[0]?.observe?.state_before || null;
  const closedMonth1 = feedbackForCycleMonth(cycle, 1);
  const stale = cycle && current && cycle.monthId !== current.id;

  if (!cycle) {
    return (
      <section className="empty-state">
        <Workflow size={40} className="warn-icon" />
        <h2>No plan yet</h2>
        <p className="narrow">
          The board plans {4} months at a time: it decides this month, simulates what follows,
          checks its own prediction and adapts before the next month. Run it on your current numbers.
        </p>
        <button className="primary-button" type="button" onClick={() => navigate("/analyzing")}>
          <RefreshCw size={15} /> Run the plan
        </button>
      </section>
    );
  }

  const summary = cycle.summary;

  return (
    <section className="content-stack plan-page">
      {stale && (
        <Banner tone="info" actions={
          <button className="primary-button small" type="button" onClick={() => navigate("/update")}>
            Close the month
          </button>
        }>
          This plan was made on your {monthName(baseMonth?.enteredAt)} numbers. Close that month with what
          actually happened and the board will plan again from your real numbers.
        </Banner>
      )}
      {cycle.status === "failed" && (
        <Banner tone="warn" icon={<AlertTriangle size={17} />}>
          The cycle failed on the engine: {cycle.error || "unknown error"}. Nothing here is made up —
          re-run it from your numbers.
        </Banner>
      )}
      {pollError && <Banner tone="warn" icon={<AlertTriangle size={17} />}>{pollError}</Banner>}
      {summary && summary.llm_ok_months === 0 && cycle.meta?.use_oracle !== false && (
        <Banner tone="warn" icon={<AlertTriangle size={17} />}>
          The AI strategist couldn't be reached for this plan. Every month here comes from the
          board's built-in rules — still grounded in your numbers, just without the strategist's read.
        </Banner>
      )}

      <div className="plan-head">
        <div>
          {running && (
            <p className="subtle">
              Month {Math.min(months.length + 1, horizon)} of {horizon} is being deliberated · {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")} elapsed
            </p>
          )}
        </div>
        {!running && !state.demo && (
          <button className="secondary-button small" type="button" onClick={() => navigate("/analyzing")}>
            <RefreshCw size={14} /> Re-run
          </button>
        )}
      </div>

      <div className="month-strip" style={{ "--cols": horizon }}>
        {Array.from({ length: horizon }, (_, i) => (
          <MonthColumn
            key={i}
            index={i}
            month={months[i] || null}
            baseIso={baseMonth?.enteredAt || new Date().toISOString()}
            dominant={i === 0}
            closed={i === 0 ? closedMonth1 : null}
            onOpenAdvice={analysisForCycle ? () => navigate(`/advice/${analysisForCycle.id}`) : null}
          />
        ))}
      </div>

      <TrajectoryChart months={months} opening={opening} />

      <article className="panel">
        <h3>Your months, then the plan</h3>
        <ol className="timeline forward">
          {state.months.filter((m) => m.index <= (baseMonth?.index ?? 0)).map((m, i, arr) => (
            <HistoryEntry key={m.id} month={m} prev={arr[i - 1] || null} />
          ))}
          <li className="timeline-now"><span>now</span></li>
          {months.map((m, i) => (
            <ProjectedEntry key={m.month_index} index={i} month={m}
                            baseIso={baseMonth?.enteredAt || new Date().toISOString()}
                            closed={i === 0 ? closedMonth1 : null} />
          ))}
        </ol>
      </article>
    </section>
  );
}
