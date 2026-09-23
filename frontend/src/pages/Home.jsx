// S5 Home — "where am I, what should I do, what changed" in 30 seconds (spec §9).

import React from "react";
import { ChevronRight, RefreshCw } from "lucide-react";
import {
  useStore, latestMonth, previousMonth, latestAnalysis, latestClosedFeedback
} from "../store.jsx";
import {
  deriveCac, deriveLtv, monthDeltas,
  money, signedPp, daysSince, dateLabel
} from "../derive.js";
import {
  runwayMonths, runwayLabel, churnLabel, churnPhrase, efficiency,
  showRuleOf40, spendRatioLabel
} from "../founderView.js";
import { positionSentence, DOMAIN_META } from "../copy.js";
import { RiskChip, KpiCard, DeltaArrow, Banner, buildPlanCards, PlanCard } from "../components.jsx";
import { predictionSentences } from "../loopView.js";

// Plan section 6.3: the board's prediction error lives on the "What changed"
// panel, in the same list as "Revenue grew 4%", because that is where the
// founder already looks. It is a real number from the close-the-month step,
// never a narrative.
function predictionErrorLines(state, prev) {
  const closed = latestClosedFeedback(state);
  if (!closed || !prev || closed.cycle.monthId !== prev.id) return [];
  const result = closed.feedback.result;
  const month1 = (closed.cycle.months || [])[0];
  if (!result || !month1) return [];
  const lines = predictionSentences({
    before: month1.observe?.state_before,
    actual: result.actual_state,
    expected: month1.execute?.expected_delta,
    error: result.prediction_error
  }).map((s) => ({ key: s.key, tone: s.tone, text: s.text }));
  for (const item of result.per_action || []) {
    if (item.done === "didnt") {
      lines.push({
        key: `skip-${item.action_key}`, tone: "muted",
        text: `You didn't do the ${DOMAIN_META[item.action_key]?.title.toLowerCase() || item.action_key} step, so we're not counting this month as evidence about it.`
      });
    }
  }
  if (result.evidence && !result.evidence.written && result.evidence.credited?.length) {
    lines.push({ key: "evidence", tone: "muted", text: `Nothing was written back as evidence: ${result.evidence.reason}.` });
  }
  return lines;
}

export default function Home({ navigate }) {
  const { state } = useStore();
  const month = latestMonth(state);
  const prev = previousMonth(state);
  const analysis = latestAnalysis(state);
  const analysisIsCurrent = analysis && month && analysis.monthId === month.id;
  const errorLines = predictionErrorLines(state, prev);

  if (!month) return null;

  const v = month.values;
  const runway = runwayMonths(v);
  const cac = deriveCac(v);
  const ltv = deriveLtv(v);
  const eff = efficiency(ltv, cac.value, v.newCustomers);
  const deltas = monthDeltas(month, prev);
  const age = daysSince(month.enteredAt);
  const stale = age != null && age > 35;

  const brief = analysis?.brief;
  const topWeightKey = analysis?.trace?.applied_weights
    ? Object.keys(analysis.trace.applied_weights).sort(
        (a, b) => analysis.trace.applied_weights[b] - analysis.trace.applied_weights[a]
      )[0]
    : "innovation";

  const planCards = analysisIsCurrent ? buildPlanCards(analysis, month) : [];

  return (
    <section className="content-stack">
      {/* 1 · position banner */}
      <button
        type="button"
        className={`position-banner ${brief ? (brief.risk_level || "MEDIUM").toLowerCase() : "none"}`}
        onClick={() => navigate("/plan")}
      >
        <div className="position-line">
          {brief && <RiskChip level={brief.risk_level} large />}
          <strong>
            {analysisIsCurrent
              ? positionSentence({ ...brief, _topFocus: topWeightKey })
              : analysis
                ? "Your numbers changed since the last analysis — run a fresh one."
                : "No analysis yet — run your first one."}
          </strong>
        </div>
        <span className="position-cta">
          {stale && <em className="stale-note">based on numbers from {dateLabel(month.enteredAt)} · </em>}
          Details <ChevronRight size={15} />
        </span>
      </button>

      {!analysisIsCurrent && (
        <Banner
          tone="info"
          actions={
            <button className="primary-button small" type="button" onClick={() => navigate("/analyzing")}>
              <RefreshCw size={14} /> {analysis ? "Re-analyse" : "Run analysis"}
            </button>
          }
        >
          {analysis
            ? "The plan below reflects your previous numbers until you re-analyse."
            : "The board hasn't reviewed these numbers yet."}
        </Banner>
      )}

      {/* Plan section 8.1: lead with the prediction error, do not bury it. */}
      {errorLines.length > 0 && (
        <article className="panel scored-panel">
          <div className="panel-title-row">
            <h3>How last month's plan held up</h3>
            <button className="link-button" type="button" onClick={() => navigate("/plan")}>
              What the board changed <ChevronRight size={15} />
            </button>
          </div>
          <ul className="changed-list">
            {errorLines.map((l) => <li key={l.key} className={`pe-line ${l.tone}`}>{l.text}</li>)}
          </ul>
        </article>
      )}

      {/* 2 · KPI row */}
      <div className="kpi-grid founder-grid">
        <KpiCard
          label="Cash lasts" value={runwayLabel(v)}
          delta={prev && deltas?.runway != null ? <DeltaArrow value={deltas.runway} format={(x) => `${x > 0 ? "+" : ""}${x.toFixed(1)} mo`} /> : null}
          sub={runway === null ? "revenue covers your costs" : "at your current costs"}
          hint="Cash in the bank divided by what you spend each month beyond what you earn, assuming both stay flat."
          band={runway !== null && runway < 12 ? "watch" : null}
        />
        <KpiCard
          label="Revenue" value={money(v.mrr)}
          delta={prev ? <DeltaArrow value={deltas?.mrrPct} /> : null}
          sub={spendRatioLabel(v) ? `you spend ${spendRatioLabel(v)} earned` : "per month"}
          hint={showRuleOf40(v.mrr)
            ? "Monthly recurring revenue."
            : "Monthly recurring revenue, and what you spend for each dollar of it. Rule of 40, the usual SaaS benchmark, doesn't mean anything below about $1M a year."}
        />
        <KpiCard
          label="Customers lost" value={churnLabel(v.churnMonthly)}
          delta={prev ? <DeltaArrow value={deltas?.churnPp} goodWhenDown format={signedPp} /> : null}
          sub="every month"
          hint={churnPhrase(v.churnMonthly)}
        />
        <KpiCard
          label="Winning customers" value={eff.label}
          sub={eff.detail}
          hint="Healthy when what a customer pays back over their life is at least 3× what they cost to win."
          band={eff.band === "unhealthy" ? "watch" : null}
        />
      </div>

      {/* 3 · this month's plan */}
      {planCards.length > 0 && (
        <article className="panel">
          <div className="panel-title-row">
            <h3>This month's plan</h3>
            <button className="link-button" type="button" onClick={() => navigate("/plan")}>
              The next {4} months <ChevronRight size={15} />
            </button>
          </div>
          <div className="plan-compact-grid">
            {planCards.map((c) => <PlanCard key={c.domain} card={c} compact />)}
          </div>
        </article>
      )}

    </section>
  );
}
