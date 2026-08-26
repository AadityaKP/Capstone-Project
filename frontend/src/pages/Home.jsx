// S5 Home — "where am I, what should I do, what changed" in 30 seconds (spec §9).

import React from "react";
import { ChevronRight, FlaskConical, RefreshCw } from "lucide-react";
import {
  useStore, latestMonth, previousMonth, latestAnalysis
} from "../store.jsx";
import {
  deriveCac, deriveLtv, monthDeltas,
  money, signedPct, signedPp, daysSince, dateLabel, eventTrigger
} from "../derive.js";
import {
  runwayMonths, runwayLabel, churnLabel, churnPhrase, efficiency,
  showRuleOf40, spendRatioLabel
} from "../founderView.js";
import { positionSentence, refreshReasonCopy, expectedOutcomeCopy, OUTCOME } from "../copy.js";
import { RiskChip, KpiCard, DeltaArrow, Banner, buildPlanCards, PlanCard, SimulatedTag } from "../components.jsx";

export default function Home({ navigate }) {
  const { state } = useStore();
  const month = latestMonth(state);
  const prev = previousMonth(state);
  const analysis = latestAnalysis(state);
  const analysisIsCurrent = analysis && month && analysis.monthId === month.id;

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
  const memories = analysis?.trace?.retrieved_memories || [];
  const outcomeCopy = brief?.expected_outcome ? expectedOutcomeCopy(brief.expected_outcome) : null;

  const changeReason = analysisIsCurrent
    ? refreshReasonCopy(analysis.trace?.refresh_reason || analysis.reason)
    : null;

  return (
    <section className="content-stack">
      {/* 1 · position banner */}
      <button
        type="button"
        className={`position-banner ${brief ? (brief.risk_level || "MEDIUM").toLowerCase() : "none"}`}
        onClick={() => navigate("/advice")}
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
            <button className="link-button" type="button" onClick={() => navigate("/advice")}>
              Full advice <ChevronRight size={15} />
            </button>
          </div>
          <div className="plan-compact-grid">
            {planCards.map((c) => <PlanCard key={c.domain} card={c} compact />)}
          </div>
        </article>
      )}

      {/* 4 · what changed */}
      {(prev || changeReason) && (
        <article className="panel changed-panel">
          <h3>What changed</h3>
          <ul className="changed-list">
            {prev && deltas?.mrrPct != null && <li>Revenue {deltas.mrrPct >= 0 ? "grew" : "fell"} {signedPct(deltas.mrrPct)} since last update.</li>}
            {prev && deltas?.churnPp != null && Math.abs(deltas.churnPp) >= 0.05 && (
              <li>Churn {deltas.churnPp > 0 ? "rose" : "improved"} {signedPp(Math.abs(deltas.churnPp) * (deltas.churnPp > 0 ? 1 : -1))}.</li>
            )}
            {changeReason && <li>Last analysis: {changeReason}.</li>}
          </ul>
          <button className="link-button" type="button" onClick={() => navigate("/history")}>
            See history <ChevronRight size={15} />
          </button>
        </article>
      )}

      {/* 5 · evidence peek */}
      {analysisIsCurrent && (outcomeCopy || memories.length > 0) && (
        <button type="button" className="evidence-peek" onClick={() => navigate("/advice")}>
          <FlaskConical size={15} />
          <span>
            {outcomeCopy || "The board weighed similar simulated situations for this plan."}
            {" "}<em>Simulated scenarios, not real companies.</em>
          </span>
          <ChevronRight size={15} />
        </button>
      )}

      {/* 6 · freshness footer */}
      <footer className={`freshness-footer ${stale ? "stale" : ""}`}>
        <span>
          Numbers from {dateLabel(month.enteredAt)}
          {age != null && age > 0 && ` · ${age} day${age === 1 ? "" : "s"} ago`} · Update takes ~2 minutes
        </span>
        <button className={stale ? "primary-button small" : "secondary-button small"} type="button" onClick={() => navigate("/update")}>
          Update my numbers
        </button>
      </footer>
    </section>
  );
}
