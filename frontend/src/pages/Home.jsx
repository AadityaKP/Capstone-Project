// This month — am I OK, what do I do, how did last month go, where does this
// take me (docs/ui_simplification_plan.md Phase C). The Plan page folded into
// here: months land visibly on the Outlook as the board deliberates, and the
// OEFA beats sit one click away under Why this plan.
//
// Top to bottom: notice slot (0–1) · status line · KPI row · Last month
// (conditional) · This month's plan · Outlook. The no-plan / planning /
// failed states render in place; nothing navigates away.

import React from "react";
import { AlertTriangle, ChevronRight, RefreshCw, Workflow } from "lucide-react";
import {
  useStore, latestMonth, previousMonth, latestAnalysis, latestClosedFeedback,
  latestCycle, feedbackForCycleMonth, monthById, answeringCycle
} from "../store.jsx";
import { useCycleRun } from "../cycleRun.jsx";
import {
  deriveCac, deriveLtv, monthDeltas, money, signedPp, daysSince, monthName
} from "../derive.js";
import {
  runwayMonths, runwayLabel, churnLabel, churnPhrase, efficiency,
  showRuleOf40, spendRatioLabel
} from "../founderView.js";
import { positionSentence, DOMAIN_META } from "../copy.js";
import {
  RiskChip, KpiCard, DeltaArrow, Notice, ProgressStages, buildPlanCards, PlanCard, confidenceLine
} from "../components.jsx";
import { predictionSentences, scoreLine, cashDeathMonth, cashDeathSentence, boardChangedLines } from "../loopView.js";
import { pickNotice, rulesOnlyMonths } from "../notice.js";
import Outlook, { defaultOutlookMetric } from "../outlook.jsx";

// Plan section 6.3: the board's prediction error is a real number from the
// close-the-month step, never a narrative. It is shown only when the closed
// cycle was planned on the previous month and the close produced a result.
function lastMonthRecord(state, month, prev, cycle) {
  const closed = latestClosedFeedback(state);
  if (!closed || !prev || closed.cycle.monthId !== prev.id) return null;
  const result = closed.feedback.result;
  const month1 = (closed.cycle.months || [])[0];
  if (!result || !month1) return null;

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

  // "What the board changed" comes from the cycle that answered this close,
  // through the same selector History uses (store.answeringCycle).
  const changed = boardChangedLines(answeringCycle(state, closed.cycle));
  return { lines, score: scoreLine(result.prediction_error), changed, monthId: prev.id };
}

// At most two adaptation sentences on the card; the rest live on the
// History entry.
const CHANGED_CAP = 2;

// What each advisor said about a held domain is third-layer text; it lives
// under Why this plan → How the board weighed it, not on the main screen.
function NoActionPanel({ analysisId, navigate }) {
  return (
    <article className="panel no-action-panel">
      <h3>Nothing to change this month</h3>
      <p className="subtle">
        The board isn't asking you to spend, hire or move price. Hold what you're
        doing and close the month when it ends.
      </p>
      <button className="link-button" type="button" onClick={() => navigate(`/advice/${analysisId}/weighed`)}>
        What each advisor said <ChevronRight size={15} />
      </button>
    </article>
  );
}

export default function Home({ navigate }) {
  const { state } = useStore();
  const { start, starting, startError, pollError, elapsed } = useCycleRun();
  const month = latestMonth(state);
  const prev = previousMonth(state);
  if (!month) return null;

  const demo = !!state.demo;
  const cycle = latestCycle(state);
  const cycleIsCurrent = !!cycle && cycle.monthId === month.id;
  const running = cycleIsCurrent && !demo && ["queued", "running"].includes(cycle.status);
  const failed = cycleIsCurrent && cycle.status === "failed";
  const cycleMonths = cycle?.months || [];
  const horizon = cycle?.horizon || cycle?.summary?.horizon_months || 4;
  const cycleAnalysis = cycle ? state.analyses.find((a) => a.cycleId === cycle.id) : null;

  // The plan on screen: while a new cycle deliberates, only its own month 1
  // counts (the previous plan is not shown as if it were this month's).
  const latest = latestAnalysis(state);
  const analysis = running ? cycleAnalysis : latest;
  const analysisIsCurrent = !!analysis && analysis.monthId === month.id;
  // The status line keeps the last read while a new plan is in progress; the
  // progress itself is shown once, in the plan section.
  const statusAnalysis = analysis || latest;
  const brief = statusAnalysis?.brief;
  const topWeightKey = statusAnalysis?.trace?.applied_weights
    ? Object.keys(statusAnalysis.trace.applied_weights).sort(
        (a, b) => statusAnalysis.trace.applied_weights[b] - statusAnalysis.trace.applied_weights[a]
      )[0]
    : "innovation";

  const v = month.values;
  const runway = runwayMonths(v);
  const cac = deriveCac(v);
  const ltv = deriveLtv(v);
  const eff = efficiency(ltv, cac.value, v.newCustomers);
  const deltas = monthDeltas(month, prev);
  const age = daysSince(month.enteredAt);
  const runwayWatch = runway !== null && runway < 12;
  const efficiencyWatch = eff.band === "unhealthy";
  const revenueFalling = deltas?.mrrPct != null && deltas.mrrPct < 0;

  const planCards = analysis ? buildPlanCards(analysis, month) : [];
  const actionCards = planCards.filter((c) => c.isAction);
  const holding = planCards.filter((c) => !c.isAction).map((c) => c.title.toLowerCase());
  const deathMonth = cycleIsCurrent ? cashDeathMonth(cycleMonths) : null;
  const record = lastMonthRecord(state, month, prev, cycle);
  // Rules-only: from the current cycle's months when there is one (whole or
  // partial), else from the current analysis. A cycle run deliberately
  // without the strategist (meta.use_oracle false) is not a failure.
  const rulesOnly = cycleIsCurrent
    ? (cycle.meta?.use_oracle === false ? null
      : rulesOnlyMonths(cycleMonths) || (cycle.summary?.llm_ok_months === 0 ? "all" : null))
    : (analysisIsCurrent && analysis.llm_ok === false ? "all" : null);
  // The plan on screen was made on an earlier month: one inline sentence in
  // the plan section, not a banner.
  const stale = !running && !!analysis && !analysisIsCurrent;

  const runLabel = cycle ? "Re-run" : "Run the plan";
  const runButton = !demo && !running && (
    <button className="primary-button small" type="button" disabled={starting} onClick={() => start()}>
      <RefreshCw size={14} /> {starting ? "Starting…" : runLabel}
    </button>
  );

  // ---- notice slot: at most one (notice.js) ----
  const { notice, inline } = pickNotice({
    startError,
    failed: failed ? { error: cycle.error } : null,
    pollError,
    rulesOnly,
    actions: {
      retry: () => start(),
      rerun: demo ? null : () => start()
    }
  });
  const planNotes = [
    stale ? "On your previous numbers until the board plans again." : null,
    ...inline.filter((n) => n.kind === "rules-only").map((n) => n.text)
  ].filter(Boolean);
  // Re-run is offered only when it would change something: the plan is on
  // older numbers or came from the built-in rules. A failed cycle's Re-run
  // sits in its notice.
  const offerRerun = !running && !demo && !failed && !startError && (stale || !!rulesOnly);

  return (
    <section className="content-stack this-month">
      {/* 1 · notice slot */}
      <Notice notice={notice} />

      {/* 2 · status line */}
      <div className={`position-banner static ${brief ? (brief.risk_level || "MEDIUM").toLowerCase() : "none"}`}>
        <div className="position-line">
          {brief && <RiskChip level={brief.risk_level} large />}
          <strong>
            {statusAnalysis
              ? positionSentence({ ...brief, _topFocus: topWeightKey })
              : running
                ? "Your first plan is on its way."
                : "No plan yet — run your first one."}
          </strong>
        </div>
        <span className="position-basis">
          Based on your {monthName(month.enteredAt)} numbers
          {running ? " · new plan in progress" : age != null && age > 0 ? ` · ${age} day${age === 1 ? "" : "s"} ago` : ""}
        </span>
      </div>

      {/* 3 · KPI row */}
      <div className="kpi-grid founder-grid">
        <KpiCard
          label="Cash lasts" value={runwayLabel(v)}
          delta={prev && deltas?.runway != null ? <DeltaArrow value={deltas.runway} format={(x) => `${x > 0 ? "+" : ""}${x.toFixed(1)} mo`} /> : null}
          sub={runway === null ? "revenue covers your costs" : "at your current costs"}
          hint="Cash in the bank divided by what you spend each month beyond what you earn, assuming both stay flat."
          band={runwayWatch ? "watch" : null}
        />
        <KpiCard
          label="Revenue" value={money(v.mrr)}
          delta={prev ? <DeltaArrow value={deltas?.mrrPct} /> : null}
          sub="per month"
          hint={[
            "Monthly recurring revenue.",
            spendRatioLabel(v) ? `You spend ${spendRatioLabel(v)} earned.` : null,
            showRuleOf40(v.mrr) ? null
              : "Rule of 40, the usual SaaS benchmark, doesn't mean anything below about $1M a year."
          ].filter(Boolean).join(" ")}
        />
        <KpiCard
          label="Customers lost" value={churnLabel(v.churnMonthly)}
          delta={prev ? <DeltaArrow value={deltas?.churnPp} goodWhenDown format={signedPp} /> : null}
          sub="every month"
          hint={churnPhrase(v.churnMonthly)}
        />
        <KpiCard
          label="Winning customers" value={eff.label}
          sub={eff.band === "unknown" ? "" : "lifetime value vs cost to win"}
          hint={`${eff.detail} Healthy when what a customer pays back over their life is at least 3× what they cost to win.`}
          band={efficiencyWatch ? "watch" : null}
        />
      </div>

      {/* 4 · last month */}
      {record && (
        <article className="panel scored-panel">
          <div className="panel-title-row">
            <h3>How last month's plan held up</h3>
            <button className="link-button" type="button" onClick={() => navigate(`/history/${record.monthId}`)}>
              Details <ChevronRight size={15} />
            </button>
          </div>
          <ul className="changed-list">
            {record.lines.map((l) => <li key={l.key} className={`pe-line ${l.tone}`}>{l.text}</li>)}
            <li className="muted">{record.score}</li>
          </ul>
          {record.changed && (
            <p className="board-changed">
              <strong>What the board changed:</strong> {record.changed.slice(0, CHANGED_CAP).join(" ")}
              {record.changed.length > CHANGED_CAP && (
                <> <button className="link-button" type="button" onClick={() => navigate(`/history/${record.monthId}`)}>
                  +{record.changed.length - CHANGED_CAP} more <ChevronRight size={14} />
                </button></>
              )}
            </p>
          )}
        </article>
      )}

      {/* 5 · this month's plan, or the state that stands in for it */}
      {running && !analysis && (
        <article className="panel planning-panel">
          <h3>Your board is planning</h3>
          <ProgressStages stage={Math.min(cycleMonths.length, 2)} narrativesOn={!!state.settings.narratives} />
          <p className="subtle elapsed">
            Month {Math.min(cycleMonths.length + 1, horizon)} of {horizon} · {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
          </p>
        </article>
      )}

      {!running && !analysis && !failed && (
        <article className="panel empty-plan">
          <Workflow size={28} className="warn-icon" />
          <h3>No plan yet</h3>
          <p className="subtle narrow">
            The board plans {horizon} months at a time: it decides this month, simulates what follows,
            checks its own prediction and adapts before the next month. Run it on your current numbers.
          </p>
          {runButton}
        </article>
      )}

      {analysis && (
        <>
          <div className="plan-section-head">
            <span className="plan-confidence">{confidenceLine(analysis, month, state.company)}</span>
            <div className="plan-section-actions">
              {running && (
                <span className="subtle elapsed">
                  Month {Math.min(cycleMonths.length + 1, horizon)} of {horizon} · {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
                </span>
              )}
              {offerRerun && (
                <button className="link-button" type="button" disabled={starting} onClick={() => start()}>
                  <RefreshCw size={14} /> {stale ? "Plan again" : "Re-run"}
                </button>
              )}
              <button className="link-button" type="button" onClick={() => navigate(`/advice/${analysis.id}`)}>
                Why this plan <ChevronRight size={15} />
              </button>
            </div>
          </div>
          {planNotes.length > 0 && (
            <p className="plan-note">{planNotes.join(" ")}</p>
          )}
          {actionCards.length > 0 ? (
            <div className="plan-grid">
              {actionCards.map((c) => <PlanCard key={c.domain} card={c} />)}
            </div>
          ) : (
            <NoActionPanel analysisId={analysis.id} navigate={navigate} />
          )}
          {actionCards.length > 0 && holding.length > 0 && (
            <p className="holding-line">Holding: {holding.join(", ")}.</p>
          )}
          {deathMonth != null && (
            <p className="cash-death-line">
              <AlertTriangle size={14} /> {cashDeathSentence(deathMonth, horizon)}
            </p>
          )}
        </>
      )}

      {/* 6 · outlook — grows month by month as the cycle lands; a cycle made
          on the previous month keeps its own base month and shows the close
          as a marked point */}
      {cycle && cycleMonths.length > 0 && (
        <Outlook
          key={cycle.id}
          cycle={cycle}
          baseIso={(monthById(state, cycle.monthId) || month).enteredAt}
          closedMonth1={feedbackForCycleMonth(cycle, 1)}
          initialMetric={defaultOutlookMetric({ revenueFalling })}
          analysisId={cycleAnalysis?.id || null}
          navigate={navigate}
        />
      )}
    </section>
  );
}
