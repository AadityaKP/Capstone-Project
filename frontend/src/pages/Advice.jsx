// Why this plan — why the board recommends it and how far to trust it
// (docs/ui_simplification_plan.md Phase D). The actions themselves are
// rendered on This month and the Close form; this page carries the reasons,
// the evidence, the assumptions and the loop trace, each one click deep.
//
// Order: notice slot · Summary · Watch-outs / Working in your favour · the
// plan against doing nothing · Evidence · Assumptions · How the board
// weighed it · How the board got here.

import React, { useState } from "react";
import { ChevronRight, RefreshCw } from "lucide-react";
import {
  useStore, latestAnalysis, latestMonth, monthById, cycleById, feedbackForCycleMonth
} from "../store.jsx";
import { expectedOutcomeCopy, scaleWord, FOCUS_LABELS } from "../copy.js";
import {
  Notice, FocusBar, EvidenceList, ConfidenceStrip, RiskBullets, SimulatedTag,
  OefaStrip, monthFromAnalysis, observedLines, Expandable, buildPlanCards
} from "../components.jsx";
import { fillableGuesses } from "./Company.jsx";
import { pickNotice, rulesOnlyMonths } from "../notice.js";
import { deriveCac, deriveLtv, monthName, monthOffsetLabel } from "../derive.js";
import { runwayMonths } from "../founderView.js";
import WhatIfPanel, { WhatIfAssumptions } from "../whatif.jsx";
import { whatif as fetchWhatIf } from "../api.js";
import { loopLines } from "../loopView.js";

export default function Advice({ navigate, params }) {
  const { state } = useStore();

  const analysis = params?.id
    ? state.analyses.find((a) => a.id === params.id) || latestAnalysis(state)
    : latestAnalysis(state);
  const month = analysis ? monthById(state, analysis.monthId) : latestMonth(state);
  const isArchived = analysis && latestAnalysis(state) && analysis.id !== latestAnalysis(state).id;

  // The loop trace (plan section 5.1): every horizon month of the cycle this
  // analysis came from, month 1 with the founder's close when there is one;
  // synthesised from the trace for pre-cycle analyses so the vocabulary is
  // the same on every surface.
  const cycle = analysis?.cycleId ? cycleById(state, analysis.cycleId) : null;
  const cycleMonths = cycle ? (cycle.months || []) : [];
  const traceMonths = cycleMonths.length
    ? cycleMonths.map((m, i) => ({
        key: m.month_index || i + 1,
        title: monthOffsetLabel(month?.enteredAt, i),
        month: m,
        closed: feedbackForCycleMonth(cycle, m.month_index || i + 1)
      }))
    : [{ key: 1, title: null, month: monthFromAnalysis(analysis), closed: null }].filter((t) => t.month);
  const thisMonth = traceMonths.find((t) => t.key === (analysis?.monthIndex || 1)) || traceMonths[0] || null;
  // Deep links: #/advice/:id/m3 opens month 3's strip, #/advice/:id/weighed
  // opens "How the board weighed it".
  const openMonth = /^m(\d+)$/.test(params?.open || "") ? Number(params.open.slice(1)) : null;
  const openWeighed = params?.open === "weighed";

  // What-if projection (D5). Run on demand rather than with the analysis: it is
  // a separate question, and firing it automatically would spend the founder's
  // attention on a counterfactual before they have read the actual advice.
  const [whatIf, setWhatIf] = useState(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState(null);
  const [shockMode, setShockMode] = useState(false);

  async function runWhatIf(shock) {
    setWhatIfLoading(true);
    setWhatIfError(null);
    const response = await fetchWhatIf(state.company, month, analysis, { shockMode: shock });
    setWhatIfLoading(false);
    if (response.ok) {
      setWhatIf(response.data);
    } else {
      setWhatIfError(
        response.offline
          ? "Projection service unreachable — the numbers above are unaffected."
          : response.error
      );
    }
  }

  function toggleShock() {
    const next = !shockMode;
    setShockMode(next);
    if (whatIf) runWhatIf(next);
  }

  if (!analysis || !month) {
    return (
      <section className="empty-state">
        <h2>No analysis yet</h2>
        <p className="narrow">Run the plan from This month and the board's reasons will appear here.</p>
        <button className="primary-button" type="button" onClick={() => navigate("/home")}>
          <RefreshCw size={15} /> This month
        </button>
      </section>
    );
  }

  const brief = analysis.brief;
  const trace = analysis.trace || {};
  const weights = trace.applied_weights || null;
  const topWeightKey = weights
    ? Object.keys(weights).sort((a, b) => weights[b] - weights[a])[0]
    : "innovation";

  const v = month.values;
  const cac = deriveCac(v);
  // runwayMonths is null when the company is not burning cash; a guardrail
  // comparing LLM claims against known numbers must not be handed a null.
  const known = [v.mrr, v.cash, v.costs, v.price, v.churnMonthly, v.newCustomers,
                 v.marketingSpend, cac.value, deriveLtv(v), runwayMonths(v)].filter((n) => n != null);
  // Older stored analyses have no `correctable` flag; treating them as
  // correctable keeps the previous behaviour rather than hiding them.
  const assumedFields = trace.assumed_fields || null;
  const correctable = (assumedFields || []).filter((a) => a.correctable !== false);
  const internalCount = (assumedFields || []).length - correctable.length;

  // The top-focus sentence lives in the Summary; the rest of the reasoning
  // stays under "How the board weighed it".
  const reasoningBullets = [
    scaleWord(trace.marketing_spend_change_pct) ? `Marketing was ${scaleWord(trace.marketing_spend_change_pct)} after the board's risk read.` : null,
    scaleWord(trace.rd_spend_change_pct) ? `Product investment was ${scaleWord(trace.rd_spend_change_pct)} to match retention pressure.` : null,
    trace.hires_change < 0 ? "Hiring was paused at the board's risk level." : null,
    ...(brief?.recommended_focus || []).slice(0, 2).map((f) => `Recommended focus: ${f.toLowerCase?.() || f}.`)
  ].filter(Boolean);

  const observed = observedLines(thisMonth?.month?.observe);
  const hasAssumptions = correctable.length > 0 || internalCount > 0 || (whatIf?.assumptions?.length > 0);
  // "Fill these in" only when Close can actually take the number, and never
  // in sample mode, where the form is disabled.
  const canFill = !state.demo && fillableGuesses(correctable).length > 0;
  // What each advisor said about the domains it is holding: the one place
  // that text appears (the action cards on This month carry their own).
  const held = buildPlanCards(analysis, month).filter((c) => !c.isAction);

  const rulesOnly = cycle
    ? (cycle.meta?.use_oracle === false ? null : rulesOnlyMonths(cycleMonths))
    : (analysis.llm_ok === false ? "all" : null);
  const { notice, inline } = pickNotice({
    rulesOnly,
    archived: isArchived ? { monthName: monthName(month.enteredAt) } : null,
    actions: { current: () => navigate("/home") }
  });

  return (
    <section className="content-stack advice-page">
      {/* notice slot: at most one (notice.js); what lost the slot is one line */}
      <Notice notice={notice} />
      {inline.map((n) => (
        <p key={n.kind} className="subtle inline-note">
          {n.text}
          {n.kind === "archived" && (
            <> <button className="link-button" type="button" onClick={() => navigate("/home")}>Current plan <ChevronRight size={14} /></button></>
          )}
        </p>
      ))}

      {/* Summary */}
      <article className="panel summary-panel">
        <h3>{weights ? `The board's top focus is ${FOCUS_LABELS[topWeightKey]}.` : "The board's read of this month."}</h3>
        <ConfidenceStrip analysis={analysis} month={month} company={state.company} archived={!!isArchived} />
      </article>

      {/* guarded LLM bullets */}
      <RiskBullets brief={brief} knownNumbers={known} />

      {/* D5 — what taking this plan actually does, against doing nothing */}
      <WhatIfPanel
        result={whatIf}
        loading={whatIfLoading}
        error={whatIfError}
        onRun={() => runWhatIf(shockMode)}
        shockMode={shockMode}
        onToggleShock={toggleShock}
      />

      {/* Evidence */}
      <Expandable title="Evidence — what this is based on">
        {observed.length > 0 && (
          <ul className="reason-list">
            {observed.map((l) => <li key={l.key} className={l.muted ? "muted" : ""}>{l.text}</li>)}
          </ul>
        )}
        <EvidenceList analysis={analysis} />
        {brief?.expected_outcome && (
          <div className="outcome-block">
            <p className="outcome-line">{expectedOutcomeCopy(brief.expected_outcome)}</p>
            {/* A single qualitative label the model returns alongside the
                brief — not a simulated range. The modelled range is the
                projection above. */}
            <p className="subtle">
              <SimulatedTag /> — the board's one-line read on the next 6–12 months, not a
              forecast of your company.
            </p>
          </div>
        )}
      </Expandable>

      {/* Assumptions. Interest rate, consumer confidence, unemployment,
          valuation multiple and innovation factor are simulator internals; no
          founder has an opinion on any of them, so they collapse to one
          sentence. What is left is what a founder could genuinely supply. */}
      {hasAssumptions && (
        <Expandable title={correctable.length ? `Assumptions — numbers we guessed (${correctable.length})` : "Assumptions"}>
          {correctable.length > 0 && (
            <>
              <p className="subtle">
                You didn't give us these, so the board used the values below. Each one is
                something you could look up, and each one changes the advice.
              </p>
              <ul className="wi-assumptions">
                {correctable.map((a) => (
                  <li key={a.field}>
                    <strong>{a.field}:</strong> {String(a.value)}
                    <span className="wi-assumption-detail">{a.why}</span>
                  </li>
                ))}
              </ul>
              {canFill && (
                <button className="link-button" type="button" onClick={() => navigate("/update/fill")}>
                  Fill these in <ChevronRight size={15} />
                </button>
              )}
            </>
          )}
          {whatIf?.assumptions?.length > 0 && (
            <>
              <p className="subtle">What the projection above assumed:</p>
              <WhatIfAssumptions assumptions={whatIf.assumptions} />
            </>
          )}
          {internalCount > 0 && (
            <p className="subtle">This analysis also assumes normal market conditions.</p>
          )}
        </Expandable>
      )}

      {/* How the board weighed it */}
      <Expandable title="How the board weighed it" defaultOpen={openWeighed}>
        <FocusBar weights={weights} />
        {reasoningBullets.length > 0 && (
          <ul className="reason-list">
            {reasoningBullets.map((b) => <li key={b}>{b}</li>)}
          </ul>
        )}
        {held.length > 0 && (
          <div className="held-domains">
            <span className="bullets-title">What each advisor said about what to hold</span>
            <ul className="reason-list">
              {held.map((c) => <li key={c.domain}><strong>{c.title} — {c.headline}.</strong> {c.rationale}</li>)}
            </ul>
          </div>
        )}
        {analysis.narratives && (
          <p className="subtle">Each action on This month carries its advisor's own two-sentence reasoning (Settings → richer explanations).</p>
        )}
      </Expandable>

      {/* How the board got here: the OEFA beats for every month of the cycle */}
      {traceMonths.length > 0 && (
        <Expandable title="How the board got here" defaultOpen={openMonth != null}>
          {cycle?.summary && (
            <ul className="trace-lines">
              {loopLines(cycle.summary).map((l) => <li key={l}>{l}</li>)}
            </ul>
          )}
          {cycleMonths.length > 0 && (
            <p className="subtle">
              Month 1 is the decision for this month. Months 2 onward are the model's physics
              compounded, checked against themselves, not a forecast.
            </p>
          )}
          {cycle?.status === "superseded" && (
            <p className="subtle">
              This plan was replaced by a newer one before it finished; the months it never
              reached are not shown.
            </p>
          )}
          <div className="trace-months">
            {traceMonths.map((t) => (
              <OefaStrip
                key={t.key}
                month={t.month}
                closed={t.closed}
                title={t.title}
                defaultOpen={openMonth != null ? t.key === openMonth : t === thisMonth}
                observedInEvidence={t === thisMonth}
              />
            ))}
          </div>
        </Expandable>
      )}
    </section>
  );
}
