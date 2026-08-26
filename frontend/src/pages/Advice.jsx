// S6/S7 Advice — the full monthly analysis in seven layers with progressive
// disclosure (spec §10). Collapsed default: verdict, four cards, checklist.

import React, { useMemo, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, RefreshCw } from "lucide-react";
import {
  useStore, latestAnalysis, latestMonth, monthById, analysisForMonth, uid
} from "../store.jsx";
import { positionSentence, expectedOutcomeCopy, scaleWord, FOCUS_LABELS } from "../copy.js";
import {
  RiskChip, Banner, buildPlanCards, PlanCard, FocusBar, EvidenceList,
  ConfidenceStrip, RiskBullets, SimulatedTag
} from "../components.jsx";
import { deriveCac, deriveLtv, monthName } from "../derive.js";
import { runwayMonths } from "../founderView.js";
import WhatIfPanel from "../whatif.jsx";
import { whatif as fetchWhatIf } from "../api.js";

function Expandable({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <article className={`panel expandable ${open ? "open" : ""}`}>
      <button className="expand-head" type="button" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={17} /> : <ChevronRight size={17} />}
        <h3>{title}</h3>
      </button>
      {open && <div className="expand-body">{children}</div>}
    </article>
  );
}

export default function Advice({ navigate, params }) {
  const { state, dispatch } = useStore();

  const analysis = params?.id
    ? state.analyses.find((a) => a.id === params.id) || latestAnalysis(state)
    : latestAnalysis(state);
  const month = analysis ? monthById(state, analysis.monthId) : latestMonth(state);
  const isArchived = analysis && latestAnalysis(state) && analysis.id !== latestAnalysis(state).id;

  const planCards = useMemo(() => buildPlanCards(analysis, month), [analysis, month]);

  // What-if projection (D5). Run on demand rather than with the analysis: it is
  // a separate question, and firing it automatically would spend the founder's
  // attention on a counterfactual before they have read the actual advice.
  const [whatIf, setWhatIf] = useState(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState(null);
  const [shockMode, setShockMode] = useState(false);
  const [showHeld, setShowHeld] = useState(false);

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
        <p className="narrow">Run your first analysis and the board's advice will appear here.</p>
        <button className="primary-button" type="button" onClick={() => navigate("/analyzing")}>
          <RefreshCw size={15} /> Run analysis
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
  // The server reports what it actually assumed (trace.assumed_fields). The old
  // client-side count could not see the macro fields the server fills in and so
  // understated them; it stays only as a fallback for analyses stored before
  // that field existed.
  const assumedFields = trace.assumed_fields || null;
  const estimatedCount = assumedFields
    ? assumedFields.length
    : (cac.source === "estimated" ? 1 : 0) + (state.company?.maturity ? 0 : 1) + 1;
  // Older stored analyses have no `correctable` flag; treating them as
  // correctable keeps the previous behaviour rather than hiding them.
  const correctable = (assumedFields || []).filter((a) => a.correctable !== false);
  const internalCount = (assumedFields || []).length - correctable.length;

  const decisions = month.decisions || [];
  const decisionFor = (domain) => decisions.find((d) => d.domain === domain) || null;

  function decide(card, nextState) {
    const existing = decisionFor(card.domain);
    dispatch({
      type: "SET_DECISION",
      monthId: month.id,
      decision: {
        id: existing?.id || uid("d"),
        domain: card.domain,
        text: card.headline,
        state: nextState
      }
    });
  }

  const reasoningBullets = [
    weights ? `The board's top focus is ${FOCUS_LABELS[topWeightKey]}.` : null,
    scaleWord(trace.marketing_spend_change_pct) ? `Marketing was ${scaleWord(trace.marketing_spend_change_pct)} after the board's risk read.` : null,
    scaleWord(trace.rd_spend_change_pct) ? `Product investment was ${scaleWord(trace.rd_spend_change_pct)} to match retention pressure.` : null,
    trace.hires_change < 0 ? "Hiring was paused at the board's risk level." : null,
    ...(brief?.recommended_focus || []).slice(0, 2).map((f) => `Recommended focus: ${f.toLowerCase?.() || f}.`)
  ].filter(Boolean);

  const nextUpdate = (() => {
    const d = new Date(month.enteredAt);
    d.setMonth(d.getMonth() + 1, 1);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  })();

  return (
    <section className="content-stack advice-page">
      {isArchived && (
        <Banner tone="info" actions={
          <button className="secondary-button small" type="button" onClick={() => navigate("/advice")}>Latest advice</button>
        }>
          Archived analysis from {monthName(month.enteredAt)} — shown as it was.
        </Banner>
      )}

      {analysis.llm_ok === false && (
        <Banner tone="warn" icon={<AlertTriangle size={17} />}>
          The AI strategist couldn't be reached for this analysis. This plan comes from the
          board's built-in rules — still grounded in your numbers, just without the
          strategist's read. Re-analyse when the service is back.
        </Banner>
      )}

      {/* L1 */}
      <div className={`position-banner static ${(brief?.risk_level || "MEDIUM").toLowerCase()}`}>
        <div className="position-line">
          <RiskChip level={brief?.risk_level} large />
          <strong>{positionSentence({ ...brief, _topFocus: topWeightKey })}</strong>
        </div>
      </div>

      {/* L6 strip */}
      <ConfidenceStrip analysis={analysis} month={month} estimatedCount={estimatedCount} />

      {/* guarded LLM bullets */}
      <RiskBullets brief={brief} knownNumbers={known} />

      {/* L2 cards. When the board's recommendation is to change nothing, say so
          once rather than rendering four cards that each ask the founder to
          confirm they did nothing — which is also what was polluting the
          accepted-action signal memory.py learns from. */}
      {planCards.every((c) => !c.isAction) ? (
        <article className="panel no-action-panel">
          <h3>Nothing to change this month</h3>
          <p className="subtle">
            The board isn't asking you to spend, hire or move price. Hold what you're
            doing and update your numbers next month.
          </p>
          <button className="link-button" type="button" onClick={() => setShowHeld(!showHeld)}>
            {showHeld ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
            {showHeld ? "Hide" : "Show"} what each advisor said
          </button>
          {showHeld && (
            <div className="plan-grid">
              {planCards.map((c) => <PlanCard key={c.domain} card={c} compact />)}
            </div>
          )}
        </article>
      ) : (
        <div className="plan-grid">
          {planCards.map((c) => (
            <PlanCard
              key={c.domain}
              card={c}
              decisionState={decisionFor(c.domain)?.state || null}
              onDecide={state.demo || !c.isAction ? null : decide}
            />
          ))}
        </div>
      )}

      {/* D5 — what taking this plan actually does, against two baselines */}
      <WhatIfPanel
        result={whatIf}
        loading={whatIfLoading}
        error={whatIfError}
        onRun={() => runWhatIf(shockMode)}
        shockMode={shockMode}
        onToggleShock={toggleShock}
      />

      {/* L3 */}
      <Expandable title="Why this plan">
        <FocusBar weights={weights} />
        <ul className="reason-list">
          {reasoningBullets.map((b) => <li key={b}>{b}</li>)}
        </ul>
        {analysis.narratives && (
          <p className="subtle">Each card above carries its advisor's own reasoning.</p>
        )}
      </Expandable>

      {/* L4 */}
      <Expandable title="Evidence — what this is based on">
        <EvidenceList analysis={analysis} />
      </Expandable>

      {/* Nothing silently assumed: every field the founder did not supply, with
          the value used and why, as reported by the server that used it. */}
      {/* Split, not listed. Interest rate, consumer confidence, unemployment,
          valuation multiple and innovation factor are simulator internals; no
          founder has an opinion on any of them, and inviting one to "enter
          anything here you actually know" invited an invented number into the
          analysis. They collapse to one sentence. What is left is what a
          founder could genuinely supply, and is worth asking for. */}
      {correctable.length > 0 && (
        <Expandable title={`Numbers we guessed (${correctable.length})`}>
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
          <button className="link-button" type="button" onClick={() => navigate("/update")}>
            Fill these in <ChevronRight size={15} />
          </button>
          {internalCount > 0 && (
            <p className="subtle">This also assumes normal market conditions.</p>
          )}
        </Expandable>
      )}
      {correctable.length === 0 && internalCount > 0 && (
        <p className="subtle assumption-line">This analysis assumes normal market conditions.</p>
      )}

      {/* L5 (qualitative in MVP) */}
      {brief?.expected_outcome && (
        <Expandable title="Expected, in simulation">
          <p className="outcome-line">{expectedOutcomeCopy(brief.expected_outcome)}</p>
          {/* This is a single qualitative label the model returns alongside the
              brief — not a simulated range. The earlier copy here described it as
              "a scenario range from a calibrated simulation", which was the
              projection panel's claim, not this one's. The projection is above. */}
          <p className="subtle">
            <SimulatedTag /> — the board's one-line read on the next 6–12 months, not a
            forecast of your company. For a modelled range, see the projection above.
          </p>
        </Expandable>
      )}

      {/* L7 */}
      <article className="panel checklist-panel">
        <h3>Next actions</h3>
        <ul className="checklist">
          {planCards.filter((c) => c.isAction).map((c) => {
            const d = decisionFor(c.domain);
            const on = d?.state === "accepted";
            return (
              <li key={c.domain}>
                <button
                  type="button"
                  className={`check-item ${on ? "on" : ""}`}
                  onClick={() => !state.demo && decide(c, on ? "suggested" : "accepted")}
                  disabled={state.demo}
                >
                  <span className="checkbox">{on ? "✓" : ""}</span>
                  {c.headline}
                </button>
              </li>
            );
          })}
          <li>
            <button type="button" className="check-item" onClick={() => navigate("/update")}>
              <span className="checkbox" />
              Update numbers around {nextUpdate}
            </button>
          </li>
        </ul>
      </article>
    </section>
  );
}
