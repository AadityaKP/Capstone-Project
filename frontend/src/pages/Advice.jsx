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
import { deriveCac, deriveLtv, monthName, runwayMonths } from "../derive.js";

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
  const known = [v.mrr, v.cash, v.costs, v.price, v.churnMonthly, v.newCustomers, v.marketingSpend, cac.value, deriveLtv(v), runwayMonths(v)];
  const estimatedCount = (cac.source === "estimated" ? 1 : 0) + (state.company?.maturity ? 0 : 1) + 1; // +1 market conditions

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

      {/* L2 cards */}
      <div className="plan-grid">
        {planCards.map((c) => (
          <PlanCard
            key={c.domain}
            card={c}
            decisionState={decisionFor(c.domain)?.state || null}
            onDecide={state.demo ? null : decide}
          />
        ))}
      </div>

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

      {/* L5 (qualitative in MVP) */}
      {brief?.expected_outcome && (
        <Expandable title="Expected, in simulation">
          <p className="outcome-line">{expectedOutcomeCopy(brief.expected_outcome)}</p>
          <p className="subtle"><SimulatedTag /> — a scenario range from a calibrated simulation, not a forecast of your company.</p>
        </Expandable>
      )}

      {/* L7 */}
      <article className="panel checklist-panel">
        <h3>Next actions</h3>
        <ul className="checklist">
          {planCards.map((c) => {
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
