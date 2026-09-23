// S10 History — one vertical monthly timeline (spec §14): numbers → advice →
// decisions → how the plan held up → matured outcomes. Mini-charts appear
// once 3 snapshots exist. Each entry expands to its own record of the plan
// made on that month (docs/ui_simplification_plan.md Phase F.1).

import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  useStore, analysisForMonth, cycleForMonth, feedbackForCycleMonth
} from "../store.jsx";
import {
  money, pct, signedPct, signedPp, monthName, monthDeltas, monthsLabel
} from "../derive.js";
import { runwayMonths, runwayLabel } from "../founderView.js";
import { RiskChip, MiniLine, OutcomeBadge } from "../components.jsx";
import { predictionSentences, scoreLine } from "../loopView.js";

// Mirror of the engine's outcome labelling (classify_realized_outcome): ±10% MRR
// over a 6-month horizon. Pure arithmetic on the founder's own numbers.
function maturedOutcome(months, index) {
  const source = months[index];
  const future = months.find((m) => m.index >= source.index + 6);
  if (!future) return null;
  const change = (future.values.mrr - source.values.mrr) / Math.max(source.values.mrr, 1);
  if (change > 0.10) return "GROWTH";
  if (change < -0.10) return "DECLINE";
  return "STAGNATION";
}

// How the plan made on `month` held up: the close's scored prediction, and
// what the board did with it in the cycle planned on the following month.
function heldUp(state, month, next) {
  const cycle = cycleForMonth(state, month.id);
  const closed = cycle ? feedbackForCycleMonth(cycle, 1) : null;
  const month1 = cycle?.months?.[0];
  if (!closed?.result || !month1) return null;
  const result = closed.result;
  const lines = predictionSentences({
    before: month1.observe?.state_before,
    actual: result.actual_state,
    expected: month1.execute?.expected_delta,
    error: result.prediction_error
  });
  const answering = next ? cycleForMonth(state, next.id) : null;
  const adapt = answering && answering.id !== cycle.id ? answering.months?.[0]?.adapt : null;
  let changed = null;
  if (adapt) {
    const adaptations = adapt.adaptations || [];
    if (adaptations.length) changed = adaptations.map((a) => `${a.agent}: ${a.sentence}`);
    else if (answering.startedFromTrackRecord) changed = ["The board planned the next month with this result in hand."];
  }
  return { lines, score: scoreLine(result.prediction_error), changed };
}

function MonthEntry({ month, prev, next, analysis, outcome, navigate }) {
  const { state } = useStore();
  const [open, setOpen] = useState(false);
  const v = month.values;
  const deltas = monthDeltas(month, prev);
  // One decision per domain, last entry wins: the Close form appends its
  // answer after any earlier toggle, so its answer is the one that counts.
  // Legacy "suggested" rows (a toggle switched back off) still render, with
  // ○, but are not something the founder was asked about, so they leave the
  // denominator.
  const byDomain = new Map();
  for (const d of month.decisions || []) byDomain.set(d.domain || d.id, d);
  const decisions = [...byDomain.values()];
  const asked = decisions.filter((d) => d.state !== "suggested");
  const did = asked.filter((d) => d.state === "accepted").length;
  const partly = asked.filter((d) => d.state === "custom").length;
  const record = heldUp(state, month, next);

  return (
    <li className="timeline-entry">
      <div className="timeline-rail"><i /></div>
      <div className="timeline-card">
        <button className="timeline-head" type="button" onClick={() => setOpen(!open)}>
          <span className="timeline-month">{monthName(month.enteredAt)}</span>
          {analysis && <RiskChip level={analysis.brief?.risk_level} />}
          {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </button>
        <p className="timeline-numbers">
          MRR {money(v.mrr)}{deltas?.mrrPct != null && <em> ({signedPct(deltas.mrrPct)})</em>}
          {" · "}churn {pct(v.churnMonthly)}{deltas?.churnPp != null && Math.abs(deltas.churnPp) >= 0.05 && <em> ({signedPp(deltas.churnPp)})</em>}
          {" · "}cash lasts {runwayLabel(v)}
        </p>
        {analysis?.brief?.recommended_focus?.length > 0 && (
          <p className="timeline-plan">Plan: {analysis.brief.recommended_focus.join(" · ").toLowerCase()}</p>
        )}
        {asked.length > 0 && (
          <p className="timeline-decisions">
            Did {did} of {asked.length}{partly > 0 && ` · partly ${partly}`}
          </p>
        )}
        {open && (
          <div className="timeline-detail">
            <ul>
              <li>Cash {money(v.cash)} · costs {money(v.costs)}/mo · price ${v.price}/user</li>
              {v.newCustomers != null && <li>{v.newCustomers} new customers · marketing {money(v.marketingSpend)}</li>}
              {decisions.map((d) => (
                <li key={d.id} className={`decision-line ${d.state}`}>
                  {d.state === "accepted" ? "✓" : d.state === "custom" ? "✎" : "○"} {d.text}
                  {d.note && <em> — {d.note}</em>}
                </li>
              ))}
            </ul>
            {record && (
              <div className="held-up">
                <span className="held-up-title">How the plan held up</span>
                <ul className="changed-list">
                  {record.lines.map((s) => <li key={s.key} className={`pe-line ${s.tone}`}>{s.text}</li>)}
                  <li className="muted">{record.score}</li>
                </ul>
                {record.changed && (
                  <p className="board-changed"><strong>What the board changed:</strong> {record.changed.join(" ")}</p>
                )}
              </div>
            )}
            {outcome && <OutcomeBadge outcome={outcome} />}
            {analysis && (
              <button className="link-button" type="button" onClick={() => navigate(`/advice/${analysis.id}`)}>
                Why this plan <ChevronRight size={14} />
              </button>
            )}
          </div>
        )}
      </div>
    </li>
  );
}

export default function History({ navigate }) {
  const { state } = useStore();
  const months = state.months;

  if (!months.length) {
    return (
      <section className="empty-state">
        <h2>No history yet</h2>
        <p className="narrow">Your months, decisions and outcomes will collect here after your first update.</p>
      </section>
    );
  }

  const newestFirst = [...months].reverse();

  return (
    <section className="content-stack">
      {months.length >= 3 && (
        <article className="panel history-charts">
          <MiniLine label="MRR" points={months.map((m) => m.values.mrr)} format={money} />
          <MiniLine label="Churn" points={months.map((m) => m.values.churnMonthly)} goodWhenDown format={(x) => pct(x)} />
          {/* null means not burning cash; the line shows it as the 60-month ceiling
              rather than dropping the month, and the label above says which. */}
          <MiniLine label="Cash lasts" points={months.map((m) => {
            const r = runwayMonths(m.values);
            return r === null ? 60 : Math.min(r, 60);
          })} format={(x) => monthsLabel(x)} />
        </article>
      )}
      <ol className="timeline">
        {newestFirst.map((m, i) => (
          <MonthEntry
            key={m.id}
            month={m}
            prev={newestFirst[i + 1] || null}
            next={newestFirst[i - 1] || null}
            analysis={analysisForMonth(state, m.id)}
            outcome={maturedOutcome(months, months.indexOf(m))}
            navigate={navigate}
          />
        ))}
      </ol>
    </section>
  );
}
