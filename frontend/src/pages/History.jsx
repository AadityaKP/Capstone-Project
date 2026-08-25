// S10 History — one vertical monthly timeline (spec §14): numbers → advice →
// decisions → matured outcomes. Mini-charts appear once 3 snapshots exist.

import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useStore, analysisForMonth } from "../store.jsx";
import {
  money, pct, signedPct, signedPp, monthName, runwayMonths, monthDeltas, monthsLabel
} from "../derive.js";
import { RiskChip, MiniLine, OutcomeBadge } from "../components.jsx";

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

function MonthEntry({ month, prev, analysis, outcome, navigate }) {
  const [open, setOpen] = useState(false);
  const v = month.values;
  const deltas = monthDeltas(month, prev);
  const decisions = month.decisions || [];
  const accepted = decisions.filter((d) => d.state === "accepted").length;

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
          {" · "}runway {monthsLabel(runwayMonths(v))}
        </p>
        {analysis?.brief?.recommended_focus?.length > 0 && (
          <p className="timeline-plan">Plan: {analysis.brief.recommended_focus.join(" · ").toLowerCase()}</p>
        )}
        {decisions.length > 0 && (
          <p className="timeline-decisions">
            You accepted {accepted} of {decisions.length} action{decisions.length === 1 ? "" : "s"}
            {decisions.some((d) => d.state === "custom") && " · adjusted one yourself"}
          </p>
        )}
        {outcome && <OutcomeBadge outcome={outcome} />}
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
            {analysis && (
              <button className="link-button" type="button" onClick={() => navigate(`/advice/${analysis.id}`)}>
                Open full advice <ChevronRight size={14} />
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
          <MiniLine label="Runway" points={months.map((m) => Math.min(runwayMonths(m.values), 60))} format={(x) => monthsLabel(x)} />
        </article>
      )}
      <ol className="timeline">
        {newestFirst.map((m, i) => (
          <MonthEntry
            key={m.id}
            month={m}
            prev={newestFirst[i + 1] || null}
            analysis={analysisForMonth(state, m.id)}
            outcome={maturedOutcome(months, months.indexOf(m))}
            navigate={navigate}
          />
        ))}
      </ol>
    </section>
  );
}
