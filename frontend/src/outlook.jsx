// Outlook — where this plan takes you, on one chart (Phase C.5 of
// docs/ui_simplification_plan.md).
//
// One FanChart with a metric switch (Revenue / Cash / Customers lost) over the
// cycle's horizon. The opening point is the founder's own numbers; everything
// to its right is the model's plan simulated forward and is shaded to say so.
// A closed month 1 plots the founder's actual numbers as a marked point. The
// list of what the plan asks for in months 2 onward sits collapsed below.

import React, { useState } from "react";
import { ChevronRight } from "lucide-react";
import { money, monthOffsetLabel } from "./derive.js";
import { FanChart } from "./whatif.jsx";
import { Expandable } from "./components.jsx";
import { actionSummary, OUTLOOK_CAVEAT } from "./loopView.js";

const STYLE = { cycle: { color: "var(--purple)", band: "rgba(60, 52, 137, 0.16)" } };

export const OUTLOOK_METRICS = [
  { key: "mrr", label: "Revenue", format: money },
  { key: "cash", label: "Cash", format: money },
  { key: "churn_pct", label: "Customers lost", format: (v) => `1 in ${Math.round(100 / Math.max(v, 0.01))}` }
];

// Cash, unless revenue is falling — the one case where Revenue is the metric
// the founder is watching.
export function defaultOutlookMetric({ revenueFalling = false } = {}) {
  return revenueFalling ? "mrr" : "cash";
}

// Opening point plus one entry per horizon month; months that have not landed
// yet are null so the x-axis stays put while the line grows.
function seriesFor(key, opening, months, horizon) {
  const median = [opening[key]];
  const p25 = [opening[key]];
  const p75 = [opening[key]];
  for (let i = 0; i < horizon; i += 1) {
    const m = months[i];
    const after = m?.feedback?.state_after;
    const band = m?.feedback?.projection_band?.[key];
    median.push(after ? after[key] : null);
    p25.push(band ? band.p25 : after ? after[key] : null);
    p75.push(band ? band.p75 : after ? after[key] : null);
  }
  return { median, p25, p75 };
}

// `analysisId` + `navigate`: each "next months" row links straight to that
// month's strip under Why this plan → How the board got here, so months 2
// onward stay two interactions from This month.
export default function Outlook({ cycle, baseIso, closedMonth1 = null, initialMetric = "cash", analysisId = null, navigate = null }) {
  const [metric, setMetric] = useState(initialMetric);
  const months = cycle?.months || [];
  const opening = months[0]?.observe?.state_before || null;
  if (!cycle || !opening) return null;

  const horizon = cycle.horizon || cycle.summary?.horizon_months || months.length;
  const current = OUTLOOK_METRICS.find((m) => m.key === metric) || OUTLOOK_METRICS[1];
  const series = seriesFor(current.key, opening, months, horizon);
  const alive = [1, ...Array.from({ length: horizon }, (_, i) => (months[i]?.feedback?.state_after?.survived === false ? 0 : 1))];

  const actual = closedMonth1?.result?.actual_state || null;
  const markers = actual && actual[current.key] != null
    ? [{ month: 1, value: actual[current.key], label: "you" }]
    : null;

  return (
    <article className="panel outlook">
      <div className="panel-title-row">
        <h3>Where this plan takes you</h3>
        <div className="metric-switch" role="group" aria-label="Outlook metric">
          {OUTLOOK_METRICS.map((m) => (
            <button
              key={m.key} type="button"
              className={`metric-option ${m.key === current.key ? "on" : ""}`}
              onClick={() => setMetric(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
      <FanChart
        title={current.label} format={current.format}
        series={{ cycle: series }} alive={{ cycle: alive }}
        policies={["cycle"]} styles={STYLE}
        xStartLabel="now" xEndLabel={monthOffsetLabel(baseIso, horizon)}
        shadeFrom={actual ? 1 : 0} markers={markers}
      />
      <p className="wi-caveat">{OUTLOOK_CAVEAT}</p>

      {horizon > 1 && (
        <Expandable title={`Next ${horizon - 1} months of the plan`}>
          <ol className="next-months">
            {Array.from({ length: horizon - 1 }, (_, i) => {
              const m = months[i + 1];
              const linkable = m && analysisId && navigate;
              return (
                <li key={i + 1}>
                  <span className="next-month-label">{monthOffsetLabel(baseIso, i + 1)}</span>
                  {linkable ? (
                    <button className="link-button next-month-link" type="button" onClick={() => navigate(`/advice/${analysisId}/m${i + 2}`)}>
                      {actionSummary(m.execute?.action).join(" · ")} <ChevronRight size={13} />
                    </button>
                  ) : m
                    ? <span>{actionSummary(m.execute?.action).join(" · ")}</span>
                    : <span className="muted">deliberating…</span>}
                </li>
              );
            })}
          </ol>
        </Expandable>
      )}
    </article>
  );
}
