// Cycle vocabulary → founder vocabulary. Everything the Plan page, the OEFA
// strip, Home's "what changed" panel and the close-the-month step say about a
// prediction is assembled here from the server's numbers, so the sentence
// "we projected $47k, you did $44k — we were 6% high" has exactly one author.

import { money, signedPct, signedPp } from "./derive.js";

// Server keys → the four words the UI uses for the KPIs it holds the board to.
export const KPI_WORDS = {
  mrr_pct: "revenue",
  cash_pct: "cash",
  churn_pp: "churn",
  runway_months: "cash runway"
};

function fmtDelta(key, value) {
  if (value == null) return "—";
  if (key === "churn_pp") return signedPp(value);
  if (key === "runway_months") return `${value > 0 ? "+" : ""}${value.toFixed(1)} mo`;
  return signedPct(value);
}

// "revenue +4.0%, churn −0.3pp, cash −6.1% over 2 months"
export function expectedLine(delta) {
  if (!delta) return null;
  const parts = ["mrr_pct", "churn_pp", "cash_pct"]
    .filter((k) => delta[k] != null)
    .map((k) => `${KPI_WORDS[k]} ${fmtDelta(k, delta[k])}`);
  if (!parts.length) return null;
  return `${parts.join(", ")} over ${delta.horizon_months || 2} month${delta.horizon_months === 1 ? "" : "s"}`;
}

// The projected level implied by a percentage delta from a starting level.
export function projectedFrom(before, pct) {
  if (before == null || pct == null) return null;
  return before * (1 + pct / 100);
}

// One sentence per scored KPI: what was predicted, what happened, and how
// far off the board was — in the founder's units, never the engine's.
export function predictionSentences({ before, actual, expected, error, basis = "actual" }) {
  const out = [];
  if (!error) return out;
  const did = basis === "actual" ? "you did" : "the simulation did";

  const mrr = error.mrr_pct;
  if (mrr && before?.mrr != null && actual?.mrr != null) {
    const projected = projectedFrom(before.mrr, mrr.expected);
    const off = projected && actual.mrr ? ((projected - actual.mrr) / actual.mrr) * 100 : null;
    out.push({
      key: "mrr_pct",
      tone: mrr.within_tolerance ? "good" : mrr.sign_agrees ? "warn" : "bad",
      text: off == null
        ? `Revenue: the board expected ${fmtDelta("mrr_pct", mrr.expected)}, it moved ${fmtDelta("mrr_pct", mrr.realized)}.`
        : `We projected ${money(projected)} of revenue, ${did} ${money(actual.mrr)} — we were ${Math.abs(off).toFixed(0)}% ${off > 0 ? "high" : "low"}.`
    });
  }
  const churn = error.churn_pp;
  if (churn) {
    out.push({
      key: "churn_pp",
      tone: churn.within_tolerance ? "good" : churn.sign_agrees ? "warn" : "bad",
      text: churn.within_tolerance
        ? `Churn moved ${signedPp(churn.realized)}, about what the board expected (${signedPp(churn.expected)}).`
        : `Churn: the board expected ${signedPp(churn.expected)}, it moved ${signedPp(churn.realized)}${churn.sign_agrees ? "" : " — the wrong direction"}.`
    });
  }
  const cash = error.cash_pct;
  if (cash && before?.cash != null && actual?.cash != null) {
    out.push({
      key: "cash_pct",
      tone: cash.within_tolerance ? "good" : cash.sign_agrees ? "warn" : "bad",
      text: `Cash: expected ${money(projectedFrom(before.cash, cash.expected))}, ended at ${money(actual.cash)}.`
    });
  }
  // The server scores runway too when both sides were burning cash; without
  // a sentence for it the score line ("3 of 4") would not match the list.
  const runway = error.runway_months;
  if (runway) {
    out.push({
      key: "runway_months",
      tone: runway.within_tolerance ? "good" : runway.sign_agrees ? "warn" : "bad",
      text: `Cash lasts: the board expected ${fmtDelta("runway_months", runway.expected)}, it moved ${fmtDelta("runway_months", runway.realized)}.`
    });
  }
  return out;
}

// "What the board changed" after a close, read from the cycle that answered
// it (store.answeringCycle): nothing while it is still deliberating; its
// per-agent adaptation sentences once month 1 has landed; otherwise, when it
// started from the track record, the one sentence that is true (decision
// D4 — the LLM path returns no adaptation sentence).
export function boardChangedLines(answering, { tense = "this" } = {}) {
  const adapt = answering?.months?.[0]?.adapt;
  if (!adapt) return null;
  const adaptations = adapt.adaptations || [];
  if (adaptations.length) return adaptations.map((a) => `${a.agent}: ${a.sentence}`);
  if (answering.startedFromTrackRecord) {
    return [tense === "this"
      ? "The board planned this month with last month's result in hand."
      : "The board planned the next month with this result in hand."];
  }
  return null;
}

// The one-line verdict on a scored prediction.
export function scoreLine(error) {
  const s = error?.summary;
  if (!s || !s.kpis_scored) return "Not scored yet.";
  return `${s.sign_agrees} of ${s.kpis_scored} predictions moved in the right direction; ${s.within_tolerance} landed within tolerance.`;
}

// What the whole cycle did, from its summary block: the loop's own health,
// stated once in the trace section rather than on the plan.
export function loopLines(summary) {
  if (!summary) return [];
  return [
    `${summary.fresh_briefs} fresh strategist read${summary.fresh_briefs === 1 ? "" : "s"}, ${summary.reused_briefs} reused`,
    summary.graph_store_enabled
      ? "what happened each month was written back as simulated evidence"
      : "causal evidence graph off — nothing was written back",
    summary.memory_scope ? "memory scoped to your company" : "ran without memory",
    `${summary.total_latency_s?.toFixed?.(0) ?? summary.total_latency_s}s of deliberation`
  ];
}

export function briefFreshness(source) {
  if (source === "llm") return { label: "fresh read", tone: "fresh" };
  if (source === "cache_hit" || source === "reuse") return { label: "reused", tone: "reused" };
  return { label: "rules only", tone: "off" };
}

// The one caveat under the Outlook chart. The shaded region says "this is the
// model"; the band is the spread across simulated runs; dashed keeps meaning
// "some runs ran out of cash" (docs/ui_simplification_plan.md D2).
export const OUTLOOK_CAVEAT =
  "The shaded months are the board's plan simulated forward, not a forecast: the band is " +
  "the spread across simulated runs, and your real numbers replace the projection each " +
  "time you close a month.";

// First horizon month (1-based) in which the simulated company ran out of
// cash, or null when it survived every landed month.
export function cashDeathMonth(months) {
  const i = (months || []).findIndex((m) => m?.feedback?.state_after?.survived === false);
  return i < 0 ? null : i + 1;
}

// The horizon is named so the sentence cannot seem to contradict the
// twelve-month what-if projection on Why this plan.
export function cashDeathSentence(month, horizon) {
  return `In simulation this plan runs out of cash around month ${month} of its ${horizon}-month horizon.`;
}

// Which of the plan's four domains were actually actions this month.
export function actionSummary(action) {
  if (!action) return [];
  const lines = [];
  const mkt = action.marketing?.spend ?? 0;
  const rnd = action.product?.r_and_d_spend ?? 0;
  const hires = action.hiring?.hires ?? 0;
  const price = action.pricing?.price_change_pct ?? 0;
  lines.push(mkt > 0 ? `Marketing ≈${money(mkt)}` : "Marketing: hold");
  lines.push(rnd > 0 ? `Product ≈${money(rnd)}` : "Product: hold");
  lines.push(hires > 0 ? `Hire ${hires}` : "No hiring");
  lines.push(price > 0.001 ? `Price +${Math.round(price * 100)}%` : "Price: hold");
  return lines;
}

export const DONE_STATES = [
  { id: "did", label: "Did it" },
  { id: "partly", label: "Partly" },
  { id: "didnt", label: "Didn't" }
];

// did/partly/didn't → the decision states History already renders.
export const DONE_TO_DECISION = { did: "accepted", partly: "custom", didnt: "declined" };
