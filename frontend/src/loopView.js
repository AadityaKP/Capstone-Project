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
  return out;
}

// The one-line verdict on a scored prediction.
export function scoreLine(error) {
  const s = error?.summary;
  if (!s || !s.kpis_scored) return "Not scored yet.";
  return `${s.sign_agrees} of ${s.kpis_scored} predictions moved in the right direction; ${s.within_tolerance} landed within tolerance.`;
}

export function briefFreshness(source) {
  if (source === "llm") return { label: "fresh read", tone: "fresh" };
  if (source === "cache_hit" || source === "reuse") return { label: "reused", tone: "reused" };
  return { label: "rules only", tone: "off" };
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
