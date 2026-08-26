// Engine vocabulary → founder vocabulary, client side.
//
// backend/founder_view.py is the authority: everything the server produces
// arrives already translated, in a `display` block, and components render that.
// This file exists for the one thing the server cannot translate — the UI is
// local-first, so Home, Company, History and Onboarding show runway and unit
// economics computed from numbers the browser holds and has never sent.
//
// The two implementations share their thresholds rather than each carrying a
// copy: config/founder_view.json is the single source, imported here and read by
// the Python module. If a rule needs a new number, it goes in that file.
//
// The rule this module enforces: anything with a name a founder would have to
// Google gets translated at the boundary. No component prints an engine field.

import RULES from "../../config/founder_view.json";

export const THRESHOLDS = RULES;

// ---- runway -------------------------------------------------------------

// Months of cash against net burn. null means "not burning cash" — never
// Infinity, which is a number no screen should print. Callers that used to
// receive Infinity got "∞" for free; they now have to say what they mean.
export function runwayMonths({ cash, costs, mrr }) {
  if (!cash || cash <= 0) return 0;
  const netBurn = (costs || 0) - (mrr || 0);
  if (netBurn <= 0) return null;
  return cash / netBurn;
}

export function runwayPhrase(values) {
  const months = runwayMonths(values);
  if (months === null) return "At current costs you're not burning cash";
  if (months <= 0) return "No cash left";
  if (months < 1) return "Less than a month of cash at current costs";
  if (months >= 60) return "Over 5 years of cash at current costs";
  return `${Math.round(months)} month${Math.round(months) === 1 ? "" : "s"} of cash at current costs`;
}

// Compact form for KPI tiles and table cells, where the sentence will not fit.
export function runwayLabel(values) {
  const months = runwayMonths(values);
  if (months === null) return "Not burning";
  if (months <= 0) return "None";
  if (months >= 60) return "5 yr+";
  return `${Math.round(months)} mo`;
}

// ---- churn --------------------------------------------------------------

// A count, not a percentage. Kept deliberately distinct from the simulator's
// churn, which is decayed by customer tenure and moved by product quality — a
// different quantity that happens to share a name.
export function churnPhrase(monthlyPct) {
  if (monthlyPct == null || monthlyPct <= 0) return "You're not losing customers";
  if (monthlyPct >= 50) return "You lose more than half your customers every month";
  if (monthlyPct < 1) return "You lose fewer than 1 in 100 customers a month";
  return `You lose about 1 in ${Math.round(100 / monthlyPct)} customers a month`;
}

export function churnLabel(monthlyPct) {
  if (monthlyPct == null || monthlyPct <= 0) return "None";
  if (monthlyPct >= 50) return "over half";
  if (monthlyPct < 1) return "<1 in 100";
  return `1 in ${Math.round(100 / monthlyPct)}`;
}

// ---- unit economics -----------------------------------------------------

function money(value) {
  if (value == null || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 10_000) return `${sign}$${Math.round(abs / 1000)}k`;
  return `${sign}$${Math.round(abs).toLocaleString("en-US")}`;
}

// LTV:CAC as a judgement, or an honest refusal. The old band said "Healthy" for
// any ratio at or above 3 with no ceiling, so a founder who spent $10 and signed
// 10 customers saw "Healthy" against a 100× ratio built from ten data points.
export function efficiency(ltv, cac, newCustomers) {
  if (!ltv || !cac || cac <= 0) {
    return {
      band: "unknown", label: "Not enough data yet", ratio: null,
      detail: "Add last month's marketing spend and new customers."
    };
  }
  if (newCustomers != null && newCustomers > 0 && newCustomers < RULES.cac_min_customers) {
    return {
      band: "unknown", label: "Not enough data yet", ratio: null,
      detail: `${Math.round(newCustomers)} new customers is too few to price acquisition from.`
    };
  }
  const ratio = ltv / cac;
  const detail = `Each customer costs ${money(cac)} to win and pays back ${money(ltv)}.`;
  // The ceiling catches tiny denominators, not large ratios as such. When the
  // customer count is visible and big enough to trust, the reason for refusing
  // is gone — see backend/founder_view.py for the measured case that showed it.
  const verifiedSample = newCustomers != null && newCustomers >= RULES.cac_min_customers;
  if (ratio > RULES.ltv_cac_unmeasurable && !verifiedSample) {
    return {
      band: "unknown", label: "Can't measure this yet", ratio,
      detail: `${detail} A return that large usually means the acquisition cost came from too few customers to trust.`
    };
  }
  if (ratio >= RULES.ltv_cac_healthy) {
    // Lifetime value is price ÷ churn, so a long payback is a claim about churn
    // holding rather than something observed.
    const caveat = ratio > RULES.ltv_cac_unmeasurable
      ? " That payback assumes your churn stays where it is." : "";
    return { band: "healthy", label: "Healthy", ratio, detail: detail + caveat };
  }
  if (ratio >= RULES.ltv_cac_watch) return { band: "watch", label: "Worth watching", ratio, detail };
  return { band: "unhealthy", label: "Costs more than it returns", ratio, detail };
}

// ---- money in, money out ------------------------------------------------

export function showRuleOf40(mrr) {
  return Boolean(mrr) && mrr >= RULES.rule_of_40_mrr_floor;
}

export function spendRatioLabel({ costs, mrr }) {
  if (!mrr || mrr <= 0) return null;
  return `$${((costs || 0) / mrr).toFixed(2)} per $1`;
}

export function spendRatioPhrase({ costs, mrr }) {
  if (!mrr || mrr <= 0) return `You're spending ${money(costs)} a month with no revenue yet`;
  return `You spend $${((costs || 0) / mrr).toFixed(2)} for every $1 of revenue`;
}

// ---- confidence ---------------------------------------------------------

// The server sends the sentence; this is the fallback for analyses stored before
// display blocks existed. Same caps, same wording.
export function confidenceSentence(modelConfidence, assumedCount) {
  const order = ["Low", "Moderate", "High"];
  let band = "Moderate";
  if (modelConfidence != null) {
    band = modelConfidence < 0.4 ? "Low" : modelConfidence <= 0.7 ? "Moderate" : "High";
  }
  let cap = "Low";
  for (const rule of RULES.confidence_caps) {
    if (assumedCount <= rule.max_assumed) { cap = rule.cap; break; }
  }
  const capped = order[Math.min(order.indexOf(band), order.indexOf(cap))];
  if (assumedCount <= 0) return `${capped} confidence, from the numbers you gave us`;
  return `${capped} confidence — ${assumedCount} of these numbers ${
    assumedCount === 1 ? "is an estimate" : "are estimates"
  }, not yours`;
}
