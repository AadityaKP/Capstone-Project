// Derived metrics and formatting rules.
// Formulas follow docs/founder_frontend_spec.md §5 (input architecture),
// §9 (KPI definitions) and §10.3 (rounding rules).

// runwayMonths and efficiencyBand moved to founderView.js. Both were
// translation rather than arithmetic, and both shipped a founder-facing
// defect: runway returned Infinity (printed as "∞" beside a projection that
// said the company died in month 1), and the efficiency band had no ceiling,
// so a $1 acquisition cost derived from ten customers read "Healthy".
import { runwayMonths } from "./founderView.js";
export { runwayMonths };

// Annual churn → monthly equivalent: 1 − (1 − annual)^(1/12).
export function annualToMonthlyChurn(annualPct) {
  const a = Math.min(Math.max(annualPct / 100, 0), 0.9999);
  return (1 - Math.pow(1 - a, 1 / 12)) * 100;
}

// CAC = marketing spend ÷ new customers (§5.1). null when underivable.
export function deriveCac({ cacDirect, marketingSpend, newCustomers }) {
  if (cacDirect > 0) return { value: cacDirect, source: "provided" };
  if (marketingSpend > 0 && newCustomers > 0) {
    return { value: marketingSpend / newCustomers, source: "derived" };
  }
  return { value: null, source: "estimated" };
}

// LTV = price ÷ monthly churn — the engine's own formula (compute_ltv). Never asked of founders.
export function deriveLtv({ price, churnMonthly }) {
  const churn = Math.max((churnMonthly || 0) / 100, 0.001);
  if (!price || price <= 0) return null;
  return price / churn;
}

// Market crowdedness → competitor count (§5.1): thresholds in the engine sit at 4/5/8/10.
export const CROWDEDNESS = [
  { id: "few", label: "Just us, mostly", competitors: 2 },
  { id: "some", label: "A few rivals", competitors: 5 },
  { id: "crowded", label: "Crowded", competitors: 9 }
];

// Product maturity self-rating → product_quality proxy (§5.1), always labelled estimated.
export const MATURITY = [
  { id: "early", label: "Early & rough", quality: 0.2 },
  { id: "solid", label: "Solid", quality: 0.5 },
  { id: "polished", label: "Polished", quality: 0.8 }
];

// virtualHeadcount lived here: non-marketing burn ÷ $8k salary slots, because the
// engine derived every burn figure from headcount. It is gone. Both of its floors
// — Math.max(…, 8000) and Math.max(1, …) — meant every company with monthly costs
// between $0 and $12,000 was charged exactly $8,000, so a founder spending $500
// was billed sixteen times over and died in month 0 of every projection. Costs now
// travel to the engine as themselves, in ScenarioConfig.monthly_costs.

// ---- formatting (§10.3) ----

// Currency: nearest $500 below $20k, nearest $1k above. Compact for display.
export function money(value) {
  if (value == null || Number.isNaN(value)) return "—";
  // §10.3 rounds currency to $500 below $20k / $1k above, but those bands are
  // written for aggregates (MRR, cash, spend). Per-unit figures — CAC, price —
  // sit under $500 and would collapse to "$0", so round those to the dollar.
  const abs = Math.abs(value);
  const rounded = abs < 1000
    ? Math.round(value)
    : abs < 20000
      ? Math.round(value / 500) * 500
      : Math.round(value / 1000) * 1000;
  if (Math.abs(rounded) >= 1_000_000) return `$${(rounded / 1_000_000).toFixed(2)}M`;
  if (Math.abs(rounded) >= 10_000) return `$${Math.round(rounded / 1000)}k`;
  if (Math.abs(rounded) >= 1_000) return `$${(rounded / 1000).toFixed(1)}k`;
  return `$${rounded.toLocaleString("en-US")}`;
}

export function moneyExact(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `$${Math.round(value).toLocaleString("en-US")}`;
}

export function pct(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

export function signedPct(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

export function signedPp(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}pp`;
}

// null now means "not burning cash" rather than Infinity, and is rendered by
// founderView.runwayLabel. This stays for plain month counts; it no longer has
// an infinity branch, because nothing can reach it.
export function monthsLabel(value) {
  if (value == null || Number.isNaN(value) || !Number.isFinite(value)) return "—";
  return `${Math.round(value)} mo`;
}

export function pctOfMrr(amount, mrr) {
  if (!mrr || mrr <= 0 || amount == null) return null;
  return (amount / mrr) * 100;
}

export function daysSince(iso) {
  if (!iso) return null;
  return Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
}

export function dateLabel(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function monthName(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", { month: "long", year: "numeric" });
}

// Deltas between two month snapshots, for KPI arrows and "what changed" (§9, §13).
export function monthDeltas(current, previous) {
  if (!current || !previous) return null;
  const mrrPct = previous.values.mrr > 0
    ? ((current.values.mrr - previous.values.mrr) / previous.values.mrr) * 100
    : null;
  // Either side can be null ("not burning cash"), and a delta between a number
  // and "not applicable" is not a number.
  const nowRunway = runwayMonths(current.values);
  const wasRunway = runwayMonths(previous.values);
  return {
    mrrPct,
    churnPp: current.values.churnMonthly - previous.values.churnMonthly,
    cash: current.values.cash - previous.values.cash,
    runway: nowRunway === null || wasRunway === null ? null : nowRunway - wasRunway
  };
}

// Client-side mirror of the engine's re-analysis event triggers (§13, thresholds from
// boardroom._has_event_trigger). Used only for display copy; the API's refresh_reason
// wins when an analysis is present.
export function eventTrigger(current, lastAnalyzed) {
  if (!current || !lastAnalyzed) return null;
  const rw = runwayMonths(current.values);
  if (rw !== null && rw < 12) return "runway";
  if (lastAnalyzed.values.mrr > 0 && current.values.mrr <= lastAnalyzed.values.mrr * 0.85) return "mrr_drop";
  if (current.values.churnMonthly - lastAnalyzed.values.churnMonthly >= 1.5) return "churn_jump";
  return null;
}
