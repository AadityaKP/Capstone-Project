// Sample company for the "Explore a sample company" preview (Welcome screen).
// Data is illustrative and clearly labelled in the UI (spec §15 honesty rules);
// shapes mirror the real engine exactly: OracleBrief fields, the boardroom
// decision_trace keys from boardroom/boardroom.py, and the cycle month shape
// from backend/cycle_service.py (observe / execute / feedback / adapt).
//
// The sample opens in the middle of the story on purpose (plan section 8.1):
// one cycle already closed with the founder's real numbers - including an
// action marked "didn't" - and a second cycle planned from those numbers.
// A first cycle has no memory, no prediction error and nothing learned.

function iso(monthsAgo, day = 1) {
  const d = new Date();
  d.setMonth(d.getMonth() - monthsAgo, day);
  d.setHours(9, 30, 0, 0);
  return d.toISOString();
}

const memoryDecline = {
  document:
    "Phase: SEED | Churn: HIGH | Innovation: DECLINING\n" +
    "Episode month 14: MRR 41,250, avg churn 0.052, innovation 0.61. " +
    "Trends were MRR FLAT, innovation DECREASING, churn INCREASING. " +
    "After 6 months the realized outcome was DECLINE.",
  metadata: { realized_outcome: "DECLINE", source_month: 14, stored_global_month: 20 },
  distance: 0.42, similarity_score: 0.7, recency_factor: 0.81, memory_weight: 0.65
};

const memoryFlat = {
  document:
    "Phase: SEED | Churn: MEDIUM | Innovation: HEALTHY\n" +
    "Episode month 22: MRR 58,900, avg churn 0.041, innovation 0.86. " +
    "Trends were MRR INCREASING, innovation FLAT, churn DECREASING. " +
    "After 6 months the realized outcome was STAGNATION.",
  metadata: { realized_outcome: "STAGNATION", source_month: 22, stored_global_month: 28 },
  distance: 0.55, similarity_score: 0.64, recency_factor: 0.74, memory_weight: 0.47
};

// ---- mirrors of boardroom/expectation.py, so the sample scores itself the
// same way the engine does (kpi_delta_between, prediction_error) ----

const TOLERANCE = { mrr_pct: 1.0, cash_pct: 2.0, churn_pp: 0.2, runway_months: 0.5 };
const r3 = (x) => Math.round(x * 1000) / 1000;

function runwayOf(s) {
  if (!s.cash || s.cash <= 0) return 0;
  const net = s.costs - s.mrr;
  return net <= 0 ? null : s.cash / net;
}

function kpis(s) {
  return { mrr: s.mrr, cash: s.cash, churn_pct: s.churn, runway_months: runwayOf(s) == null ? null : r3(runwayOf(s)) };
}

function deltaBetween(a, b) {
  const ra = runwayOf(a), rb = runwayOf(b);
  return {
    mrr_pct: r3(((b.mrr - a.mrr) / Math.max(Math.abs(a.mrr), 1)) * 100),
    cash_pct: r3(((b.cash - a.cash) / Math.max(Math.abs(a.cash), 1)) * 100),
    churn_pp: r3(b.churn - a.churn),
    runway_months: ra == null || rb == null ? null : r3(rb - ra)
  };
}

function score(expected, realized) {
  const out = {};
  const scored = [];
  for (const key of Object.keys(TOLERANCE)) {
    const exp = expected[key], got = realized[key];
    if (exp == null || got == null) { out[key] = null; continue; }
    const err = r3(got - exp);
    const item = {
      expected: r3(exp), realized: r3(got), error: err,
      within_tolerance: Math.abs(err) <= TOLERANCE[key],
      sign_agrees: (exp === 0 && Math.abs(got) <= TOLERANCE[key]) || (exp > 0 && got > -TOLERANCE[key]) || (exp < 0 && got < TOLERANCE[key])
    };
    out[key] = item;
    scored.push(item);
  }
  out.summary = {
    kpis_scored: scored.length,
    within_tolerance: scored.filter((s) => s.within_tolerance).length,
    sign_agrees: scored.filter((s) => s.sign_agrees).length,
    horizon_months: expected.horizon_months
  };
  return out;
}

function band(v, spread) {
  return { p25: Math.round(v * (1 - spread)), median: Math.round(v), p75: Math.round(v * (1 + spread)) };
}

function expectation(after, { mrr_pct, cash_pct, churn_pp, runway_months }) {
  return {
    mrr_pct, cash_pct, churn_pp, runway_months, horizon_months: 2, n_seeds: 8, survival: 1,
    basis: "simulated",
    band: { mrr: band(after.mrr * 1.04, 0.05), cash: band(after.cash * 0.97, 0.03), churn_pct: band(after.churn, 0.06) }
  };
}

function trackRecord(action, expected, realized, month, source) {
  return {
    month, source, action,
    expected_delta: { mrr_pct: expected.mrr_pct, cash_pct: expected.cash_pct, churn_pp: expected.churn_pp, runway_months: expected.runway_months, horizon_months: 2 },
    realized_delta: realized,
    prediction_error: score(expected, realized)
  };
}

const ASSUMED = [
  { field: "Unemployment", value: "4.0%", why: "not asked at onboarding; typical conditions", correctable: false },
  { field: "Valuation multiple", value: "10.0x ARR", why: "not asked at onboarding; engine default", correctable: false }
];

function proposalsFor(action, expected, adaptation = null) {
  const base = { risks: [], rationale: null, causal_confidence: null, base_score: 0.61, final_confidence: 0.74,
                 score_vector: { efficiency: 0.4, growth: 0.6, innovation: 0.7, macro: 0.87 } };
  return [
    { agent: "CFO", objective: "Preserve runway and improve efficiency", actions: { hiring: action.hiring, pricing: action.pricing },
      expected_impact: "Lower burn, improved survival probability", expected_delta: { ...expected, mrr_pct: 0.4, cash_pct: expected.cash_pct + 3.5 }, adaptation: null, ...base },
    { agent: "CMO", objective: "Maximize growth under CAC constraints", actions: { marketing: action.marketing },
      expected_impact: "Increased MRR growth", expected_delta: { ...expected, churn_pp: 0 }, adaptation: null, ...base },
    { agent: "CPO", objective: "Reduce churn and improve retention", actions: { product: action.product },
      expected_impact: "Higher NRR and lower churn", expected_delta: { ...expected, mrr_pct: 1.2 }, adaptation, ...base }
  ];
}

// One cycle month in the server's shape.
function cycleMonth({ index, before, after, action, brief, expected, memories, briefSource, refreshReason,
                      stress, weights, pre, whatChanged, adaptation = null, track = null, evidenceWritten = true }) {
  const kpiDelta = deltaBetween(before, after);
  const error = score(expected, kpiDelta);
  const proposals = proposalsFor(action, expected, adaptation);
  const trace = {
    month: 8 + index, oracle_mode: "oracle_v4_causal", used_oracle: true,
    refresh_reason: refreshReason, brief_source: briefSource, cache_key: null, shock_label: null,
    base_weights: { efficiency: 0.3, growth: 0.2, innovation: 0.4, macro: 0.1 },
    applied_weights: weights, brief,
    memory_count: memories.length, retrieved_memories: memories,
    proposal_source: "llm", proposal_error: null,
    causal_stress_node: stress, stress_persistence_months: 1, previous_final_action: null,
    causal_contexts: {}, proposals,
    pre_modifier_action: pre, post_modifier_action: action, final_action: action,
    expected_delta: expected, track_record: track,
    action_modifier_applied: true, brief_floor_applied: [],
    marketing_spend_change_pct: ((action.marketing.spend - pre.marketing.spend) / Math.max(pre.marketing.spend, 1)) * 100,
    rd_spend_change_pct: ((action.product.r_and_d_spend - pre.product.r_and_d_spend) / Math.max(pre.product.r_and_d_spend, 1)) * 100,
    hires_change: action.hiring.hires - pre.hiring.hires,
    spend_ceiling: null, assumed_fields: ASSUMED, history_months_replayed: index === 0 ? 2 : 0, absolute_scale: 0.6,
    graph_summary: { stress_node: stress, observed: [], expected: ["Tech_Debt_Remediation", "Acquisition_Channel_Reallocation"], confidence: 0.67, roles: 3 },
    memory_scope: "company:sample-acme", graph_store_enabled: true,
    display: {
      confidence: { band: "Moderate", sentence: "Moderate confidence — 2 of these numbers are estimates, not yours" },
      runway: `${Math.round(runwayOf(before))} months of cash at current costs`,
      spend_ratio: `You spend $${((before.costs + action.marketing.spend + action.product.r_and_d_spend) / before.mrr).toFixed(2)} for every $1 of revenue`,
      show_rule_of_40: false, monthly_burn: before.costs, monthly_burn_supplied: true
    }
  };
  return {
    month_index: index + 1,
    projection: index > 0,
    observe: {
      state_before: kpis(before),
      memory_count: memories.length, memories,
      trend: { mrr_trend: "INCREASING", innovation_trend: "FLAT", churn_trend: "DECREASING", history_points: 3 },
      graph: { stress_node: stress, contexts: {}, summary: trace.graph_summary, enabled: true },
      memory_scope: "company:sample-acme", pending_memories: 3 + index, matured_memories: 0
    },
    execute: {
      action, brief, llm_ok: true, brief_source: briefSource, refresh_reason: refreshReason,
      proposal_source: "llm", proposals, weights, base_weights: trace.base_weights,
      expected_delta: expected, spend_ceiling: null, display: trace.display, trace
    },
    feedback: {
      state_after: { ...kpis(after), survived: true },
      kpi_delta: kpiDelta, prediction_error: error,
      projection_band: { mrr: band(after.mrr, 0.04), cash: band(after.cash, 0.025), churn_pct: band(after.churn, 0.05) },
      evidence_written: evidenceWritten, evidence_source: "sim", shock_label: "NO_SHOCK", basis: "simulated"
    },
    adapt: {
      what_changed: whatChanged, refresh_reason: refreshReason, brief_source: briefSource,
      weight_moves: [], adaptations: adaptation ? [{ agent: "CPO", sentence: adaptation }] : [],
      track_record_for_next_month: trackRecord(action, expected, kpiDelta, `month ${index + 1}`, "simulated"),
      memory: { pending: 3 + index, matured_this_month: 0 }
    },
    latency_s: [41.2, 3.1, 38.7, 2.9][index] || 3
  };
}

const juneBrief = {
  risk_level: "HIGH", growth_outlook: "DECLINING", efficiency_pressure: "MEDIUM",
  innovation_urgency: "HIGH", macro_condition: "NEUTRAL", expected_outcome: "DECLINE",
  key_risks: ["Churn is eating most new revenue", "Runway shrinks fast at current burn"],
  key_opportunities: ["Retention work has room to compound"],
  recommended_focus: ["Protect retention", "Keep acquisition lean"],
  confidence: 0.58
};

const julyBrief = {
  risk_level: "HIGH", growth_outlook: "STABLE", efficiency_pressure: "MEDIUM",
  innovation_urgency: "HIGH", macro_condition: "NEUTRAL", expected_outcome: "STAGNATION",
  key_risks: ["Churn is still high for this stage", "Runway is under 12 months"],
  key_opportunities: ["Acquisition cost is sustainable", "Early retention gains showing"],
  recommended_focus: ["Protect retention", "Hold pricing steady"],
  confidence: 0.62
};

const augustBrief = {
  risk_level: "MEDIUM", growth_outlook: "STABLE", efficiency_pressure: "MEDIUM",
  innovation_urgency: "HIGH", macro_condition: "NEUTRAL", expected_outcome: "STAGNATION",
  key_risks: ["Churn improving but still above healthy range", "Runway near 12 months"],
  key_opportunities: ["Retention gains are compounding", "Room to grow spend if churn holds"],
  recommended_focus: ["Keep retention the priority", "Modest acquisition restart"],
  confidence: 0.66
};

const act = (mkt, rnd, hires = 0, price = 0) => ({
  marketing: { spend: mkt, channel: "ppc" }, hiring: { hires, cost_per_employee: 10000 },
  product: { r_and_d_spend: rnd }, pricing: { price_change_pct: price }
});
const PRE = act(10000, 8000, 1, 0);
const W1 = { efficiency: 0.29, growth: 0.17, innovation: 0.44, macro: 0.1 };
const W2 = { efficiency: 0.28, growth: 0.19, innovation: 0.43, macro: 0.1 };

// ---- cycle 1: planned on the July numbers, closed with the August ones ----

const july = { mrr: 30000, cash: 360000, costs: 31000, churn: 5.2 };
const c1m1 = { mrr: 31600, cash: 350500, costs: 31000, churn: 5.0 };
const c1m2 = { mrr: 32900, cash: 341000, costs: 31000, churn: 4.8 };
const c1m3 = { mrr: 34100, cash: 331500, costs: 31000, churn: 4.7 };
const c1m4 = { mrr: 35200, cash: 322000, costs: 31000, churn: 4.6 };
const c1a1 = act(5000, 9500, 0, 0.03);
const c1e1 = expectation(c1m1, { mrr_pct: 12.5, cash_pct: -4.5, churn_pp: -0.5, runway_months: -0.9 });

const cycle1Months = [
  cycleMonth({ index: 0, before: july, after: c1m1, action: c1a1, brief: julyBrief, expected: c1e1,
    memories: [memoryDecline, memoryFlat], briefSource: "llm", refreshReason: "initial", stress: "Churn_Spike",
    weights: W1, pre: PRE, whatChanged: ["brief refreshed by the strategist (initial)", "prediction scored: 3 of 4 KPIs moved in the predicted direction"] }),
  cycleMonth({ index: 1, before: c1m1, after: c1m2, action: act(5000, 9500), brief: julyBrief,
    expected: expectation(c1m2, { mrr_pct: 8.0, cash_pct: -4.2, churn_pp: -0.3, runway_months: -0.7 }),
    memories: [memoryDecline, memoryFlat], briefSource: "reuse", refreshReason: null, stress: "Churn_Spike",
    weights: W1, pre: PRE, whatChanged: ["brief reused - no event moved the numbers enough to re-read them", "prediction scored: 4 of 4 KPIs moved in the predicted direction"] }),
  cycleMonth({ index: 2, before: c1m2, after: c1m3, action: act(4500, 10000), brief: julyBrief,
    expected: expectation(c1m3, { mrr_pct: 7.0, cash_pct: -4.4, churn_pp: -0.3, runway_months: -0.6 }),
    memories: [memoryFlat], briefSource: "llm", refreshReason: "event", stress: "Churn_Spike",
    weights: W2, pre: PRE, whatChanged: ["brief refreshed by the strategist (event)", "growth weight up 0.020 to 0.190", "prediction scored: 4 of 4 KPIs moved in the predicted direction"] }),
  cycleMonth({ index: 3, before: c1m3, after: c1m4, action: act(4500, 10000), brief: julyBrief,
    expected: expectation(c1m4, { mrr_pct: 6.5, cash_pct: -4.6, churn_pp: -0.2, runway_months: -0.6 }),
    memories: [memoryFlat], briefSource: "reuse", refreshReason: null, stress: "Churn_Spike",
    weights: W2, pre: PRE, whatChanged: ["brief reused - no event moved the numbers enough to re-read them", "prediction scored: 4 of 4 KPIs moved in the predicted direction"] })
];

// The close: what the founder actually did and what the numbers actually were.
const august = { mrr: 31900, cash: 348000, costs: 30000, churn: 4.6 };
const c1ActualDelta = deltaBetween(july, august);
const c1Track = trackRecord(c1a1, c1e1, c1ActualDelta, "month 1", "observed");
const cycle1Feedback = {
  monthIndex: 1,
  closedAt: iso(0, 1),
  submitted: {
    per_action: [
      { action_key: "marketing", done: "did", note: null },
      { action_key: "product", done: "partly", note: "Shipped the onboarding fix, not the reporting work" },
      { action_key: "pricing", done: "didnt", note: "Didn't feel right to raise mid-quarter" }
    ],
    actuals: { mrr: 31900, cash: 348000, churn: 4.6, costs: 30000 }
  },
  result: {
    cycle_id: "c1", month_index: 1,
    expected_delta: c1e1, actual_delta: c1ActualDelta, prediction_error: c1Track.prediction_error,
    actual_state: kpis(august),
    evidence: {
      written: true, edges: 4, source: "observed", weight: 0.5,
      credited: ["marketing", "product"], skipped: ["hiring", "pricing"],
      reason: "4 edge(s) strengthened or weakened for the actions taken (half weight: partly done)"
    },
    per_action: [
      { action_key: "marketing", done: "did", counted: true, note: null, why: "counted as evidence" },
      { action_key: "product", done: "partly", counted: true, note: "Shipped the onboarding fix, not the reporting work", why: "counted at half weight" },
      { action_key: "hiring", done: null, counted: false, note: null, why: "not part of last month's plan" },
      { action_key: "pricing", done: "didnt", counted: false, note: "Didn't feel right to raise mid-quarter", why: "you didn't do this, so this month is not counted as evidence about it" }
    ],
    memory: { observed: true, pending: 4, matured: 0, scope: "company:sample-acme" },
    track_record: c1Track,
    graph_store_enabled: true,
    closed_at: iso(0, 1)
  }
};

// ---- cycle 2: planned on the August numbers, starting from the close ----

const c2m1 = { mrr: 33400, cash: 339500, costs: 30000, churn: 4.5 };
const c2m2 = { mrr: 34700, cash: 330500, costs: 30000, churn: 4.7 };
const c2m3 = { mrr: 35600, cash: 322500, costs: 30000, churn: 4.6 };
const c2m4 = { mrr: 36500, cash: 314000, costs: 30000, churn: 4.5 };
const c2a1 = act(6500, 9000);
const c2e1 = expectation(c2m1, { mrr_pct: 9.5, cash_pct: -3.5, churn_pp: -0.3, runway_months: -0.6 });

const cycle2Months = [
  cycleMonth({ index: 0, before: august, after: c2m1, action: c2a1, brief: augustBrief, expected: c2e1,
    memories: [memoryFlat, memoryDecline], briefSource: "llm", refreshReason: "initial", stress: "Churn_Spike",
    weights: W2, pre: PRE, track: c1Track,
    whatChanged: ["brief refreshed by the strategist (initial)", "started from your real August numbers; last month's plan ran 6% high on revenue", "prediction scored: 4 of 4 KPIs moved in the predicted direction"] }),
  cycleMonth({ index: 1, before: c2m1, after: c2m2, action: act(6500, 9000), brief: augustBrief,
    expected: expectation(c2m2, { mrr_pct: 7.5, cash_pct: -3.8, churn_pp: -0.3, runway_months: -0.5 }),
    memories: [memoryFlat], briefSource: "reuse", refreshReason: null, stress: "Churn_Spike",
    weights: W2, pre: PRE, whatChanged: ["brief reused - no event moved the numbers enough to re-read them", "prediction scored: 3 of 4 KPIs moved in the predicted direction"] }),
  cycleMonth({ index: 2, before: c2m2, after: c2m3, action: act(6500, 6750), brief: augustBrief,
    expected: expectation(c2m3, { mrr_pct: 6.0, cash_pct: -4.0, churn_pp: -0.1, runway_months: -0.5 }),
    memories: [memoryFlat], briefSource: "llm", refreshReason: "event", stress: "Churn_Spike",
    weights: W2, pre: PRE,
    adaptation: "Last month's plan expected churn -0.3pp and saw +0.2pp; product spend is held back 25% this month.",
    whatChanged: ["brief refreshed by the strategist (event)", "prediction scored: 4 of 4 KPIs moved in the predicted direction"] }),
  cycleMonth({ index: 3, before: c2m3, after: c2m4, action: act(6500, 6750), brief: augustBrief,
    expected: expectation(c2m4, { mrr_pct: 5.5, cash_pct: -4.2, churn_pp: -0.1, runway_months: -0.4 }),
    memories: [memoryFlat], briefSource: "reuse", refreshReason: null, stress: "Steady_State",
    weights: W2, pre: PRE, whatChanged: ["brief reused - no event moved the numbers enough to re-read them", "prediction scored: 4 of 4 KPIs moved in the predicted direction"] })
];

function cycleSummary(months, fresh, reused) {
  const last = months[months.length - 1];
  return {
    months_completed: months.length, horizon_months: 4, survived: true,
    projected_state: last.feedback.state_after, runway_at_horizon: last.feedback.state_after.runway_months,
    fresh_briefs: fresh, reused_briefs: reused, llm_ok_months: 4, llm_calls: fresh, proposal_llm_calls: fresh,
    evidence_written_months: 4, total_latency_s: months.reduce((s, m) => s + m.latency_s, 0),
    graph_store_enabled: true, memory_scope: "company:sample-acme",
    prediction: { kpis_scored: 16, sign_agrees: 15, within_tolerance: 11 }
  };
}

const META = {
  oracle_mode: "oracle_v4_causal", use_oracle: true, seed: 0, history_months_replayed: 2,
  assumed_fields: ASSUMED, absolute_scale: 0.6, sim_profile: "founder", expected_delta_basis: "simulated"
};

// ---- the traces the single-month Advice detail reads come from month 1 ----

function makeTrace({ month, refreshReason, weights, pre, post, final: finalAction, brief, memories }) {
  return {
    month, oracle_mode: "oracle_v3", used_oracle: true, refresh_reason: refreshReason,
    brief_source: "llm", cache_key: null, shock_label: null,
    base_weights: { efficiency: 0.3, growth: 0.2, innovation: 0.4, macro: 0.1 },
    applied_weights: weights, brief, memory_count: memories.length, retrieved_memories: memories,
    pre_modifier_action: pre, post_modifier_action: post, final_action: finalAction,
    action_modifier_applied: true,
    marketing_spend_change_pct: pre.marketing.spend
      ? ((post.marketing.spend - pre.marketing.spend) / Math.max(pre.marketing.spend, 1)) * 100 : 0,
    rd_spend_change_pct: pre.product.r_and_d_spend
      ? ((post.product.r_and_d_spend - pre.product.r_and_d_spend) / Math.max(pre.product.r_and_d_spend, 1)) * 100 : 0,
    hires_change: post.hiring.hires - pre.hiring.hires
  };
}

export const SAMPLE = {
  demo: true,
  company: {
    id: "sample-acme",
    name: "Acme Analytics",
    whatYouSell: "Usage analytics for e-commerce teams",
    ageMonths: 8,
    crowdedness: "crowded",
    maturity: "solid",
    headcountReal: 4,
    createdAt: iso(2)
  },
  months: [
    {
      id: "m1", index: 0, enteredAt: iso(2),
      values: { mrr: 28800, cash: 378000, costs: 30000, price: 85, churnMonthly: 5.6, newCustomers: 38, marketingSpend: 6000 },
      decisions: [
        { id: "d1", domain: "product", text: "Allocate ≈$9k to product work", state: "accepted" },
        { id: "d2", domain: "marketing", text: "Cut marketing to ≈$3k", state: "custom", note: "Did $4k instead" },
        { id: "d3", domain: "pricing", text: "Hold pricing", state: "accepted" }
      ]
    },
    {
      id: "m2", index: 1, enteredAt: iso(1),
      values: { mrr: 30000, cash: 360000, costs: 31000, price: 85, churnMonthly: 5.2, newCustomers: 41, marketingSpend: 6000 },
      decisions: [
        { id: "d4", domain: "marketing", text: "Spend ≈$5,000 on performance channels", state: "accepted" },
        { id: "d5", domain: "product", text: "Invest ≈$9,500 in product this month", state: "custom", note: "Shipped the onboarding fix, not the reporting work" },
        { id: "d6", domain: "pricing", text: "Consider a ≈3% price increase", state: "declined", note: "Didn't feel right to raise mid-quarter" }
      ]
    },
    {
      id: "m3", index: 2, enteredAt: iso(0, 1),
      values: { mrr: 31900, cash: 348000, costs: 30000, price: 85, churnMonthly: 4.6, newCustomers: 44, marketingSpend: 4000 },
      decisions: []
    }
  ],
  cycles: [
    {
      id: "c1", monthId: "m2", createdAt: iso(1), source: "sample", status: "completed", horizon: 4,
      months: cycle1Months, summary: cycleSummary(cycle1Months, 2, 2), meta: META,
      feedback: [cycle1Feedback], startedFromTrackRecord: false
    },
    {
      id: "c2", monthId: "m3", createdAt: iso(0, 1), source: "sample", status: "completed", horizon: 4,
      months: cycle2Months, summary: cycleSummary(cycle2Months, 2, 2), meta: META,
      feedback: [], startedFromTrackRecord: true
    }
  ],
  analyses: [
    {
      id: "a1", monthId: "m1", createdAt: iso(2), source: "sample", llm_ok: true,
      brief: juneBrief,
      trace: makeTrace({
        month: 8, refreshReason: "initial",
        weights: { efficiency: 0.28, growth: 0.16, innovation: 0.46, macro: 0.1 },
        pre: PRE,
        post: act(5000, 9000),
        final: act(3000, 9000),
        brief: juneBrief, memories: [memoryDecline]
      })
    },
    {
      id: "a2", monthId: "m2", cycleId: "c1", monthIndex: 1, createdAt: iso(1), source: "sample", llm_ok: true,
      brief: julyBrief, trace: cycle1Months[0].execute.trace, display: cycle1Months[0].execute.display
    },
    {
      id: "a3", monthId: "m3", cycleId: "c2", monthIndex: 1, createdAt: iso(0, 1), source: "sample", llm_ok: true,
      brief: augustBrief, trace: cycle2Months[0].execute.trace, display: cycle2Months[0].execute.display
    }
  ],
  settings: { narratives: false }
};
