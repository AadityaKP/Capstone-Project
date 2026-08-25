// Sample company for the "Explore a sample company" preview (Welcome screen).
// Data is illustrative and clearly labelled in the UI (spec §15 honesty rules);
// shapes mirror the real engine exactly: OracleBrief fields and the boardroom
// decision_trace keys from boardroom/boardroom.py.

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

function makeTrace({ month, refreshReason, weights, pre, post, final: finalAction, brief, memories }) {
  return {
    month,
    oracle_mode: "oracle_v3",
    used_oracle: true,
    refresh_reason: refreshReason,
    brief_source: "llm",
    cache_key: null,
    shock_label: null,
    base_weights: { efficiency: 0.3, growth: 0.2, innovation: 0.4, macro: 0.1 },
    applied_weights: weights,
    brief,
    memory_count: memories.length,
    retrieved_memories: memories,
    pre_modifier_action: pre,
    post_modifier_action: post,
    final_action: finalAction,
    action_modifier_applied: true,
    marketing_spend_change_pct: pre.marketing.spend
      ? ((post.marketing.spend - pre.marketing.spend) / Math.max(pre.marketing.spend, 1)) * 100
      : 0,
    rd_spend_change_pct: pre.product.r_and_d_spend
      ? ((post.product.r_and_d_spend - pre.product.r_and_d_spend) / Math.max(pre.product.r_and_d_spend, 1)) * 100
      : 0,
    hires_change: post.hiring.hires - pre.hiring.hires
  };
}

const juneBrief = {
  risk_level: "HIGH",
  growth_outlook: "DECLINING",
  efficiency_pressure: "MEDIUM",
  innovation_urgency: "HIGH",
  macro_condition: "NEUTRAL",
  expected_outcome: "DECLINE",
  key_risks: ["Churn is eating most new revenue", "Runway shrinks fast at current burn"],
  key_opportunities: ["Retention work has room to compound"],
  recommended_focus: ["Protect retention", "Keep acquisition lean"],
  confidence: 0.58
};

const julyBrief = {
  risk_level: "HIGH",
  growth_outlook: "STABLE",
  efficiency_pressure: "MEDIUM",
  innovation_urgency: "HIGH",
  macro_condition: "NEUTRAL",
  expected_outcome: "STAGNATION",
  key_risks: ["Churn is still high for this stage", "Runway is under 12 months"],
  key_opportunities: ["Acquisition cost is sustainable", "Early retention gains showing"],
  recommended_focus: ["Protect retention", "Hold pricing steady"],
  confidence: 0.62
};

const augustBrief = {
  risk_level: "MEDIUM",
  growth_outlook: "STABLE",
  efficiency_pressure: "MEDIUM",
  innovation_urgency: "HIGH",
  macro_condition: "NEUTRAL",
  expected_outcome: "STAGNATION",
  key_risks: ["Churn improving but still above healthy range", "Runway near 12 months"],
  key_opportunities: ["Retention gains are compounding", "Room to grow spend if churn holds"],
  recommended_focus: ["Keep retention the priority", "Modest acquisition restart"],
  confidence: 0.66
};

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
      values: { mrr: 28800, cash: 238000, costs: 47000, price: 85, churnMonthly: 5.6, newCustomers: 38, marketingSpend: 6000 },
      decisions: [
        { id: "d1", domain: "product", text: "Allocate ≈$9k to product work", state: "accepted" },
        { id: "d2", domain: "marketing", text: "Cut marketing to ≈$3k", state: "custom", note: "Did $4k instead" },
        { id: "d3", domain: "pricing", text: "Hold pricing", state: "accepted" }
      ]
    },
    {
      id: "m2", index: 1, enteredAt: iso(1),
      values: { mrr: 30000, cash: 220000, costs: 48000, price: 85, churnMonthly: 5.2, newCustomers: 41, marketingSpend: 6000 },
      decisions: [
        { id: "d4", domain: "product", text: "Keep product investment ≈$9k", state: "accepted" },
        { id: "d5", domain: "marketing", text: "Hold marketing ≈$5k on performance channels", state: "accepted" }
      ]
    },
    {
      id: "m3", index: 2, enteredAt: iso(0, 1),
      values: { mrr: 31900, cash: 208000, costs: 47000, price: 85, churnMonthly: 4.6, newCustomers: 44, marketingSpend: 4000 },
      decisions: []
    }
  ],
  analyses: [
    {
      id: "a1", monthId: "m1", createdAt: iso(2), source: "sample", llm_ok: true,
      brief: juneBrief,
      trace: makeTrace({
        month: 8, refreshReason: "initial",
        weights: { efficiency: 0.28, growth: 0.16, innovation: 0.46, macro: 0.1 },
        pre: { marketing: { spend: 10000, channel: "ppc" }, hiring: { hires: 1, cost_per_employee: 10000 }, product: { r_and_d_spend: 6000 }, pricing: { price_change_pct: 0 } },
        post: { marketing: { spend: 5000, channel: "ppc" }, hiring: { hires: 0, cost_per_employee: 10000 }, product: { r_and_d_spend: 9000 }, pricing: { price_change_pct: 0 } },
        final: { marketing: { spend: 3000, channel: "ppc" }, hiring: { hires: 0, cost_per_employee: 10000 }, product: { r_and_d_spend: 9000 }, pricing: { price_change_pct: 0 } },
        brief: juneBrief, memories: [memoryDecline]
      })
    },
    {
      id: "a2", monthId: "m2", createdAt: iso(1), source: "sample", llm_ok: true,
      brief: julyBrief,
      trace: makeTrace({
        month: 9, refreshReason: "event",
        weights: { efficiency: 0.29, growth: 0.17, innovation: 0.44, macro: 0.1 },
        pre: { marketing: { spend: 10000, channel: "ppc" }, hiring: { hires: 1, cost_per_employee: 10000 }, product: { r_and_d_spend: 8000 }, pricing: { price_change_pct: 0 } },
        post: { marketing: { spend: 5000, channel: "ppc" }, hiring: { hires: 0, cost_per_employee: 10000 }, product: { r_and_d_spend: 9600 }, pricing: { price_change_pct: 0 } },
        final: { marketing: { spend: 5000, channel: "ppc" }, hiring: { hires: 0, cost_per_employee: 10000 }, product: { r_and_d_spend: 9500 }, pricing: { price_change_pct: 0 } },
        brief: julyBrief, memories: [memoryDecline, memoryFlat]
      })
    },
    {
      id: "a3", monthId: "m3", createdAt: iso(0, 1), source: "sample", llm_ok: true,
      brief: augustBrief,
      trace: makeTrace({
        month: 10, refreshReason: "cadence",
        weights: { efficiency: 0.28, growth: 0.19, innovation: 0.43, macro: 0.1 },
        pre: { marketing: { spend: 10000, channel: "ppc" }, hiring: { hires: 1, cost_per_employee: 10000 }, product: { r_and_d_spend: 8000 }, pricing: { price_change_pct: 0 } },
        post: { marketing: { spend: 6500, channel: "ppc" }, hiring: { hires: 0, cost_per_employee: 10000 }, product: { r_and_d_spend: 8800 }, pricing: { price_change_pct: 0 } },
        final: { marketing: { spend: 6500, channel: "ppc" }, hiring: { hires: 0, cost_per_employee: 10000 }, product: { r_and_d_spend: 9000 }, pricing: { price_change_pct: 0 } },
        brief: augustBrief, memories: [memoryFlat, memoryDecline]
      })
    }
  ],
  settings: { narratives: false }
};
