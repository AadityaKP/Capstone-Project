// API client for the founder product contract (docs/founder_frontend_spec.md §28).
//
// Endpoints this client speaks (server work tracked as gaps G1–G3 in the spec):
//   GET  /api/health
//   POST /api/advise            — one analysis of a founder state (async-short)
//   POST /api/companies/{id}/months   (reserved; months are local-first in MVP)
//
// The UI is local-first: company + monthly snapshots live in the browser store.
// Only analysis requires the engine. When the API is unreachable, callers get
// {ok:false, offline:true} and render the spec §17.6 honest failure states —
// the UI never fabricates an analysis.

import { virtualHeadcount, deriveCac, CROWDEDNESS, MATURITY } from "./derive.js";

const BASE = "/api";

async function request(path, options = {}, timeoutMs = 120_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      return { ok: false, offline: false, status: response.status, error: body.detail || `Request failed (${response.status})` };
    }
    return { ok: true, data: body };
  } catch (error) {
    return { ok: false, offline: true, error: "Analysis service unreachable" };
  } finally {
    clearTimeout(timer);
  }
}

export async function health() {
  return request("/health", {}, 4000);
}

// Founder inputs → the engine's ScenarioConfig shape (§5 mappings):
// blended churn fills all three segments (§5.2); burn becomes virtual engine
// headcount at $8k slots (§5.4); crowdedness maps to a competitor count;
// maturity maps to a product-quality proxy. Macro fields stay at the engine's
// "typical conditions" defaults and are labelled estimated in the UI.
export function buildAdvisePayload(company, month) {
  const v = month.values;
  const cac = deriveCac(v);
  const crowd = CROWDEDNESS.find((c) => c.id === company.crowdedness) || CROWDEDNESS[1];
  const maturity = MATURITY.find((m) => m.id === company.maturity);
  return {
    company_id: company.id,
    company_age_months: (company.ageMonths || 0) + (month.index || 0),
    config: {
      company_name: company.name,
      initial_mrr: v.mrr,
      initial_cash: v.cash,
      average_price: v.price,
      cac: cac.value ?? 50,
      churn_enterprise: (v.churnEnt ?? v.churnMonthly) / 100,
      churn_smb: (v.churnSmb ?? v.churnMonthly) / 100,
      churn_b2c: (v.churnB2c ?? v.churnMonthly) / 100,
      competitors: crowd.competitors,
      product_quality: maturity ? maturity.quality : 0.5,
      initial_headcount: virtualHeadcount(v),
      monthly_burn_override: v.costs
    },
    history: (month.history || []).map((h) => ({
      mrr: h.mrr,
      churn: h.churnMonthly / 100,
      entered_at: h.enteredAt
    }))
  };
}

// One analysis (G1). Response contract:
// { analysis: { brief, trace, narratives?, llm_ok, created_at } }
export async function advise(company, month) {
  return request("/advise", {
    method: "POST",
    body: JSON.stringify(buildAdvisePayload(company, month))
  });
}
