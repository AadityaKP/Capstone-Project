from env.schemas import EnvState

def build_prompt(state: EnvState) -> str:
    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    return f"""You are a strategic advisor for a SaaS company.

Current State:
- MRR: {state.mrr:,.0f}
- Cash: {state.cash:,.0f}
- CAC: {state.cac:.1f}
- LTV: {state.ltv:,.0f}
- Churn (Avg): {avg_churn:.3f}
- Innovation: {state.innovation_factor:.3f}
- Unemployment: {state.unemployment:.1f}%
- Confidence: {state.consumer_confidence:.1f}
- Competitors: {state.competitors}
- Interest Rate: {state.interest_rate:.1f}%

Task:
Analyze the situation and return a JSON object with:

{{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "growth_outlook": "WEAK|STABLE|STRONG",
  "efficiency_pressure": "LOW|MEDIUM|HIGH|CRITICAL",
  "innovation_urgency": "LOW|MEDIUM|HIGH|CRITICAL",
  "macro_condition": "EXPANSION|NEUTRAL|RECESSION",
  "key_risks": ["..."],
  "key_opportunities": ["..."],
  "recommended_focus": ["..."],
  "confidence": 0.9
}}

Ensure valid enums and exactly 0.0 to 1.0 for confidence.
Only output JSON. Do not output markdown blocks or surrounding text.
"""
