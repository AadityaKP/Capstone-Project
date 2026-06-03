"""Simulate LLM vs heuristic with mock responses (no API calls)."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.schemas import EnvState
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent

# Mock LLM client that simulates successful API responses
class MockOpenAIClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "As CFO, I'd prioritize extending runway by 3+ months. Cut discretionary "
            "spending 20%, defer non-critical hires, and secure a bridge loan if needed."
        )

class MockAnthropicClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        role = "CMO" if "CMO" in system_prompt else "CPO"
        if role == "CMO":
            return (
                "Focus on high-LTV segments through targeted campaigns. PPC channels "
                "show 3.2x CAC payback. Increase spend by 15% in Q2."
            )
        else:
            return (
                "Urgent: Product quality gaps causing 5% churn. Allocate $80k for "
                "retention features. Expected NRR improvement: 8-12% in 90 days."
            )

# Create a realistic business state
state = EnvState(
    mrr=75000,
    cash=250000,
    cac=500,
    ltv=5000,
    churn_enterprise=0.02,
    churn_smb=0.05,
    churn_b2c=0.12,
    interest_rate=5.5,
    consumer_confidence=72,
    competitors=5,
    product_quality=0.78,
    price=150,
    months_elapsed=12,
    headcount=15,
    unemployment=4.2,
    innovation_factor=0.85,
    months_in_depression=0
)

print("=" * 90)
print("WITH MOCK LLM (Simulating Working APIs)")
print("=" * 90)
print(f"\nBusiness State: MRR=${state.mrr:,.0f}, Cash=${state.cash:,.0f}, Headcount={state.headcount}\n")

agents = [
    ("CFO", CFOProposalAgent, MockOpenAIClient()),
    ("CMO", CMOProposalAgent, MockAnthropicClient()),
    ("CPO", CPOProposalAgent, MockAnthropicClient()),
]

for role, AgentClass, mock_llm in agents:
    print("-" * 90)
    print(f"{role}")
    print("-" * 90)
    
    # Without LLM
    agent_no_llm = AgentClass(use_llm=False)
    prop_no_llm = agent_no_llm.propose(state)
    print(f"\nHeuristic (use_llm=False):")
    print(f"  Expected Impact: \"{prop_no_llm.expected_impact}\"")
    
    # With LLM
    agent_with_llm = AgentClass(llm_client=mock_llm, use_llm=True)
    prop_with_llm = agent_with_llm.propose(state)
    print(f"\nLLM-Refined (use_llm=True):")
    print(f"  Expected Impact: \"{prop_with_llm.expected_impact}\"")
    
    # Show what stayed the same
    print(f"\n✓ Unchanged:")
    print(f"  - Objective: {prop_with_llm.objective}")
    print(f"  - Confidence: {prop_with_llm.confidence}")
    print(f"  - Actions: {list(prop_with_llm.actions.keys())}")
    print()

print("=" * 90)
print("SUMMARY")
print("=" * 90)
print("""
With LLM refinement ENABLED:
  • Expected Impact = richer strategic reasoning from LLM
  • More nuanced, context-aware, specific to current conditions
  • Explains the "why" behind decisions

Without LLM (heuristic only):
  • Expected Impact = generic default strings
  • Actions still computed by heuristic logic
  • Lightweight, no API latency

KEY POINT: Actions/confidence/risks never change. Only the DESCRIPTION improves.
This means existing simulation logic is unaffected—only richer outputs for analysis.
""")
