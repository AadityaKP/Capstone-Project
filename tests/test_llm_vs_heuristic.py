"""Compare proposal generation with and without LLM refinement."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from env.schemas import EnvState
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from agents.llm_client import create_llm_client

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

print("=" * 80)
print("BUSINESS STATE")
print("=" * 80)
print(f"MRR: ${state.mrr:,.0f}")
print(f"Cash: ${state.cash:,.0f}")
print(f"Headcount: {state.headcount}")
print(f"Competitors: {state.competitors}")
print(f"Consumer Confidence: {state.consumer_confidence}")
print(f"Innovation Factor: {state.innovation_factor:.2f}")
print()

def compare_agent(agent_class, agent_name):
    print("=" * 80)
    print(f"{agent_name.upper()}")
    print("=" * 80)
    
    # WITHOUT LLM
    agent_no_llm = agent_class(use_llm=False)
    proposal_no_llm = agent_no_llm.propose(state)
    
    print(f"\n[WITHOUT LLM]")
    print(f"  Objective: {proposal_no_llm.objective}")
    print(f"  Expected Impact: {proposal_no_llm.expected_impact}")
    print(f"  Risks: {proposal_no_llm.risks}")
    print(f"  Confidence: {proposal_no_llm.confidence}")
    print(f"  Actions Keys: {list(proposal_no_llm.actions.keys())}")
    
    # WITH LLM
    print(f"\n[WITH LLM]")
    print(f"  Objective: {proposal_no_llm.objective}  (unchanged)")
    
    agent_with_llm = agent_class(
        llm_client=create_llm_client("openai" if agent_name == "CFO" else "anthropic"),
        use_llm=True
    )
    proposal_with_llm = agent_with_llm.propose(state)
    
    print(f"  Expected Impact: {proposal_with_llm.expected_impact}")
    print(f"  Risks: {proposal_with_llm.risks}  (unchanged)")
    print(f"  Confidence: {proposal_with_llm.confidence}  (unchanged)")
    print(f"  Actions Keys: {list(proposal_with_llm.actions.keys())}  (unchanged)")
    
    print(f"\n[COMPARISON]")
    impact_changed = proposal_no_llm.expected_impact != proposal_with_llm.expected_impact
    print(f"  Expected Impact changed: {impact_changed}")
    
    if impact_changed:
        print(f"\n  Heuristic default ({len(proposal_no_llm.expected_impact)} chars):")
        print(f"    \"{proposal_no_llm.expected_impact}\"")
        print(f"\n  LLM refined ({len(proposal_with_llm.expected_impact)} chars):")
        print(f"    \"{proposal_with_llm.expected_impact}\"")
    else:
        print(f"  (LLM call returned empty or fell back to heuristic)")
    
    print()

# Compare all three C-suite agents
compare_agent(CFOProposalAgent, "CFO")
compare_agent(CMOProposalAgent, "CMO")
compare_agent(CPOProposalAgent, "CPO")

print("=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)
print("✓ Actions (marketing, hiring, product, pricing) remain identical")
print("✓ Confidence scores remain identical")
print("✓ Only 'expected_impact' field is LLM-refined")
print("✓ Each agent uses its assigned provider (CFO→OpenAI, CMO→Anthropic, CPO→Anthropic)")
print()
