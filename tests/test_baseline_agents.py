import sys
import os
import pytest
from unittest.mock import MagicMock

# Add project root to python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.schemas import EnvState
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent, merge_actions

# Mock state factory
def create_mock_state(
    cash=1_000_000,
    headcount=10,
    mrr=50_000,
    cac=100,
    ltv=500,
    churn_enterprise=0.01,
    churn_smb=0.02,
    churn_b2c=0.05,
    consumer_confidence=100.0
):
    return EnvState(
        mrr=mrr,
        cash=cash,
        cac=cac,
        ltv=ltv,
        churn_enterprise=churn_enterprise,
        churn_smb=churn_smb,
        churn_b2c=churn_b2c,
        interest_rate=3.0,
        consumer_confidence=consumer_confidence,
        competitors=5,
        product_quality=0.5,
        price=50.0,
        headcount=headcount,
        months_elapsed=0
    )

class TestCFOAgent:
    def test_hiring_freeze_low_runway(self):
        # Runway < 24 months. 
        # Burn approx 10 * 8000 = 80,000. 
        # Cash needed for 24 months = 1,920,000.
        # Set cash to 100,000 -> Runway ~1.2 months.
        state = create_mock_state(cash=100_000, headcount=10)
        agent = CFOAgent()
        action = agent.act(state)
        
        assert action["hiring"]["hires"] == 0
        
    def test_hiring_allowed_high_runway(self):
        # Cash = 5,000,000. Burn = 80,000. Runway ~62 months.
        # Efficiency: LTV 500 / CAC 100 = 5 > 3.
        state = create_mock_state(cash=5_000_000, headcount=10, ltv=500, cac=100)
        agent = CFOAgent()
        action = agent.act(state)
        
        assert action["hiring"]["hires"] == 1

    def test_hiring_freeze_inefficient(self):
        # High cash but low efficiency (LTV:CAC < 3)
        # LTV 200 / CAC 100 = 2 < 3.
        state = create_mock_state(cash=5_000_000, headcount=10, ltv=200, cac=100)
        agent = CFOAgent()
        action = agent.act(state)
        
        assert action["hiring"]["hires"] == 0

    def test_pricing_increase_inefficient(self):
        # LTV:CAC < 3 -> should raise price
        state = create_mock_state(ltv=200, cac=100)
        agent = CFOAgent()
        action = agent.act(state)
        
        assert action["pricing"]["price_change_pct"] == 0.05

class TestCMOAgent:
    def test_marketing_spend_aggressive(self):
        # LTV:CAC > 4 -> Spend 20,000
        state = create_mock_state(ltv=500, cac=100) # Efficiency 5
        agent = CMOAgent()
        action = agent.act(state)
        
        assert action["marketing"]["spend"] == 20000

    def test_marketing_spend_moderate(self):
        # LTV:CAC between 2 and 4 -> Spend 10,000
        state = create_mock_state(ltv=300, cac=100) # Efficiency 3
        agent = CMOAgent()
        action = agent.act(state)
        
        assert action["marketing"]["spend"] == 10000

    def test_marketing_channel_low_confidence(self):
        # Confidence < 90 -> PPC
        state = create_mock_state(consumer_confidence=80)
        agent = CMOAgent()
        action = agent.act(state)
        
        assert action["marketing"]["channel"] == "ppc"

class TestCPOAgent:
    def test_rd_spend_high_churn(self):
        # High churn > 4% -> Emergency RD
        state = create_mock_state(churn_enterprise=0.05, churn_smb=0.05, churn_b2c=0.05)
        agent = CPOAgent()
        action = agent.act(state)
        
        assert action["product"]["r_and_d_spend"] == 15000

    def test_rd_cut_low_cash(self):
        # Low cash < 200,000 -> Cut RD by half
        state = create_mock_state(cash=100_000, churn_enterprise=0.05, churn_smb=0.05, churn_b2c=0.05)
        agent = CPOAgent()
        action = agent.act(state)
        
        # Normal high churn spend is 15000. Cut by half = 7500.
        assert action["product"]["r_and_d_spend"] == 7500

class TestMergeActions:
    def test_merge_completeness(self):
        state = create_mock_state()
        bundle = merge_actions(state)
        
        assert "hiring" in bundle
        assert "pricing" in bundle
        assert "marketing" in bundle
        assert "product" in bundle
        assert bundle["hiring"]["cost_per_employee"] == 10000
        
