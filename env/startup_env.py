import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from config import sim_config
from env.schemas import EnvState, ActionBundle, MarketingAction, HiringAction, ProductAction, PricingAction
from env import business_logic

class StartupEnv(gym.Env):
    """
    The Gymnasium Environment for the Startup Simulator (Physics Engine).
    
    Responsibilities:
    1. Maintain the State (EnvState).
    2. Orchestrate time steps (monthly).
    3. Decode ActionBundle.
    4. Run Business Logic (Shocks -> Physics -> Financials).
    5. Calculate Rewards (Rule of 40).
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(StartupEnv, self).__init__()
        
        self.action_space = spaces.Dict({
            "marketing": spaces.Dict({
                "spend": spaces.Box(0, np.inf, (1,)),
                "channel": spaces.Discrete(2) 
            }),
            "hiring": spaces.Dict({
                "hires": spaces.Box(0, np.inf, (1,)),
                "cost_per_employee": spaces.Box(0, np.inf, (1,))
            }),
            "product": spaces.Dict({
                "r_and_d_spend": spaces.Box(0, np.inf, (1,))
            }),
            "pricing": spaces.Dict({
                "price_change_pct": spaces.Box(-1.0, 10.0, (1,))
            })
        })

        low = np.array([0, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, np.inf, 200, np.inf, 1.0, sim_config.MAX_STEPS, np.inf, 100.0, 1.0, sim_config.MAX_STEPS], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state: EnvState = None
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.state = EnvState(
            mrr=50_000,
            cash=sim_config.INITIAL_CASH,
            cac=sim_config.BASE_CAC,
            ltv=7_000,
            churn_enterprise=0.01,
            churn_smb=0.03,
            churn_b2c=0.05,
            interest_rate=3.0,
            consumer_confidence=100.0,
            competitors=5,
            product_quality=sim_config.INITIAL_PRODUCT_QUALITY,
            price=50.0, 
            months_elapsed=0,
            valuation_multiple=10.0,
            unemployment=4.0,
            innovation_factor=1.0,
            months_in_depression=0
        )
        
        return self._get_obs(), {}

    def step(self, action_dict: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advances the simulation by 1 Month.
        """
        try:
            action = ActionBundle(
                marketing=MarketingAction(**action_dict.get("marketing", {"spend": 0.0, "channel": "ppc"})),
                hiring=HiringAction(**action_dict.get("hiring", {"hires": 0, "cost_per_employee": 10000})),
                product=ProductAction(**action_dict.get("product", {"r_and_d_spend": 0.0})),
                pricing=PricingAction(**action_dict.get("pricing", {"price_change_pct": 0.0}))
            )
        except Exception as e:
            print(f"Action Decoding Failed: {e}. Using defaults.")
            action = ActionBundle(
                marketing=MarketingAction(spend=0.0, channel="ppc"),
                hiring=HiringAction(hires=0, cost_per_employee=10000),
                product=ProductAction(r_and_d_spend=0.0),
                pricing=PricingAction(price_change_pct=0.0)
            )

        prev_mrr = self.state.mrr

        business_logic.interest_rate_shock(self.state)
        business_logic.consumer_confidence_shock(self.state)
        business_logic.competitive_entry_shock(self.state)

        business_logic.apply_recession_cascade(self.state)
        business_logic.apply_hysteresis(self.state)

        business_logic.apply_recovery(self.state)

        new_mrr = business_logic.compute_new_mrr(self.state, action.marketing)

        expansion = business_logic.compute_expansion_mrr(self.state, action.product)

        churn_rate = business_logic.compute_churn_rate(self.state)

        self.state.mrr = self.state.mrr * (1 - churn_rate) + new_mrr + expansion

        self.state.cash += self.state.mrr

        business_logic.apply_pricing_effect(self.state, action.pricing) 
        
        if action.hiring.hires > 0:
            max_hires = int((self.state.cash / 18.0) / action.hiring.cost_per_employee)
            if action.hiring.hires > max_hires:
                action.hiring.hires = max_hires
        
        one_time_hiring_cost = action.hiring.hires * action.hiring.cost_per_employee
        business_logic.apply_hiring_cost(self.state, action.hiring) 
        self.state.headcount += action.hiring.hires
        
        salary_burn = self.state.headcount * 8000.0
        
        total_spend = action.marketing.spend + action.product.r_and_d_spend
        
        self.state.cash -= (salary_burn + total_spend)
        
        if self.state.price > 0:
            estimated_new_users = new_mrr / self.state.price
            raw_cac = business_logic.compute_cac(action.marketing.spend, estimated_new_users)
            self.state.cac = business_logic.scale_cac_by_macro(raw_cac, self.state)
        
        self.state.ltv = business_logic.compute_ltv(self.state.price, churn_rate)

        rule40_burn = one_time_hiring_cost + salary_burn + total_spend

        rule40 = business_logic.compute_rule_of_40(prev_mrr, self.state.mrr, rule40_burn)
        reward = business_logic.compute_reward(self.state, rule40)

        self.state.months_elapsed += 1

        terminated = self.state.cash <= 0
        truncated = self.state.months_elapsed >= 120 

        return self._get_obs(), reward, terminated, truncated, {
            "rule_of_40": rule40,
            "state": self.state.model_dump()
        }

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.state.mrr,
            self.state.cash,
            self.state.cac,
            self.state.ltv,
            self.state.churn_enterprise,
            self.state.churn_smb,
            self.state.churn_b2c,
            self.state.interest_rate,
            self.state.consumer_confidence,
            self.state.competitors,
            self.state.product_quality,
            self.state.months_elapsed,
            self.state.valuation_multiple,
            self.state.unemployment,
            self.state.innovation_factor,
            self.state.months_in_depression
        ], dtype=np.float32)
