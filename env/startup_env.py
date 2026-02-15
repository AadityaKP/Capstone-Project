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
        
        # -------------------
        # Action Space
        # -------------------
        # We expect a Dictionary that maps to ActionBundle.
        # However, for Gym compatibility, we define a Dict space that roughly shapes it.
        # Real validation happens in Pydantic.
        self.action_space = spaces.Dict({
            "marketing": spaces.Dict({
                "spend": spaces.Box(0, np.inf, (1,)),
                "channel": spaces.Discrete(2) # 0=ppc, 1=brand
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

        # -------------------
        # Observation Space
        # -------------------
        # Vector: 
        # [mrr, cash, cac, ltv, churn_ent, churn_smb, churn_b2c, 
        #  interest, confidence, competitors, quality, months,
        #  valuation, unemployment, innovation, depression_months]
        low = np.array([0, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, np.inf, 200, np.inf, 1.0, sim_config.MAX_STEPS, np.inf, 100.0, 1.0, sim_config.MAX_STEPS], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state: EnvState = None
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize State with reasonable defaults (could be moved to config)
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
            price=50.0, # Initial ARPU
            months_elapsed=0,
            # Shock Engine State
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
        # 1. Decode Action (Assumes ActionAdapter has already cleaned it or it's a valid dict)
        # We manually construct models to ensure types are correct before logic
        try:
            action = ActionBundle(
                marketing=MarketingAction(**action_dict.get("marketing", {"spend": 0.0, "channel": "ppc"})),
                hiring=HiringAction(**action_dict.get("hiring", {"hires": 0, "cost_per_employee": 10000})),
                product=ProductAction(**action_dict.get("product", {"r_and_d_spend": 0.0})),
                pricing=PricingAction(**action_dict.get("pricing", {"price_change_pct": 0.0}))
            )
        except Exception as e:
            # Fallback for empty/bad actions
            print(f"Action Decoding Failed: {e}. Using defaults.")
            action = ActionBundle(
                marketing=MarketingAction(spend=0.0, channel="ppc"),
                hiring=HiringAction(hires=0, cost_per_employee=10000),
                product=ProductAction(r_and_d_spend=0.0),
                pricing=PricingAction(price_change_pct=0.0)
            )

        prev_mrr = self.state.mrr

        # --- 2. APPLY SHOCKS ---
        # Exogenous (Macro/External)
        business_logic.interest_rate_shock(self.state)
        business_logic.consumer_confidence_shock(self.state)
        business_logic.competitive_entry_shock(self.state)

        # Endogenous (Systemic Feedback)
        business_logic.apply_recession_cascade(self.state)
        business_logic.apply_hysteresis(self.state)

        # Recovery (Mean Reversion)
        business_logic.apply_recovery(self.state)

        # --- 3. APPLY LOGIC ---
        # Marketing → new MRR
        new_mrr = business_logic.compute_new_mrr(self.state, action.marketing)

        # Product → expansion
        expansion = business_logic.compute_expansion_mrr(self.state, action.product)

        # Churn
        churn_rate = business_logic.compute_churn_rate(self.state)

        # Update MRR (Organic + Marketing + Product)
        self.state.mrr = self.state.mrr * (1 - churn_rate) + new_mrr + expansion

        # Revenue Collection (Cash In) -> Happens BEFORE Pricing changes impact future MRR
        # User feedback: "update MRR fully -> then collect revenue"
        self.state.cash += self.state.mrr

        # Pricing effect (Updates MRR for NEXT step, but doesn't affect current cash collection?)
        # Actually user said: "state.cash += state.mrr then pricing effect mutates mrr... cash is collected at pre-pricing revenue... Correct order: update MRR fully -> then collect revenue"
        # Wait. If I update MRR first, then collect, that IS collecting post-update revenue.
        # The user said "state.cash += state.mrr THEN pricing effect mutates mrr" is WRONG.
        # Implies pricing effect SHOULD happen before collection? OR pricing effect should happen AFTER collection?
        # "Correct order: update MRR fully -> then collect revenue".
        # If "pricing effect mutates mrr", then it should correspond to the *next* period's billing or effective immediately? 
        # Usually price changes affect *future* billing or *current* if immediate.
        # Let's assume: 
        # 1. Base MRR update (Churn, New, Expansion).
        # 2. Pricing Effect (Elasticity impacts MRR).
        # 3. THEN Collect Revenue (Cash += Final MRR).
        
        business_logic.apply_pricing_effect(self.state, action.pricing) # Mutates MRR
        
        # NOW Collect (after all MRR mutations for this step are done)
        # Note: I previously had cash+=mrr BEFORE pricing. The user said that was "collecting at pre-pricing revenue" (which implies pricing SHOULD impact it).
        # BUT they also said "Correct order: update MRR fully -> then collect revenue".
        # So yes: Update MRR (including pricing) -> Collect.
        
        # Hiring (One-time cost)
        # CFO Constraint: Max hires based on 18-month runway
        # Max additional burn allowed = Cash / 18
        # Max hires = (Cash / 18) / cost_per_employee
        if action.hiring.hires > 0:
            max_hires = int((self.state.cash / 18.0) / action.hiring.cost_per_employee)
            if action.hiring.hires > max_hires:
                # CFO Reject: Not enough cash for safe runway
                action.hiring.hires = max_hires
        
        one_time_hiring_cost = action.hiring.hires * action.hiring.cost_per_employee
        business_logic.apply_hiring_cost(self.state, action.hiring) # Deducts one-time cost
        self.state.headcount += action.hiring.hires
        
        # Recurring Burn (Salaries)
        salary_burn = self.state.headcount * 8000.0
        
        # Deduct other burn (Marketing + Product)
        total_spend = action.marketing.spend + action.product.r_and_d_spend
        
        # DEDUCT BURN (Fixing Double Subtraction)
        # business_logic.apply_hiring_cost already deducted `one_time_hiring_cost`.
        # So we deduct `salary_burn` and `total_spend`.
        self.state.cash -= (salary_burn + total_spend)
        
        # --- UPDATE UNIT ECONOMICS ---
        # Calculate CAC (Marketing Spend / New Users)
        # We need to estimate 'new_users' generated this step. 
        # mrr_gain = new_mrr. Assuming avg price ~ state.price (approx)
        # new_users = new_mrr / state.price
        if self.state.price > 0:
            estimated_new_users = new_mrr / self.state.price
            raw_cac = business_logic.compute_cac(action.marketing.spend, estimated_new_users)
            self.state.cac = business_logic.scale_cac_by_macro(raw_cac, self.state)
        
        # Calculate LTV (ARPU / Churn)
        # ARPU = Price (simplified)
        self.state.ltv = business_logic.compute_ltv(self.state.price, churn_rate)

        # Calculate total burn for Rule of 40 (Spend + Hiring Cost + Salaries)
        rule40_burn = one_time_hiring_cost + salary_burn + total_spend

        # --- 4. REWARDS & METRICS ---
        rule40 = business_logic.compute_rule_of_40(prev_mrr, self.state.mrr, rule40_burn)
        reward = business_logic.compute_reward(self.state, rule40)

        # Time progression
        self.state.months_elapsed += 1

        terminated = self.state.cash <= 0
        truncated = self.state.months_elapsed >= 120 # 10 years or config max

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
