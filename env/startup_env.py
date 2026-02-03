import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from config import sim_config
from env.schemas import StartupState
from env import business_logic

class StartupEnv(gym.Env):
    """
    The Gymnasium Environment for the Startup Simulator.
    
    Responsibilities:
    1. Maintain the State (StartupState).
    2. Orchestrate time steps (week by week).
    3. Route Actions to proper Logic handlers.
    4. Enforce Invariants (e.g., Cash cannot be negative).
    5. Calculate Rewards.
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(StartupEnv, self).__init__()
        
        # -------------------
        # Action Space
        # -------------------
        # We formally define a Dict space for Gym compatibility, 
        # though the 'step' function is flexible with inputs.
        # "type": Discrete(5) maps to -> 0:marketing, 1:hiring, 2:product, 3:pricing, 4:skip
        self.action_space = spaces.Dict({
            "type": spaces.Discrete(5), 
            "amount": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32) 
        })

        # -------------------
        # Observation Space
        # -------------------
        # Defines the bounds of what the agent can see.
        # Vector: [cash, users, revenue, burn, cac, churn, growth, quality, brand, price, time]
        low = np.array([0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0], dtype=np.float32)
        # using np.inf for unbounded upper limits
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, np.inf, np.inf, sim_config.MAX_STEPS], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_state: StartupState = None
        
        # Internal State (Hidden from Agent directly, but effects are visible)
        self.headcount = 1  # Starting with just the Founder.
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to Week 0.
        Must be called before the first step.
        """
        super().reset(seed=seed)
        
        # Initialize State with Config defaults
        self.current_state = StartupState(
            cash=sim_config.INITIAL_CASH,
            users=sim_config.INITIAL_USERS,
            revenue=0.0,
            burn_rate=sim_config.MIN_BURN_RATE,
            cac=sim_config.BASE_CAC,
            churn=sim_config.MIN_CHURN,
            growth_rate=0.0,
            product_quality=sim_config.INITIAL_PRODUCT_QUALITY,
            brand_strength=sim_config.INITIAL_BRAND,
            price=10.0, # Default entry price
            time_step=0
        )
        
        self.headcount = 1
        
        return self._get_obs(), {}

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        The Heart of the Simulator.
        Advances time by 1 unit (1 Week).
        
        Process:
        1. Parse & Apply Action (Spend Cash, Change Price).
        2. Update Financials (Deduct Burn).
        3. Enforce Invariants (Check Bankruptcy).
        4. Run Market Dynamics (Revenue, Churn, Stochastic events).
        5. Check Termination Conditions.
        6. Calculate Reward.
        """
        
        # --- 1. CONFIGURATION & ACTION APPLICATION ---
        action_type = action.get("type", "skip")
        action_params = action.get("params", {})
        
        spend = 0.0 # Variable costs incurred THIS specific step (one-time).
        
        # Route actions to Logic
        if action_type == "marketing":
            spend = action_params.get("amount", 0.0)
            self.current_state.users, self.current_state.brand_strength = \
                business_logic.apply_marketing_effect(self.current_state.users, self.current_state.brand_strength, spend)
        
        elif action_type == "pricing":
            new_price = action_params.get("price", self.current_state.price)
            self.current_state.price = float(new_price)
            
        elif action_type == "hiring":
            hire_count = int(action_params.get("count", 0))
            if hire_count > 0:
                # Recruiting Fee: $2000 one-time cost per hire.
                spend += hire_count * 2000.0 
                self.headcount += hire_count
                
        elif action_type == "product":
            invest_amount = action_params.get("amount", 0.0)
            spend += invest_amount
            # Improve product quality through investment
            self.current_state.product_quality = business_logic.apply_product_investment(
                self.current_state.product_quality, 
                invest_amount
            )
                
        # --- 2. FINANCIAL UPDATES ---
        # Calculate weekly burn (Fixed + Salaries + This week's One-time Spend)
        self.current_state.burn_rate = business_logic.calculate_burn(self.headcount, 0.0) + spend
        self.current_state.cash -= self.current_state.burn_rate
        
        # CRITICAL INVARIANT: Cash cannot be negative.
        # If it drops below zero, we clamp it to 0.0 and will terminate later.
        if self.current_state.cash < 0:
            self.current_state.cash = 0.0
        
        # --- 3. MARKET DYNAMICS (The "World" Reacts) ---
        
        # Churn: Users leaving depending on Price vs Value
        self.current_state.churn = business_logic.calculate_churn(
            self.current_state.product_quality, 
            self.current_state.price
        )
        
        # Remove churned users
        churned_users = int(self.current_state.users * self.current_state.churn)
        self.current_state.users = max(0, self.current_state.users - churned_users)
        
        # Revenue Generation: Users * Price + Random Market Noise
        raw_revenue = business_logic.calculate_revenue(self.current_state.users, self.current_state.price)
        self.current_state.revenue = business_logic.apply_stochastic_shock(raw_revenue)
        
        # Add Revenue to Cash
        self.current_state.cash += self.current_state.revenue
        
        # --- 4. TIME & TERMINATION ---
        self.current_state.time_step += 1
        
        terminated = False # Reached terminal state (Bankruptcy)
        truncated = False  # Time Limit Exceeded
        
        # Bankruptcy Condition
        if self.current_state.cash <= 0:
            terminated = True
            
        # Time Limit Condition
        if self.current_state.time_step >= sim_config.MAX_STEPS:
            truncated = True
            
        # --- 5. REWARD FUNCTION ---
        # Current Goal: Profitability + Growth.
        # Reward = Net Profit - Churn Penalty.
        reward = (self.current_state.revenue - self.current_state.burn_rate)
        # Heavy penalty for losing users
        reward -= (churned_users * 10.0) 
        
        # Massive penalty for Bankruptcy to discourage reckless spending
        if terminated and self.current_state.cash <= 0:
            reward -= 10000.0
            
        return self._get_obs(), reward, terminated, truncated, {"state": self.current_state.model_dump()}

    def _get_obs(self) -> np.ndarray:
        """
        Helper to flatten the State Dict into a Numpy Array for Gym.
        """
        return np.array([
            self.current_state.cash,
            self.current_state.users,
            self.current_state.revenue,
            self.current_state.burn_rate,
            self.current_state.cac,
            self.current_state.churn,
            self.current_state.growth_rate,
            self.current_state.product_quality,
            self.current_state.brand_strength,
            self.current_state.price,
            self.current_state.time_step
        ], dtype=np.float32)
