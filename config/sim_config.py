# Simulation Configuration
# ========================
# This file serves as the "Control Panel" for the entire simulation.
# All magic numbers and constants are centralized here to allow for easy tuning
# without diving into the business logic code.

# -------------------
# Time Configurations
# -------------------
# The simulation runs in discrete weekly steps.
# 52 weeks/year * 5 years = 260 steps total.
MAX_STEPS = 52 * 5 

# ------------------------
# Initial State Parameters
# ------------------------
# These values define the starting point of the startup (Week 0).

INITIAL_CASH = 1000000.0  # Starting Capital ($1M Seed Round).
INITIAL_USERS = 100       # Initial user base (Friends & Family beta).
INITIAL_BRAND = 0.0       # Brand awareness starts at 0.
INITIAL_PRODUCT_QUALITY = 0.1 # MVP level quality (very rough).

# -----------------
# Market Dynamics
# -----------------
# These parameters define the "Difficulty" of the market.

BASE_CAC = 50.0           # Cost Per Acquisition baseline. Marketing efficiency is relative to this.
MARKET_VOLATILITY = 0.05  # Standard deviation (5%) for random shocks to Revenue/metrics.
                          # Higher volatility = more unpredictable market.

# Capping Churn (User Attrition)
MAX_CHURN = 0.30          # Maximum 30% of users can leave per week (Catastrophic failure).
MIN_CHURN = 0.02          # Minimum 2% distinct "natural churn" that cannot be removed.

# ----------------------------
# operational Constraints
# ----------------------------
MIN_BURN_RATE = 5000.0    # Fixed operational overhead (legal, server costs, etc.) 
                          # Independent of headcount.
