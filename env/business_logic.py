import random
import math
from env.schemas import EnvState, MarketingAction, ProductAction, PricingAction, HiringAction


def _stream(rng):
    """The random source to draw from.

    None means the global `random` module, which is what every caller used
    before StartupEnv gained a private generator. Passing an rng isolates the
    physics from anything else in the process that draws - see
    StartupEnv.deterministic_rng for why that matters for cross-policy
    comparison.
    """
    return rng if rng is not None else random


# The engine's original cost convention: one employee costs this much per month,
# and headcount is the only thing that sets burn. Every subsystem needing a burn
# figure reimplemented `headcount * 8000` locally - the physics, the boardroom,
# the Oracle, the prompt builder and two agent modules - which made the constant
# a protocol rather than a number, and made it impossible to change in one place.
SALARY_SLOT_USD = 8000.0


def monthly_burn(state: EnvState) -> float:
    """The company's fixed monthly operating cost.

    `state.monthly_burn` carries the founder's actual costs when a product
    surface supplied them. None means "not supplied" and falls back to the
    headcount-slot convention, so every recorded research run reproduces
    byte-identically.

    The fallback is not a rounding error at founder scale, which is the whole
    reason this function exists. The client used to encode costs as
    `max(1, round(max(costs - marketing, 8000) / 8000))`, so every company with
    monthly costs between $0 and $12,000 was charged exactly $8,000 - sixteen
    times over for a founder spending $500, which killed the company in month 0
    of every projection regardless of which plan was being tested.
    """
    if state.monthly_burn is not None:
        return float(state.monthly_burn)
    return state.headcount * SALARY_SLOT_USD


def interest_rate_shock(state: EnvState, prob: float = 0.1, rng=None) -> None:
    if _stream(rng).random() < prob:
        state.interest_rate += 1.5
        state.valuation_multiple *= 0.85
        state.churn_smb *= 1.2

def consumer_confidence_shock(state: EnvState, prob: float = 0.1, rng=None) -> None:
    if _stream(rng).random() < prob:
        state.consumer_confidence -= 20
        state.unemployment += 1.0

def competitive_entry_shock(state: EnvState, prob: float = 0.1, rng=None) -> None:
    market_attractiveness = (state.mrr - 50_000) / 50_000
    dynamic_prob = 1 / (1 + math.exp(-market_attractiveness))
    actual_prob = prob * (2 * dynamic_prob)

    if _stream(rng).random() < actual_prob:
        state.competitors += 1
        state.price *= 0.9


def inject_hard_shock(state: EnvState, shock_type: str) -> str:
    """
    Deterministic, hard shocks for controlled experiments.
    Returns a shock label string for the Oracle prompt.
    """
    if shock_type == "competitor_surge":
        state.competitors += 3
        state.price *= 0.75
        state.churn_smb *= 1.5
        return "COMPETITOR_SURGE: 3 new entrants, forced price cut, SMB churn +50%"
    if shock_type == "rate_hike":
        state.interest_rate += 4.0
        state.valuation_multiple *= 0.6
        state.consumer_confidence -= 25
        return "RATE_HIKE: +400bps, valuation -40%, confidence crash"
    if shock_type == "recession":
        state.consumer_confidence = 55
        state.unemployment += 4.0
        state.churn_b2c *= 2.0
        return "RECESSION: confidence=55, unemployment spike, B2C churn doubled"
    return "NO_SHOCK"

def apply_recession_cascade(state: EnvState, rng=None, always_draw: bool = False) -> None:
    """
    Credit-Bankruptcy Loop.
    If Unemployment High + Rates High -> Confidence Crash.

    `always_draw` is the whole reason cross-policy comparison can be trusted.

    This is the only conditional draw in the module: every other call site
    consumes exactly one number per step regardless of state, but this one draws
    only when unemployment > 8 AND interest_rate > 7. That makes the per-step
    consumption of the shared random stream *state-dependent*, so two policies
    that reach different macro states fall out of step with each other and stop
    experiencing the same world - measured at 7 draws/step normally against 8
    once the condition holds, and reached in 20 of 20 episodes at a median of
    around month 33, inside the months 25-60 window the thesis analyses.

    Drawing unconditionally and testing the result afterwards is identical in
    distribution and fixes the desynchronisation outright. It does consume one
    extra number per step, so it changes trajectories and cannot be the default
    without invalidating every recorded result; StartupEnv turns it on together
    with its private generator via `deterministic_rng`.
    """
    if always_draw:
        roll = _stream(rng).random()
        if state.unemployment > 8.0 and state.interest_rate > 7.0 and roll < 0.2:
            state.consumer_confidence -= 10
            state.valuation_multiple *= 0.8
            state.unemployment += 0.5
        return

    if state.unemployment > 8.0 and state.interest_rate > 7.0:
        if _stream(rng).random() < 0.2:
            state.consumer_confidence -= 10
            state.valuation_multiple *= 0.8
            state.unemployment += 0.5

def apply_hysteresis(state: EnvState) -> None:
    """
    Growth Hysteresis.
    Long depressions permanently scar innovation.
    """
    if state.consumer_confidence < 50:
        state.months_in_depression += 1
    else:
        state.months_in_depression = max(0, state.months_in_depression - 1)

    if state.months_in_depression >= 6:
        state.innovation_factor *= 0.95

    state.innovation_factor = max(0.0, min(1.0, state.innovation_factor))

def apply_recovery(state: EnvState) -> None:
    """
    Mean-reversion mechanics.
    """
    if state.innovation_factor < 1.0:
        state.innovation_factor += 0.003

    if state.valuation_multiple < 10.0:
        state.valuation_multiple += 0.05
    elif state.valuation_multiple > 10.0:
        state.valuation_multiple -= 0.05

    if state.consumer_confidence < 100 and state.unemployment < 8.0:
        state.consumer_confidence += 2.0

def hill_response(spend: float, alpha: float, beta: float, gamma: float) -> float:
    """
    Hill Function for Marketing Response.
    alpha: Shape parameter (S-curve steepness)
    beta: Max potential capacity (Saturation point)
    gamma: Half-saturation point (Spend needed to reach 50% of beta)
    """
    if spend <= 0: return 0.0
    return beta * (spend ** alpha) / (gamma ** alpha + spend ** alpha)

# Share of its existing customer base a company could plausibly add in one month
# at full marketing saturation. This is the one free parameter left in the
# scale-aware curve below, and no public dataset fixes it - it is ASSUMED, and
# must be reported as such wherever the calibration provenance is surfaced.
SATURATION_ACQUISITION_RATE = 0.20


def marketing_curve_params(state: EnvState, channel: str, rng=None) -> tuple[float, float, float]:
    """Hill parameters expressed in customers, not bare dollars.

    The original constants were dimensionally wrong: gamma is a spend level, but
    it was drawn from uniform(15_000, 50_000) with no reference to who was being
    bought. A $12k-MRR company spending $3k therefore sat at ~1% of potential on
    the brand curve, so every policy above the physics correctly concluded that
    small companies should spend far more than they earn.

    Reparameterised so both anchors scale with the company:

        acquirable = current_customers * SATURATION_ACQUISITION_RATE
        beta       = acquirable * price          # max new MRR per month
        gamma      = (acquirable / 2) * CAC      # spend that buys half of them

    gamma is now "what it costs to acquire half the customers you could plausibly
    win this month", which is a quantity with units that make sense. alpha (curve
    shape) is unchanged - it is a shape parameter, not a scale one.
    """
    current_customers = state.mrr / max(1.0, state.price)
    acquirable = max(1.0, current_customers * SATURATION_ACQUISITION_RATE)
    cac = max(1.0, state.cac)

    beta = acquirable * max(1.0, state.price)
    gamma = max(1.0, (acquirable / 2.0) * cac)

    draw = _stream(rng)
    if channel == "ppc":
        alpha = draw.uniform(0.5, 1.0)
    else:
        # Brand converts more slowly at low spend and compounds harder at high
        # spend; it also reaches further than performance at saturation.
        alpha = draw.uniform(1.5, 3.0)
        beta *= 1.5

    return alpha, beta, gamma


def compute_new_mrr(state: EnvState, action: MarketingAction, scale_aware: bool = False,
                    rng=None) -> float:
    draw = _stream(rng)
    if scale_aware:
        alpha, beta, gamma = marketing_curve_params(state, action.channel, rng=rng)
    elif action.channel == "ppc":
        alpha = draw.uniform(0.5, 1.0)
        gamma = draw.uniform(15_000, 50_000)
        beta = draw.uniform(10_000, 50_000)
    else:
        alpha = draw.uniform(1.5, 3.0)
        gamma = draw.uniform(15_000, 50_000)
        beta = draw.uniform(50_000, 100_000)

    response = hill_response(action.spend, alpha, beta, gamma)

    if state.consumer_confidence < 80:
        response *= 0.85
    elif state.consumer_confidence > 120:
        response *= 1.08

    if state.competitors >= 10:
        response *= 0.6
    elif state.competitors >= 4:
        response *= 0.8

    return response

def compute_churn_rate(state: EnvState) -> float:
    base = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3

    quality_factor = 1.0 - (state.product_quality * 0.5) 

    macro_multiplier = 1.0
    if state.consumer_confidence < 80:
        macro_multiplier *= 1.3
        
    avg_tenure_proxy = max(1, state.months_elapsed * 0.4)
    tenure_decay = math.exp(-0.15 * avg_tenure_proxy)
    
    decay_multiplier = max(0.3, tenure_decay)

    return base * quality_factor * macro_multiplier * decay_multiplier

# R&D spend at which a company reaches half the achievable monthly product
# improvement, as a share of its own revenue. SaaS Capital's 2026 survey puts
# median R&D for private B2B SaaS at 24% of ARR, which for a monthly figure is
# 24% of MRR: a company spending what the median company spends buys half the
# achievable rate.
#
# This deliberately does NOT preserve the old constant's calibration point. The
# original half-saturation was $100,000, which at the $50k-MRR company the
# engine was tuned for means spending twice your revenue on R&D to get half the
# available improvement. That is not a plausible anchor at any company size, and
# it is part of why the lever did nothing. Replacing it with a published median
# is a change of belief about the world, not a refactor, and is why this sits
# behind a flag.
#
# The median is published; placing half-saturation AT the median is a modelling
# judgement, as is MAX_MONTHLY_QUALITY_GAIN. Both are ASSUMED and are reported
# as such wherever calibration provenance is surfaced.
RND_HALF_SATURATION_SHARE = 0.24

# Ceiling on how much of the remaining quality headroom one month of R&D can
# close, at any spend. Carried over from the original curve unchanged.
MAX_MONTHLY_QUALITY_GAIN = 0.05

# product_quality moves at half the rate innovation_factor does, as it always
# has - R&D repairs capability faster than customers feel it.
QUALITY_GAIN_SHARE = 0.5

# Floor under the revenue a spend share is measured against, so a pre-revenue
# company does not divide by zero and read every dollar as total saturation.
MIN_SCALE_MRR = 1_000.0


def apply_innovation_investment(state: EnvState, action: ProductAction,
                                scale_aware: bool = False) -> None:
    """Converts R&D spend into innovation gains (nonlinear, saturating).

    The original had two defects that compounded into a lever that did
    literally nothing for a founder, not merely too little:

      1. `gain *= (1.0 - state.innovation_factor)`, applied to BOTH outputs.
         innovation_factor is a scarring variable - it starts at 1.0 and only
         depression hysteresis pushes it down - and every founder analysis
         starts it at 1.0. So the multiplier was exactly zero, and $500,000 of
         R&D moved product_quality by 0.000000. Since product_quality is the
         only input to compute_churn_rate a plan can move, churn was identical
         across every plan by construction.
      2. The $100,000 saturation constant was absolute, so the same dollar of
         R&D meant something different at every company size.

    Scale-aware separates the two headrooms - innovation_factor still repairs
    scarring, product_quality gets its own - and measures spend as a share of
    the company's own revenue.
    """
    spend = action.r_and_d_spend

    if spend <= 0:
        return

    if scale_aware:
        share = spend / max(state.mrr, MIN_SCALE_MRR)
        response = share / (share + RND_HALF_SATURATION_SHARE)
        gain = MAX_MONTHLY_QUALITY_GAIN * response
        # Each variable closes its own headroom. A whole innovation_factor has
        # nothing left to repair, which is correct; it must not also mean the
        # product cannot improve.
        state.innovation_factor += gain * (1.0 - state.innovation_factor)
        state.product_quality += (
            gain * (1.0 - state.product_quality) * QUALITY_GAIN_SHARE
        )
    else:
        # Saturation curve (Hill-type response)
        scale = 100_000  # tuning parameter
        gain = (spend / (spend + scale)) * 0.05  # max ~0.05/month

        # Harder to improve when already high
        gain *= (1.0 - state.innovation_factor)

        state.innovation_factor += gain

        # Ensures innovation -> lower churn (since churn uses product_quality)
        state.product_quality += gain * 0.5

    state.product_quality = min(1.0, max(0.0, state.product_quality))
    state.innovation_factor = min(1.0, max(0.0, state.innovation_factor))


def compute_expansion_mrr(state: EnvState, action: ProductAction,
                          scale_aware: bool = False) -> float:
    """Upsell into the existing base, lifted by R&D.

    The $50,000 saturation constant is the $50k-MRR calibration company
    spending 1.0x its revenue to reach the cap, so the scale-aware form uses
    exactly that multiple and the calibration point is preserved - unlike
    apply_innovation_investment above, whose old anchor was not defensible at
    any size and had to be replaced rather than rescaled.

    The flat 2% underneath is untouched, and remains the largest single term
    in a founder's projection: it is why the do-nothing arm still grows.
    """
    effective_rnd = action.r_and_d_spend * state.innovation_factor
    saturation = max(state.mrr, MIN_SCALE_MRR) if scale_aware else 50_000
    upsell_factor = 1 + min(effective_rnd / saturation, 0.5)
    return state.mrr * 0.02 * upsell_factor

def apply_pricing_effect(state: EnvState, action: PricingAction, rng=None) -> None:
    elasticity = _stream(rng).uniform(-0.9, -0.2)
    demand_change = elasticity * action.price_change_pct
    
    state.price *= (1 + action.price_change_pct)
    
    state.mrr *= (1 + action.price_change_pct) * (1 + demand_change)

def apply_hiring_cost(state: EnvState, action: HiringAction) -> None:
    total_cost = action.hires * action.cost_per_employee
    state.cash -= total_cost

def compute_cac(marketing_spend: float, new_users: float) -> float:
    if new_users <= 0: return 0.0 
    raw_cac = marketing_spend / new_users
    return raw_cac

def scale_cac_by_macro(raw_cac: float, state: EnvState) -> float:
    modifier = 1.0
    
    if state.interest_rate > 5.0:
        modifier *= 1.2
        
    if state.consumer_confidence < 80:
        modifier *= 1.3
    elif state.consumer_confidence > 120:
        modifier *= 0.8
        
    if state.competitors > 5:
        modifier *= 1.15
        
    if state.competitors >= 8:
         modifier *= 1.3
        
    return raw_cac * modifier

def compute_ltv(mrr_per_user: float, churn_rate: float, discount_rate: float = 0.0) -> float:
    if churn_rate <= 0.001: churn_rate = 0.001 
    return mrr_per_user / churn_rate

def compute_rule_of_40(prev_mrr: float, new_mrr: float, burn: float) -> float:
    if prev_mrr <= 0: prev_mrr = 1.0 
    if new_mrr <= 0: new_mrr = 1.0
    
    growth_pct = ((new_mrr - prev_mrr) / prev_mrr) * 100
    margin_pct = (-burn / new_mrr) * 100
    return growth_pct + margin_pct

def compute_reward(state: EnvState, rule_of_40: float) -> float:
    reward = state.mrr / 1_000_000 

    if rule_of_40 < 15:
        reward -= 2
    if rule_of_40 < 0:
        reward -= 5

    if state.cac > 0 and state.ltv > 0:
        ratio = state.ltv / state.cac
        if ratio < 3.0:
            reward -= 5.0 
            if ratio < 1.0:
                reward -= 10.0 

    if state.cash <= 0:
        reward -= 20

    if state.innovation_factor < 0.8:
        reward -= 5

    if state.valuation_multiple < 5.0:
        reward -= 2

    return reward
