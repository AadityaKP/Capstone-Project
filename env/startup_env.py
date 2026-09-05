import random

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

    def __init__(self, initial_config: Dict[str, Any] | None = None):
        super(StartupEnv, self).__init__()
        self.initial_config = initial_config or {}
        self.max_steps = int(self.initial_config.get("max_months", sim_config.MAX_STEPS))
        # Opt-in: marketing response anchored to the company's own customers and
        # CAC rather than to absolute dollar constants. Off by default so
        # existing runs reproduce exactly (see business_logic.compute_new_mrr).
        self.scale_aware_marketing = bool(
            self.initial_config.get("scale_aware_marketing", False)
        )
        # physics_v2 (F1). marketing_curve selects between three regimes:
        #   "legacy"      absolute-dollar Hill constants (the original engine)
        #   "scale_aware" customer/CAC-anchored curve, ASSUMED rate 0.20 - what
        #                 every recorded scale-aware run (incl. the v1 backtest)
        #                 used
        #   "v2"          same curve with the CAL-fitted saturation rate
        #                 (business_logic.SATURATION_ACQUISITION_RATE_V2)
        # None (default) preserves the old two-flag behaviour exactly, so every
        # recorded config keeps reproducing byte-identically.
        marketing_curve = self.initial_config.get("marketing_curve")
        if marketing_curve is not None:
            if marketing_curve not in {"legacy", "scale_aware", "v2"}:
                raise ValueError(f"unknown marketing_curve: {marketing_curve!r}")
            self.scale_aware_marketing = marketing_curve != "legacy"
        self.marketing_curve = marketing_curve
        # Explicit override wins; otherwise "v2" uses the fitted constant and
        # the other regimes use the module default (0.20) via None.
        rate = self.initial_config.get("saturation_acquisition_rate")
        if rate is None and marketing_curve == "v2":
            rate = business_logic.SATURATION_ACQUISITION_RATE_V2
            if rate is None:
                raise ValueError(
                    "marketing_curve='v2' requires the fitted "
                    "SATURATION_ACQUISITION_RATE_V2 (run the F1 fit) or an "
                    "explicit saturation_acquisition_rate override")
        self.saturation_acquisition_rate = None if rate is None else float(rate)
        # physics_v2 (D1). competitive_entry="scale_neutral" removes the $50k
        # market-attractiveness anchor from the random entry shock (see
        # business_logic.competitive_entry_shock). None/"legacy" reproduces
        # recorded runs byte-identically.
        competitive_entry = self.initial_config.get("competitive_entry")
        if competitive_entry not in {None, "legacy", "scale_neutral"}:
            raise ValueError(f"unknown competitive_entry: {competitive_entry!r}")
        self.competitive_entry_scale_neutral = competitive_entry == "scale_neutral"
        # physics_v2 (F2). Environment financing rule, parameters measured from
        # the EDGAR panel (see business_logic.FINANCING_* provenance). Off by
        # default: enabling it adds exactly one RNG draw per step, which would
        # change every recorded trajectory. Research-scale runs leave it off.
        self.financing_enabled = bool(self.initial_config.get("financing_enabled", False))
        self.financing_runway_threshold_months = float(self.initial_config.get(
            "financing_runway_threshold_months",
            business_logic.FINANCING_RUNWAY_THRESHOLD_MONTHS))
        self.financing_raise_multiple = float(self.initial_config.get(
            "financing_raise_multiple", business_logic.FINANCING_RAISE_MULTIPLE))
        self.financing_monthly_prob = float(self.initial_config.get(
            "financing_monthly_prob", business_logic.FINANCING_MONTHLY_PROB))
        self.financing_events: list[dict] = []
        # Re-estimate CAC only in a month that actually acquired a customer.
        # Defaults to scale_aware_marketing because that curve is what closes
        # the loop: marketing_curve_params places gamma from state.cac, so a
        # month with a fractional response writes an enormous CAC, the CAC
        # pushes gamma further right, and the next response is smaller still.
        # Measured on the rule-based arm of a 12-month projection, cac went
        # 1.4e17 -> 9.0e43 in a single step and overflowed the float32
        # observation. Under the absolute constants gamma never read state.cac,
        # so the loop had no path to close and research runs are untouched.
        self.stable_cac = bool(
            self.initial_config.get("stable_cac", self.scale_aware_marketing)
        )
        # Opt-in: R&D measured as a share of the company's own revenue, and
        # product quality given its own headroom instead of borrowing
        # innovation_factor's. Off by default - it changes what the engine
        # believes about R&D, not just its scale. See
        # business_logic.apply_innovation_investment.
        self.scale_aware_rnd = bool(
            self.initial_config.get("scale_aware_rnd", False)
        )
        # Opt-in gross margin on recognised revenue. None reproduces the
        # original behaviour exactly - revenue booked to cash at 100% margin,
        # no cost of revenue deducted - so every recorded result is untouched.
        # Product surfaces pass the calibrated figure; see calibration.gross_margin_pct.
        gross_margin = self.initial_config.get("gross_margin")
        self.gross_margin = None if gross_margin is None else float(gross_margin)
        # The hard shocks at months 24/48/72 are a fixture of the 120-month
        # research episode, not a property of the world. A founder projecting 12
        # months from month 20 would inherit one at month 4 of their forecast for
        # no reason they could see. Product surfaces turn them off; research runs
        # leave this True and are unaffected.
        self.scheduled_shocks = bool(self.initial_config.get("scheduled_shocks", True))
        # Seed-matched cross-policy comparison. Off by default because it changes
        # trajectories and would invalidate everything already in results/.
        #
        # Two things are wrong without it, and both break the premise that a
        # shared seed list means a shared world:
        #
        #   1. The physics draws from the global `random` module, so anything
        #      else in the process that draws - a policy sampling its own
        #      actions, for instance - shifts the environment's stream. Measured:
        #      identical seed and identical actions, but a policy consuming six
        #      numbers per step saw competitors go 5->9 where a policy consuming
        #      none saw 5->6.
        #   2. apply_recession_cascade draws only when unemployment > 8 AND
        #      interest_rate > 7, making per-step consumption state-dependent, so
        #      policies desynchronise once they reach different macro states.
        #
        # On, the environment owns a private generator nothing else can perturb,
        # and every step consumes a fixed number of draws from it. Turn this on
        # for any run whose conclusion rests on comparing policies at equal seeds
        # - which is every ablation.
        self.deterministic_rng = bool(self.initial_config.get("deterministic_rng", False))
        self._rng: random.Random | None = None
        
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
        self.episode_seed: int | None = None
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # gym's super().reset seeds self.np_random, which the physics never uses:
        # env/business_logic.py draws from the global `random` module throughout.
        # Without this line env.reset(seed=N) does not reproduce, and callers have
        # to know to seed the module themselves. simulation_runner.py already does
        # exactly this immediately after reset, so re-seeding here is a no-op for
        # every existing run and makes the documented contract true for new ones.
        #
        # This gives reproducibility, NOT isolation: the stream is still global and
        # its per-step draw count is state-dependent (apply_recession_cascade only
        # draws when unemployment > 8 and interest_rate > 7), so two policies that
        # reach different macro states will desynchronise. Comparisons that need a
        # shared shock tape must keep their horizon short enough to stay clear of
        # that, and say so.
        if seed is not None:
            random.seed(seed)
        # The private stream is seeded from the same episode seed, so a run is
        # reproducible from the seed alone. The global module is still seeded
        # above: policies and baselines that draw for themselves need to be
        # reproducible too, they just must not share a stream with the physics.
        self._rng = random.Random(seed) if self.deterministic_rng else None
        self.episode_seed = seed
        self.financing_events = []
        
        self.state = EnvState(
            mrr=float(self.initial_config.get("initial_mrr", 50_000)),
            cash=float(self.initial_config.get("initial_cash", sim_config.INITIAL_CASH)),
            cac=float(self.initial_config.get("cac", sim_config.BASE_CAC)),
            ltv=float(self.initial_config.get("ltv", 7_000)),
            churn_enterprise=float(self.initial_config.get("churn_enterprise", 0.01)),
            churn_smb=float(self.initial_config.get("churn_smb", 0.03)),
            churn_b2c=float(self.initial_config.get("churn_b2c", 0.05)),
            interest_rate=float(self.initial_config.get("interest_rate", 3.0)),
            consumer_confidence=float(self.initial_config.get("consumer_confidence", 100.0)),
            competitors=int(self.initial_config.get("competitors", 5)),
            product_quality=float(self.initial_config.get("product_quality", sim_config.INITIAL_PRODUCT_QUALITY)),
            price=float(self.initial_config.get("average_price", 50.0)),
            months_elapsed=0,
            headcount=int(self.initial_config.get("initial_headcount", 1)),
            monthly_burn=(
                None
                if self.initial_config.get("monthly_burn") is None
                else float(self.initial_config["monthly_burn"])
            ),
            valuation_multiple=float(self.initial_config.get("valuation_multiple", 10.0)),
            unemployment=float(self.initial_config.get("unemployment", 4.0)),
            innovation_factor=float(self.initial_config.get("innovation_factor", 1.0)),
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
        shock_label = "NO_SHOCK"

        business_logic.interest_rate_shock(self.state, rng=self._rng)
        business_logic.consumer_confidence_shock(self.state, rng=self._rng)
        business_logic.competitive_entry_shock(
            self.state, rng=self._rng,
            scale_neutral=self.competitive_entry_scale_neutral)

        if self.scheduled_shocks and self.state.months_elapsed in {24, 48, 72}:
            shock_cycle = ["competitor_surge", "rate_hike", "recession"]
            shock_type = shock_cycle[(self.episode_seed or 0) % len(shock_cycle)]
            shock_label = business_logic.inject_hard_shock(self.state, shock_type)

        business_logic.apply_recession_cascade(
            self.state, rng=self._rng, always_draw=self.deterministic_rng
        )
        business_logic.apply_hysteresis(self.state)

        business_logic.apply_recovery(self.state)

        new_mrr = business_logic.compute_new_mrr(
            self.state,
            action.marketing,
            scale_aware=self.scale_aware_marketing,
            rng=self._rng,
            saturation_rate=self.saturation_acquisition_rate,
        )

        expansion = business_logic.compute_expansion_mrr(
            self.state, action.product, scale_aware=self.scale_aware_rnd
        )

        business_logic.apply_innovation_investment(
            self.state, action.product, scale_aware=self.scale_aware_rnd
        )

        churn_rate = business_logic.compute_churn_rate(self.state)

        self.state.mrr = self.state.mrr * (1 - churn_rate) + new_mrr + expansion

        # Revenue lands in cash net of cost of revenue when a margin is configured.
        # gross_margin=None keeps the original 100%-margin behaviour.
        margin = 1.0 if self.gross_margin is None else self.gross_margin
        self.state.cash += self.state.mrr * margin

        business_logic.apply_pricing_effect(self.state, action.pricing, rng=self._rng)
        
        if action.hiring.hires > 0:
            max_hires = int((self.state.cash / 18.0) / action.hiring.cost_per_employee)
            if action.hiring.hires > max_hires:
                action.hiring.hires = max_hires
        
        one_time_hiring_cost = action.hiring.hires * action.hiring.cost_per_employee
        business_logic.apply_hiring_cost(self.state, action.hiring) 
        self.state.headcount += action.hiring.hires

        # A hire adds ongoing payroll, not just a one-time cost. Under the
        # headcount convention that happened for free, because burn was derived
        # from headcount; once burn is a real number it has to be added
        # explicitly, at the same salary slot the convention always used.
        if self.state.monthly_burn is not None and action.hiring.hires > 0:
            self.state.monthly_burn += (
                action.hiring.hires * business_logic.SALARY_SLOT_USD
            )

        salary_burn = business_logic.monthly_burn(self.state)
        
        total_spend = action.marketing.spend + action.product.r_and_d_spend
        
        self.state.cash -= (salary_burn + total_spend)
        
        if self.state.price > 0:
            estimated_new_users = new_mrr / self.state.price
            # A month that acquired a fraction of a customer has no
            # cost-per-customer: `spend / 1e-50` is a division artifact, not a
            # number about the business. compute_cac's other branch is no better
            # - it writes cac=0 for a month with no acquisition at all, which
            # makes marketing look free the month after. Either way the previous
            # estimate is the honest carry-forward, and it is what a founder
            # would say about a month in which nobody signed up.
            measurable = estimated_new_users >= 1.0 and action.marketing.spend > 0
            if measurable or not self.stable_cac:
                raw_cac = business_logic.compute_cac(
                    action.marketing.spend, estimated_new_users
                )
                self.state.cac = business_logic.scale_cac_by_macro(raw_cac, self.state)
        
        self.state.ltv = business_logic.compute_ltv(self.state.price, churn_rate)

        rule40_burn = one_time_hiring_cost + salary_burn + total_spend

        rule40 = business_logic.compute_rule_of_40(prev_mrr, self.state.mrr, rule40_burn)
        reward = business_logic.compute_reward(self.state, rule40)

        # physics_v2 (F2): environment financing rule, parameters measured from
        # the panel (business_logic.FINANCING_* provenance). The draw happens
        # UNCONDITIONALLY whenever the flag is on, so per-step draw count stays
        # fixed and matched-seed arms keep sharing a world (same reasoning as
        # apply_recession_cascade's always_draw). Runway is measured against
        # this month's realized net burn - the in-sim analogue of |OCF|/3 that
        # D4 measured. A month that drove cash negative is still eligible:
        # rescue rounds close while cash crosses zero, which is exactly what
        # the six all-seed bankruptcy companies did in reality.
        financing_raise = 0.0
        if self.financing_enabled:
            roll = (self._rng if self._rng is not None else random).random()
            net_burn = (salary_burn + total_spend + one_time_hiring_cost) \
                - self.state.mrr * margin
            if net_burn > 0:
                runway = self.state.cash / net_burn
                if (runway < self.financing_runway_threshold_months
                        and roll < self.financing_monthly_prob):
                    financing_raise = self.financing_raise_multiple * net_burn
                    self.state.cash += financing_raise
                    self.financing_events.append({
                        "month": self.state.months_elapsed,
                        "amount": financing_raise,
                        "net_burn": net_burn,
                        "runway_before": runway,
                    })

        self.state.months_elapsed += 1

        terminated = self.state.cash <= 0
        truncated = self.state.months_elapsed >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "rule_of_40": rule40,
            "state": self.state.model_dump(),
            "shock_label": shock_label,
            "financing_raise": financing_raise,
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
