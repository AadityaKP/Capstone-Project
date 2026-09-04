"""Shared library for the physics_v2 calibration pipeline.

Reimplements the v1 backtest mapping (validation/real_company_backtest/
backtest.py) as importable functions, and adds two things v1 did not have:

  1. `extra_env_config` — arbitrary StartupEnv config overrides, so v2 flags
     (marketing_curve, financing_enabled, corridor) can be exercised without
     touching the v1 script.
  2. Diagnostic patches (Phase 1 only) — churn overrides, zeroed acquisition,
     zeroed expansion, saturation-rate overrides. Every patch preserves the
     deterministic RNG draw count: patched functions still call the original
     (consuming identical draws) and then discard/replace the result, so two
     runs differing only in a patch experience the identical shock tape.

This module changes NO physics. It only calls the engine.
"""
from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter                      # noqa: E402
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent  # noqa: E402
from agents.proposal_agents import (                          # noqa: E402
    CFOProposalAgent, CMOProposalAgent, CPOProposalAgent,
)
from boardroom.boardroom import Boardroom                     # noqa: E402
from env import business_logic                                # noqa: E402
from env.schemas import EnvState                              # noqa: E402
from env.startup_env import StartupEnv                        # noqa: E402

CAL_DIR = ROOT / "validation/calibration"
HORIZON = 12
CHURN_BY_PRICE = {250.0: 0.027, 50.0: 0.034}  # ChartMogul monthly logo churn medians
PRIMARY_PRICE = 250.0
V1_ALL_SEED_HOLD_DEATHS = ["ASAN", "CRWD", "DOMO", "ESTC", "RPD", "TENB"]


# ---------------------------------------------------------------- panel + mapping

def qidx(fp: str) -> int:
    y, q = fp.split("Q")
    return int(y) * 4 + int(q)


def load_panel() -> pd.DataFrame:
    ratios = pd.read_csv(ROOT / "data/edgar_ratios.csv")
    con = sqlite3.connect(ROOT / "data/edgar.db")
    ga = pd.read_sql_query(
        "SELECT cik, fiscal_period, value AS ga FROM facts WHERE concept='ga_expense'", con)
    con.close()
    df = ratios.merge(ga, on=["cik", "fiscal_period"], how="left")
    df["qi"] = df.fiscal_period.map(qidx)
    return df.sort_values(["ticker", "qi"])


def pick_init(df_co: pd.DataFrame):
    """Earliest quarter with complete core + 4 consecutive later revenue quarters.

    Identical to v1 backtest.py::pick_init.
    """
    have_rev = {r.qi: r.revenue for r in df_co.itertuples() if pd.notna(r.revenue)}
    for r in df_co.itertuples():
        if any(pd.isna(v) for v in (r.revenue, r.sm_pct_revenue, r.rnd_pct_revenue,
                                    r.gross_margin, r.cash_and_investments, r.ga)):
            continue
        future = [have_rev.get(r.qi + k) for k in range(1, 5)]
        if all(v is not None for v in future):
            return r, future
    return None, None


def company_table(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per company: init quarter, init revenue, actual 4q growth."""
    panel = load_panel() if panel is None else panel
    rows = []
    for ticker, df_co in panel.groupby("ticker"):
        row, future = pick_init(df_co)
        if row is None:
            continue
        rows.append(dict(
            ticker=ticker, init_quarter=row.fiscal_period,
            init_revenue=row.revenue, init_cash=row.cash_and_investments,
            gross_margin=row.gross_margin, ga=row.ga,
            sm_pct_revenue=row.sm_pct_revenue, rnd_pct_revenue=row.rnd_pct_revenue,
            actual_4q_growth=future[3] / row.revenue - 1.0))
    return pd.DataFrame(rows)


def build_state(row, price: float = PRIMARY_PRICE) -> EnvState:
    """Identical to v1 backtest.py::build_state."""
    mrr = row.revenue / 3.0
    churn = CHURN_BY_PRICE[price]
    return EnvState(
        mrr=mrr, cash=float(row.cash_and_investments),
        cac=price / churn / 3.0, ltv=price / churn,
        churn_enterprise=churn, churn_smb=churn, churn_b2c=churn,
        interest_rate=3.0, consumer_confidence=100.0, competitors=5,
        product_quality=0.5, price=price, months_elapsed=0,
        headcount=1, monthly_burn=float(row.ga) / 3.0,
        valuation_multiple=10.0, unemployment=4.0, innovation_factor=1.0,
        months_in_depression=0)


def make_env(gm: float, extra_env_config: dict | None = None) -> StartupEnv:
    config = {
        "max_months": 10_000, "scheduled_shocks": False,
        "scale_aware_marketing": True, "scale_aware_rnd": True,
        "gross_margin": float(gm), "deterministic_rng": True,
    }
    if extra_env_config:
        config.update(extra_env_config)
    return StartupEnv(initial_config=config)


def hold_bundle(row) -> dict:
    return {"marketing": {"spend": row.revenue * row.sm_pct_revenue / 3.0, "channel": "ppc"},
            "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
            "product": {"r_and_d_spend": row.revenue * row.rnd_pct_revenue / 3.0},
            "pricing": {"price_change_pct": 0.0}}


NOOP = {"marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0}}


# ---------------------------------------------------------------- diagnostic patches

@contextmanager
def physics_patches(churn_override: float | None = None,
                    zero_acquisition: bool = False,
                    zero_expansion: bool = False,
                    saturation_rate: float | None = None):
    """Temporarily patch business_logic for diagnostics (Phase 1 / fit only).

    RNG-alignment invariant: each patched function calls the ORIGINAL first so
    it consumes exactly the draws the unpatched physics would, then replaces
    the return value. Runs differing only in patches share the world.
    """
    orig_churn = business_logic.compute_churn_rate
    orig_new_mrr = business_logic.compute_new_mrr
    orig_expansion = business_logic.compute_expansion_mrr
    orig_rate = business_logic.SATURATION_ACQUISITION_RATE
    try:
        if churn_override is not None:
            def churn_patched(state, _orig=orig_churn, _v=churn_override):
                _orig(state)  # consumes nothing today; kept for safety
                return _v
            business_logic.compute_churn_rate = churn_patched
        if zero_acquisition:
            def new_mrr_patched(state, action, scale_aware=False, rng=None,
                                _orig=orig_new_mrr):
                _orig(state, action, scale_aware=scale_aware, rng=rng)
                return 0.0
            business_logic.compute_new_mrr = new_mrr_patched
        if zero_expansion:
            def expansion_patched(state, action, scale_aware=False,
                                  _orig=orig_expansion):
                _orig(state, action, scale_aware=scale_aware)
                return 0.0
            business_logic.compute_expansion_mrr = expansion_patched
        if saturation_rate is not None:
            business_logic.SATURATION_ACQUISITION_RATE = float(saturation_rate)
        yield
    finally:
        business_logic.compute_churn_rate = orig_churn
        business_logic.compute_new_mrr = orig_new_mrr
        business_logic.compute_expansion_mrr = orig_expansion
        business_logic.SATURATION_ACQUISITION_RATE = orig_rate


# ---------------------------------------------------------------- rollout

def rollout(row, state: EnvState, arm: str, seed: int, scale: float,
            extra_env_config: dict | None = None,
            boardroom_kwargs: dict | None = None,
            board=None,
            collect_trace: bool = False):
    """One 12-month episode. Returns dict with growth (None if died) and trace.

    `board` lets a caller pass a pre-built Boardroom (e.g. an oracle arm whose
    memory persists across seeds the way research runs do); otherwise the
    boardroom arm builds the no-oracle default used by v1.
    """
    env = make_env(row.gross_margin, extra_env_config)
    env.reset(seed=seed)
    env.state = state.model_copy(deep=True)
    if arm == "boardroom" and board is None:
        kwargs = dict(use_oracle=False, scale_absolutes=scale)
        if boardroom_kwargs:
            kwargs.update(boardroom_kwargs)
        board = Boardroom(
            [CFOProposalAgent(scale=scale), CMOProposalAgent(scale=scale),
             CPOProposalAgent(scale=scale)], **kwargs)
        board.start_episode(seed)
    heuristic_agents = None
    if arm == "heuristic":
        heuristic_agents = (CFOAgent(scale=scale), CMOAgent(scale=scale),
                            CPOAgent(scale=scale))

    mrr_path, trace = [], []
    for month in range(HORIZON):
        if arm == "hold":
            action = deepcopy(hold_bundle(row))
        elif arm == "noop":
            action = deepcopy(NOOP)
        elif arm == "heuristic":
            action = {}
            for agent in heuristic_agents:
                action.update(agent.act(env.state))
        else:
            action = board.decide(env.state)
        clean = ActionAdapter.translate_action(action)
        _, _, terminated, _, info = env.step(clean)
        mrr_path.append(env.state.mrr)
        if collect_trace:
            trace.append(dict(
                month=month, mrr=env.state.mrr, cash=env.state.cash,
                mkt_spend=clean["marketing"]["spend"],
                rd_spend=clean["product"]["r_and_d_spend"],
                financing_raise=info.get("financing_raise", 0.0)))
        if terminated:
            break
    growth = None
    if len(mrr_path) >= HORIZON:
        growth = sum(mrr_path[9:12]) / (state.mrr * 3.0) - 1.0
    return dict(growth=growth, died=len(mrr_path) < HORIZON,
                months_survived=len(mrr_path), trace=trace)


def run_company_arm(row, arm: str, seeds, price: float = PRIMARY_PRICE,
                    extra_env_config: dict | None = None,
                    boardroom_kwargs: dict | None = None,
                    patches: dict | None = None) -> dict:
    """Run one arm over seeds for one company. Returns growths + death count."""
    state = build_state(row, price)
    scale = state.mrr / 50_000.0
    growths, deaths = [], 0
    with physics_patches(**(patches or {})):
        for seed in seeds:
            out = rollout(row, state, arm, seed, scale,
                          extra_env_config=extra_env_config,
                          boardroom_kwargs=boardroom_kwargs)
            if out["died"]:
                deaths += 1
            else:
                growths.append(out["growth"])
    return dict(ticker=row.ticker, arm=arm, price=price,
                growths=growths, deaths=deaths, n_seeds=len(list(seeds)),
                median_growth=float(np.median(growths)) if growths else np.nan)


def load_split() -> pd.DataFrame:
    return pd.read_csv(CAL_DIR / "panel_split.csv")


def cal_tickers() -> list[str]:
    s = load_split()
    return sorted(s[s.split == "CAL"].ticker)


def holdout_tickers() -> list[str]:
    s = load_split()
    return sorted(s[s.split == "HOLDOUT"].ticker)
