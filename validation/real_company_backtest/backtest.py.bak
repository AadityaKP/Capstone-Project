"""C1: real-company counterfactual backtest from EDGAR states.

For each panel company: earliest complete quarter (revenue, S&M, R&D, GM, cash)
with 4 consecutive subsequent revenue quarters. Map to EnvState, roll 12 months
x 30 seeds under 4 arms, compare with the actually observed next-4-quarter
revenue growth.

Mapping (assumed values labelled; see mapped_states.csv):
  mrr           = quarterly revenue / 3                       (observed)
  cash          = cash_and_investments                        (observed)
  monthly_burn  = G&A / 3  (fixed opex; S&M and R&D travel as actions) (observed)
  gross_margin  = company's own quarterly GM                  (observed)
  price         = $250/mo ARPA (sensitivity: $50)             (ASSUMED)
  churn (all segments) = ChartMogul monthly logo churn median for the price band
                  ($100-250: 2.70%; $25-100: 3.40%)           (benchmark, ASSUMED band)
  cac           = price/churn/3 (i.e. LTV:CAC = 3)            (ASSUMED)
  product_quality = 0.5, innovation_factor = 1.0, macro at defaults (ASSUMED)

Physics: scale-aware marketing + R&D, company gross margin, real monthly burn,
scheduled research shocks OFF, deterministic_rng ON. The backtest therefore
validates the scale-aware configuration; the legacy absolute constants are
meaningless at this scale (see edgar_data_audit.md section 4).

Arms:
  hold       company's own S&M/3 and R&D/3 held constant  (retrodiction arm)
  noop       no discretionary spend
  heuristic  CFO/CMO/CPO rule agents, dollar tiers scaled by mrr/50k
  boardroom  full boardroom (no oracle), scale_absolutes = mrr/50k

Decomposition reported per company:
  retrodiction_error = sim hold 4q growth (median over seeds) - actual 4q growth
  agent_increment    = per-seed (boardroom - hold) 4q growth, summarized

This is model-based counterfactual evidence. It never claims any agent would
have outperformed any real company.
"""
from __future__ import annotations

import sqlite3
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from boardroom.boardroom import Boardroom
from env.schemas import EnvState
from env.startup_env import StartupEnv

OUT = ROOT / "validation/real_company_backtest"
RESULTS = ROOT / "validation/results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

N_SEEDS = 30
HORIZON = 12
CHURN_BY_PRICE = {250.0: 0.027, 50.0: 0.034}   # ChartMogul monthly logo churn medians


def qidx(fp: str) -> int:
    y, q = fp.split("Q")
    return int(y) * 4 + int(q)


def load_panel():
    ratios = pd.read_csv(ROOT / "data/edgar_ratios.csv")
    con = sqlite3.connect(ROOT / "data/edgar.db")
    ga = pd.read_sql_query(
        "SELECT cik, fiscal_period, value AS ga FROM facts WHERE concept='ga_expense'", con)
    con.close()
    df = ratios.merge(ga, on=["cik", "fiscal_period"], how="left")
    df["qi"] = df.fiscal_period.map(qidx)
    return df.sort_values(["ticker", "qi"])


def pick_init(df_co):
    """Earliest quarter with complete core + 4 consecutive later revenue quarters."""
    have_rev = {r.qi: r.revenue for r in df_co.itertuples() if pd.notna(r.revenue)}
    for r in df_co.itertuples():
        if any(pd.isna(v) for v in (r.revenue, r.sm_pct_revenue, r.rnd_pct_revenue,
                                    r.gross_margin, r.cash_and_investments, r.ga)):
            continue
        future = [have_rev.get(r.qi + k) for k in range(1, 5)]
        if all(v is not None for v in future):
            return r, future
    return None, None


def build_state(row, price):
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


def make_env(gm):
    return StartupEnv(initial_config={
        "max_months": 10_000, "scheduled_shocks": False,
        "scale_aware_marketing": True, "scale_aware_rnd": True,
        "gross_margin": float(gm), "deterministic_rng": True})


def hold_bundle(row):
    return {"marketing": {"spend": row.revenue * row.sm_pct_revenue / 3.0, "channel": "ppc"},
            "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
            "product": {"r_and_d_spend": row.revenue * row.rnd_pct_revenue / 3.0},
            "pricing": {"price_change_pct": 0.0}}


NOOP = {"marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0}}


def rollout(row, state, arm, seed, scale):
    env = make_env(row.gross_margin)
    env.reset(seed=seed)
    env.state = state.model_copy(deep=True)
    board = None
    if arm == "boardroom":
        board = Boardroom(
            [CFOProposalAgent(scale=scale), CMOProposalAgent(scale=scale),
             CPOProposalAgent(scale=scale)],
            use_oracle=False, scale_absolutes=scale)
        board.start_episode(seed)
    mrr_path = []
    for _ in range(HORIZON):
        if arm == "hold":
            action = deepcopy(hold_bundle(row))
        elif arm == "noop":
            action = deepcopy(NOOP)
        elif arm == "heuristic":
            action = {}
            for agent in (CFOAgent(scale=scale), CMOAgent(scale=scale), CPOAgent(scale=scale)):
                action.update(agent.act(env.state))
        else:
            action = board.decide(env.state)
        _, _, terminated, _, _ = env.step(ActionAdapter.translate_action(action))
        mrr_path.append(env.state.mrr)
        if terminated:
            break
    if len(mrr_path) < HORIZON:
        return None  # died: growth over 4 quarters undefined; recorded as death
    q4_rev = sum(mrr_path[9:12])
    return q4_rev / (state.mrr * 3.0) - 1.0


def main():
    panel = load_panel()
    mapped_rows, results = [], []
    for ticker, df_co in panel.groupby("ticker"):
        row, future = pick_init(df_co)
        if row is None:
            mapped_rows.append(dict(ticker=ticker, status="no eligible init quarter"))
            continue
        actual_growth = future[3] / row.revenue - 1.0
        for price in (250.0, 50.0):
            state = build_state(row, price)
            scale = state.mrr / 50_000.0
            mapped_rows.append(dict(
                ticker=ticker, status="ok", init_quarter=row.fiscal_period,
                price_assumed=price, mrr=state.mrr, cash=state.cash,
                monthly_burn=state.monthly_burn, gross_margin=row.gross_margin,
                sm_monthly=row.revenue * row.sm_pct_revenue / 3.0,
                rnd_monthly=row.revenue * row.rnd_pct_revenue / 3.0,
                churn_assumed=CHURN_BY_PRICE[price], cac_assumed=state.cac,
                actual_4q_growth=actual_growth))
            arm_growths = {}
            deaths = {}
            for arm in ("hold", "noop", "heuristic", "boardroom"):
                vals = []
                died = 0
                for seed in range(N_SEEDS):
                    g = rollout(row, state, arm, seed, scale)
                    if g is None:
                        died += 1
                    else:
                        vals.append(g)
                arm_growths[arm] = vals
                deaths[arm] = died
            hold_v = np.array(arm_growths["hold"])
            board_v = np.array(arm_growths["boardroom"])
            n_pair = min(len(hold_v), len(board_v))
            results.append(dict(
                ticker=ticker, init_quarter=row.fiscal_period, price_assumed=price,
                actual_4q_growth=actual_growth,
                sim_hold_median=float(np.median(hold_v)) if len(hold_v) else np.nan,
                sim_noop_median=float(np.median(arm_growths["noop"])) if arm_growths["noop"] else np.nan,
                sim_heuristic_median=float(np.median(arm_growths["heuristic"])) if arm_growths["heuristic"] else np.nan,
                sim_boardroom_median=float(np.median(board_v)) if len(board_v) else np.nan,
                deaths_hold=deaths["hold"], deaths_boardroom=deaths["boardroom"],
                retrodiction_error=(float(np.median(hold_v)) - actual_growth) if len(hold_v) else np.nan,
                agent_increment_median=(float(np.median(board_v[:n_pair] - hold_v[:n_pair]))
                                        if n_pair else np.nan)))
            print(f"{ticker} {row.fiscal_period} price={price:.0f}: actual={actual_growth:+.1%} "
                  f"hold={results[-1]['sim_hold_median']:+.1%} "
                  f"boardroom={results[-1]['sim_boardroom_median']:+.1%} "
                  f"err={results[-1]['retrodiction_error']:+.1%}")

    pd.DataFrame(mapped_rows).to_csv(OUT / "mapped_states.csv", index=False)
    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "real_company_backtest.csv", index=False)

    for price in (250.0, 50.0):
        sub = res[res.price_assumed == price].dropna(subset=["retrodiction_error"])
        mae = sub.retrodiction_error.abs().median()
        sign_ok = (np.sign(sub.sim_hold_median) == np.sign(sub.actual_4q_growth)).mean()
        inc_pos = (sub.agent_increment_median > 0).mean()
        print(f"\nprice={price:.0f}: n={len(sub)} | median |retrodiction error| = {mae:.1%} "
              f"| growth-sign agreement = {sign_ok:.0%} | boardroom increment > 0 in {inc_pos:.0%} of companies")


if __name__ == "__main__":
    main()
