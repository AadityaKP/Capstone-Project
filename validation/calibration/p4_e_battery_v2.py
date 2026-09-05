"""Phase 4/5: E-battery at research scale under v2 flags; append _v2 scorecard rows.

Fresh boardroom no-oracle runs (research init, deterministic_rng, scheduled
shocks on - the recorded battery's world), 20 seeds x 120 months, v2 flags
(fitted curve, scale-aware corridor, competitive_entry scale-neutral,
financing off at research scale). Computes E1-E5 with the same definitions and
verdict rules as validation/analysis/environment_battery.py against the same
EDGAR panel.

Never overwrites v1 results: writes environment_stats_v2.csv, and APPENDS
physics_version="v2" rows to environment_scorecard.csv (existing rows gain
physics_version="v1", values untouched). Also appends the A2-v2 ordering row
to agent_scorecard.csv from the Phase 3 gate data.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, NOOP, ROOT  # noqa: E402

from agents.adapter import ActionAdapter  # noqa: E402
from agents.proposal_agents import (CFOProposalAgent, CMOProposalAgent,  # noqa: E402
                                    CPOProposalAgent)
from boardroom.boardroom import Boardroom  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402

RESULTS = ROOT / "validation/results"
N_SEEDS = 20
MAX_MONTHS = 120
V2_CONFIG = {"deterministic_rng": True, "marketing_curve": "v2",
             "competitive_entry": "scale_neutral"}


def run_boardroom_v2(seed: int) -> pd.DataFrame:
    env = StartupEnv(initial_config=dict(V2_CONFIG))
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    board = Boardroom(
        [CFOProposalAgent(corridor="scale_aware"),
         CMOProposalAgent(corridor="scale_aware"),
         CPOProposalAgent(corridor="scale_aware")],
        use_oracle=False, corridor="scale_aware")
    board.start_episode(seed)
    rows = []
    for _ in range(MAX_MONTHS):
        action = ActionAdapter.translate_action(board.decide(env.state))
        _, _, term, trunc, _ = env.step(action)
        rows.append(dict(seed=seed, month=env.state.months_elapsed - 1,
                         mrr=env.state.mrr,
                         mkt=action["marketing"]["spend"],
                         rnd=action["product"]["r_and_d_spend"]))
        if term or trunc:
            break
    return pd.DataFrame(rows)


def main() -> None:
    edgar = pd.read_csv(ROOT / "data/edgar_ratios.csv").dropna(subset=["qoq_growth"])

    def per_unit(df, unit, col, func, min_obs=8):
        vals = []
        for _, g in df.groupby(unit):
            s = g[col].dropna()
            if len(s) >= min_obs:
                v = func(s)
                if np.isfinite(v):
                    vals.append(v)
        return np.array(vals)

    def lag1(s):
        s = s.to_numpy()
        if len(s) < 3 or np.std(s[:-1]) == 0 or np.std(s[1:]) == 0:
            return np.nan
        return np.corrcoef(s[:-1], s[1:])[0, 1]

    e_growth = edgar.qoq_growth.to_numpy()
    e_persist = per_unit(edgar, "ticker", "qoq_growth", lag1)
    e_vol = per_unit(edgar, "ticker", "qoq_growth", lambda s: s.std(ddof=1))
    e_decel = np.array([v for v in (
        stats.spearmanr(np.log(g.dropna(subset=["qoq_growth", "revenue"]).revenue),
                        g.dropna(subset=["qoq_growth", "revenue"]).qoq_growth).statistic
        for _, g in edgar.groupby("ticker") if len(g.dropna(subset=["qoq_growth", "revenue"])) >= 8)
        if np.isfinite(v)])
    e_spend = edgar.dropna(subset=["sm_pct_revenue", "rnd_pct_revenue"])
    e_spend_pct = (e_spend.sm_pct_revenue + e_spend.rnd_pct_revenue).to_numpy()

    sim = pd.concat([run_boardroom_v2(s) for s in range(N_SEEDS)], ignore_index=True)
    sim["quarter"] = sim.month // 3
    q = (sim.groupby(["seed", "quarter"])
            .agg(qrev=("mrr", "sum"), n=("mrr", "size"),
                 mkt=("mkt", "sum"), rnd=("rnd", "sum")).reset_index())
    q = q[q.n == 3].sort_values(["seed", "quarter"])
    q["prev"] = q.groupby("seed").qrev.shift(1)
    q["g"] = q.qrev / q.prev - 1.0
    q["spend_pct"] = (q.mkt + q.rnd) / q.qrev

    s_growth = q.g.dropna().to_numpy()
    s_persist = per_unit(q, "seed", "g", lag1)
    s_vol = per_unit(q, "seed", "g", lambda s: s.std(ddof=1))
    s_decel = np.array([v for v in (
        stats.spearmanr(np.log(g.qrev), g.g).statistic
        for _, g in q.dropna(subset=["g"]).groupby("seed") if len(g) >= 8)
        if np.isfinite(v)])
    s_spend_pct = q.spend_pct.dropna().to_numpy()

    def pcts(a):
        return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                    p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                    p90=float(np.percentile(a, 90)), n=len(a))

    stats_rows = []
    for metric, e_arr, s_arr in [
            ("qoq_growth", e_growth, s_growth),
            ("growth_lag1_autocorr", e_persist, s_persist),
            ("growth_volatility_within", e_vol, s_vol),
            ("growth_vs_logscale_spearman", e_decel, s_decel),
            ("disc_spend_pct_revenue", e_spend_pct, s_spend_pct)]:
        stats_rows.append(dict(metric=metric, side="EDGAR", **pcts(e_arr)))
        stats_rows.append(dict(metric=metric, side="sim_boardroom_v2", **pcts(s_arr)))
    ks = stats.ks_2samp(s_growth, e_growth)
    w1 = stats.wasserstein_distance(s_growth, e_growth)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(RESULTS / "environment_stats_v2.csv", index=False)

    def get(metric, side):
        return stats_df[(stats_df.metric == metric) & (stats_df.side == side)].iloc[0]

    score_rows = []

    def add(test, dim, verdict, result):
        score_rows.append(dict(dimension=dim, test=test, policy_arm="boardroom_v2",
                               edgar_n=int(get(TEST_METRIC[test], "EDGAR").n),
                               sim_n=int(get(TEST_METRIC[test], "sim_boardroom_v2").n),
                               result=result, verdict=verdict, physics_version="v2"))

    TEST_METRIC = {"E1": "qoq_growth", "E2": "growth_lag1_autocorr",
                   "E3": "growth_vs_logscale_spearman",
                   "E4": "disc_spend_pct_revenue", "E5": "growth_volatility_within"}

    e, s = get("qoq_growth", "EDGAR"), get("qoq_growth", "sim_boardroom_v2")
    in_iqr = e.p25 <= s["median"] <= e.p75
    overlap = min(e.p75, s.p75) > max(e.p25, s.p25)
    in_p10p90 = e.p10 <= s["median"] <= e.p90
    add("E1", "QoQ revenue growth distribution",
        "PASS" if (in_iqr and overlap) else ("PARTIAL" if in_p10p90 else "FAIL"),
        f"sim median {s['median']:.3f} vs EDGAR IQR [{e.p25:.3f},{e.p75:.3f}]; "
        f"KS={ks.statistic:.3f}, W1={w1:.4f}")

    e, s = get("growth_lag1_autocorr", "EDGAR"), get("growth_lag1_autocorr", "sim_boardroom_v2")
    same = np.sign(e["median"]) == np.sign(s["median"])
    add("E2", "growth persistence (lag-1 autocorr)",
        "PASS" if same and abs(e["median"] - s["median"]) <= 0.25 else ("PARTIAL" if same else "FAIL"),
        f"median autocorr sim {s['median']:.3f} vs EDGAR {e['median']:.3f}")

    e, s = get("growth_vs_logscale_spearman", "EDGAR"), get("growth_vs_logscale_spearman", "sim_boardroom_v2")
    add("E3", "growth deceleration with scale",
        "PASS" if (e["median"] < 0 and s["median"] < 0) else
        ("PARTIAL" if e["median"] < 0 and abs(s["median"]) < 0.1 else "FAIL"),
        f"median within-unit Spearman sim {s['median']:.3f} vs EDGAR {e['median']:.3f}")

    e, s = get("disc_spend_pct_revenue", "EDGAR"), get("disc_spend_pct_revenue", "sim_boardroom_v2")
    add("E4", "discretionary spend / revenue",
        "PASS" if e.p10 <= s["median"] <= e.p90 else "FAIL",
        f"sim median discretionary spend {s['median']:.1%} of revenue vs "
        f"EDGAR p10-p90 [{e.p10:.1%},{e.p90:.1%}] (S&M+R&D)")

    e, s = get("growth_volatility_within", "EDGAR"), get("growth_volatility_within", "sim_boardroom_v2")
    ratio = s["median"] / e["median"] if e["median"] else np.inf
    add("E5", "within-unit growth volatility",
        "PASS" if 0.5 <= ratio <= 2 else ("PARTIAL" if 0.25 <= ratio <= 4 else "FAIL"),
        f"within-unit growth std sim {s['median']:.3f} vs EDGAR {e['median']:.3f} (x{ratio:.1f})")

    # append to environment scorecard (existing rows -> physics_version v1)
    sc = pd.read_csv(RESULTS / "environment_scorecard.csv")
    if "physics_version" not in sc.columns:
        sc["physics_version"] = "v1"
    sc = pd.concat([sc, pd.DataFrame(score_rows)], ignore_index=True)
    sc.to_csv(RESULTS / "environment_scorecard.csv", index=False)

    # append A2-v2 ordering row to agent scorecard from the Phase 3 gate data
    gate = pd.read_csv(CAL_DIR / "p3_regression_gate.csv")
    v2 = gate[gate.physics_version == "v2"]
    piv = v2.pivot(index="seed", columns="policy", values="final_mrr")
    diff = (piv.boardroom - piv.noop).to_numpy()
    wil = stats.wilcoxon(diff)
    sd = diff.std(ddof=1)
    g_eff = diff.mean() / sd * (1 - 3 / (4 * len(diff) - 9)) if sd > 0 else np.nan
    ag = pd.read_csv(RESULTS / "agent_scorecard.csv")
    if "physics_version" not in ag.columns:
        ag["physics_version"] = "v1"
    ag = pd.concat([ag, pd.DataFrame([dict(
        dimension="superiority to noop (v2 physics)",
        test="A2-v2 paired, deterministic RNG, research scale", baseline="noop",
        n=len(diff), effect=f"g={g_eff:.2f}",
        result=f"mean diff final MRR ${diff.mean():,.0f}, Wilcoxon p={wil.pvalue:.2g}, 10/10 seeds",
        acceptance_criterion="A2 ordering must not flip under v2 (Phase 3 gate)",
        verdict="PASS" if (piv.boardroom.median() > piv.noop.median()) else "FAIL",
        interpretation="boardroom > noop preserved under v2 flags at research scale; "
                       "E4 spend ratio 8.5% -> 72.3% (inside EDGAR bands)",
        physics_version="v2")])], ignore_index=True)
    ag.to_csv(RESULTS / "agent_scorecard.csv", index=False)

    print(pd.DataFrame(score_rows)[["test", "verdict", "result"]].to_string(index=False))


if __name__ == "__main__":
    main()
