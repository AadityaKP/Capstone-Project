"""E-battery (E1-E5) under shock_recovery="mean_revert" for legacy and v2
physics (Addendum A companion; no LLM). Reported; no new criteria.

Fresh boardroom no-oracle runs, 20 seeds x 120 months, deterministic_rng,
scheduled shocks on, same E1-E5 definitions and verdict rules as
validation/calibration/p4_e_battery_v2.py against the same EDGAR panel:
  legacy_mr: legacy physics + mean_revert, legacy-corridor proposal agents
  v2_mr:     v2 flags (fitted curve, scale-aware corridor, scale-neutral
             competitive entry, financing off) + mean_revert

Writes ONLY new files: environment_stats_legacy_mr.csv,
environment_stats_v2_mr.csv, and e_battery_mr_scorecard_rows.csv (candidate
rows with tests suffixed _mr). It does NOT touch environment_scorecard.csv -
appending scorecard rows is the morning step (S13; never edit existing rows).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter  # noqa: E402
from agents.proposal_agents import (CFOProposalAgent, CMOProposalAgent,  # noqa: E402
                                    CPOProposalAgent)
from boardroom.boardroom import Boardroom  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402

RESULTS = ROOT / "validation/results"
N_SEEDS = 20
MAX_MONTHS = 120
CONFIGS = {
    "legacy_mr": dict(
        config={"deterministic_rng": True, "shock_recovery": "mean_revert"},
        corridor="legacy", physics="legacy"),
    "v2_mr": dict(
        config={"deterministic_rng": True, "marketing_curve": "v2",
                "competitive_entry": "scale_neutral",
                "shock_recovery": "mean_revert"},
        corridor="scale_aware", physics="v2"),
}
TEST_METRIC = {"E1_mr": "qoq_growth", "E2_mr": "growth_lag1_autocorr",
               "E3_mr": "growth_vs_logscale_spearman",
               "E4_mr": "disc_spend_pct_revenue",
               "E5_mr": "growth_volatility_within"}
DIMENSION = {"E1_mr": "QoQ revenue growth distribution",
             "E2_mr": "growth persistence (lag-1 autocorr)",
             "E3_mr": "growth deceleration with scale",
             "E4_mr": "discretionary spend / revenue",
             "E5_mr": "within-unit growth volatility"}


def run_boardroom(seed: int, config: dict, corridor: str) -> pd.DataFrame:
    env = StartupEnv(initial_config=dict(config))
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    if corridor == "scale_aware":
        board = Boardroom(
            [CFOProposalAgent(corridor=corridor),
             CMOProposalAgent(corridor=corridor),
             CPOProposalAgent(corridor=corridor)],
            use_oracle=False, corridor=corridor)
    else:
        board = Boardroom(
            [CFOProposalAgent(), CMOProposalAgent(), CPOProposalAgent()],
            use_oracle=False)
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


def pcts(a):
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                p90=float(np.percentile(a, 90)), n=len(a))


def battery(tag: str, spec: dict, edgar_arrays: dict) -> tuple[pd.DataFrame, list[dict]]:
    sim = pd.concat([run_boardroom(s, spec["config"], spec["corridor"])
                     for s in range(N_SEEDS)], ignore_index=True)
    print(f"{tag}: {N_SEEDS} boardroom seeds done", flush=True)
    sim["quarter"] = sim.month // 3
    q = (sim.groupby(["seed", "quarter"])
            .agg(qrev=("mrr", "sum"), n=("mrr", "size"),
                 mkt=("mkt", "sum"), rnd=("rnd", "sum")).reset_index())
    q = q[q.n == 3].sort_values(["seed", "quarter"])
    q["prev"] = q.groupby("seed").qrev.shift(1)
    q["g"] = q.qrev / q.prev - 1.0
    q["spend_pct"] = (q.mkt + q.rnd) / q.qrev

    s_growth = q.g.dropna().to_numpy()
    s_arrays = {
        "qoq_growth": s_growth,
        "growth_lag1_autocorr": per_unit(q, "seed", "g", lag1),
        "growth_volatility_within": per_unit(q, "seed", "g",
                                             lambda s: s.std(ddof=1)),
        "growth_vs_logscale_spearman": np.array([v for v in (
            stats.spearmanr(np.log(g.qrev), g.g).statistic
            for _, g in q.dropna(subset=["g"]).groupby("seed") if len(g) >= 8)
            if np.isfinite(v)]),
        "disc_spend_pct_revenue": q.spend_pct.dropna().to_numpy(),
    }
    side = f"sim_boardroom_{tag}"
    stats_rows = []
    for metric in s_arrays:
        stats_rows.append(dict(metric=metric, side="EDGAR",
                               **pcts(edgar_arrays[metric])))
        stats_rows.append(dict(metric=metric, side=side, **pcts(s_arrays[metric])))
    ks = stats.ks_2samp(s_growth, edgar_arrays["qoq_growth"])
    w1 = stats.wasserstein_distance(s_growth, edgar_arrays["qoq_growth"])
    stats_df = pd.DataFrame(stats_rows)

    def get(metric, which):
        return stats_df[(stats_df.metric == metric) & (stats_df.side == which)].iloc[0]

    score_rows = []

    def add(test, verdict, result):
        score_rows.append(dict(
            dimension=DIMENSION[test], test=test,
            policy_arm=f"boardroom_{tag}",
            edgar_n=int(get(TEST_METRIC[test], "EDGAR").n),
            sim_n=int(get(TEST_METRIC[test], side).n),
            result=result, verdict=verdict,
            physics_version=spec["physics"], shock_recovery="mean_revert"))

    e, s = get("qoq_growth", "EDGAR"), get("qoq_growth", side)
    in_iqr = e.p25 <= s["median"] <= e.p75
    overlap = min(e.p75, s.p75) > max(e.p25, s.p25)
    in_p10p90 = e.p10 <= s["median"] <= e.p90
    add("E1_mr",
        "PASS" if (in_iqr and overlap) else ("PARTIAL" if in_p10p90 else "FAIL"),
        f"sim median {s['median']:.3f} vs EDGAR IQR [{e.p25:.3f},{e.p75:.3f}]; "
        f"KS={ks.statistic:.3f}, W1={w1:.4f}")

    e, s = get("growth_lag1_autocorr", "EDGAR"), get("growth_lag1_autocorr", side)
    same = np.sign(e["median"]) == np.sign(s["median"])
    add("E2_mr",
        "PASS" if same and abs(e["median"] - s["median"]) <= 0.25 else ("PARTIAL" if same else "FAIL"),
        f"median autocorr sim {s['median']:.3f} vs EDGAR {e['median']:.3f}")

    e, s = get("growth_vs_logscale_spearman", "EDGAR"), get("growth_vs_logscale_spearman", side)
    add("E3_mr",
        "PASS" if (e["median"] < 0 and s["median"] < 0) else
        ("PARTIAL" if e["median"] < 0 and abs(s["median"]) < 0.1 else "FAIL"),
        f"median within-unit Spearman sim {s['median']:.3f} vs EDGAR {e['median']:.3f}")

    e, s = get("disc_spend_pct_revenue", "EDGAR"), get("disc_spend_pct_revenue", side)
    add("E4_mr",
        "PASS" if e.p10 <= s["median"] <= e.p90 else "FAIL",
        f"sim median discretionary spend {s['median']:.1%} of revenue vs "
        f"EDGAR p10-p90 [{e.p10:.1%},{e.p90:.1%}] (S&M+R&D)")

    e, s = get("growth_volatility_within", "EDGAR"), get("growth_volatility_within", side)
    ratio = s["median"] / e["median"] if e["median"] else np.inf
    add("E5_mr",
        "PASS" if 0.5 <= ratio <= 2 else ("PARTIAL" if 0.25 <= ratio <= 4 else "FAIL"),
        f"within-unit growth std sim {s['median']:.3f} vs EDGAR {e['median']:.3f} (x{ratio:.1f})")

    return stats_df, score_rows


def main() -> None:
    edgar = pd.read_csv(ROOT / "data/edgar_ratios.csv").dropna(subset=["qoq_growth"])
    e_spend = edgar.dropna(subset=["sm_pct_revenue", "rnd_pct_revenue"])
    edgar_arrays = {
        "qoq_growth": edgar.qoq_growth.to_numpy(),
        "growth_lag1_autocorr": per_unit(edgar, "ticker", "qoq_growth", lag1),
        "growth_volatility_within": per_unit(edgar, "ticker", "qoq_growth",
                                             lambda s: s.std(ddof=1)),
        "growth_vs_logscale_spearman": np.array([v for v in (
            stats.spearmanr(
                np.log(g.dropna(subset=["qoq_growth", "revenue"]).revenue),
                g.dropna(subset=["qoq_growth", "revenue"]).qoq_growth).statistic
            for _, g in edgar.groupby("ticker")
            if len(g.dropna(subset=["qoq_growth", "revenue"])) >= 8)
            if np.isfinite(v)]),
        "disc_spend_pct_revenue": (e_spend.sm_pct_revenue
                                   + e_spend.rnd_pct_revenue).to_numpy(),
    }

    all_scores = []
    for tag, spec in CONFIGS.items():
        stats_df, score_rows = battery(tag, spec, edgar_arrays)
        stats_df.to_csv(RESULTS / f"environment_stats_{tag}.csv", index=False)
        all_scores.extend(score_rows)
        print(pd.DataFrame(score_rows)[["test", "policy_arm", "verdict", "result"]]
              .to_string(index=False))
    pd.DataFrame(all_scores).to_csv(
        RESULTS / "e_battery_mr_scorecard_rows.csv", index=False)


if __name__ == "__main__":
    main()
