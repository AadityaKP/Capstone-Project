"""A8: post-shock Rule-of-40 recovery by policy (COMPARATIVE).

"Recovery" here is RULE-OF-40 recovery - the month the episode's Rule of 40
first regains its pre-shock level (the value one month before the scheduled
shock), the same definition the recorded thesis metrics use. It is NOT
revenue-peak recovery, which essentially does not occur in this simulator
(see E6, validation/results/e6_drawdown_recovery.csv).

Sources (all deterministic_rng runs):
  A2 arms  validation/agents/decision_log.csv          (noop/random/heuristic/boardroom, 50 seeds)
  A3 arms  validation/results/a3/monthly_*.csv         (boardroom + oracle arms, 20 seeds)

Scheduled shocks land at months 24/48/72. For each (policy, seed, shock):
  pre-shock R40  = rule_of_40 at shock_month - 1
  event-time     = mean R40 at relative months -6..+24 (dead episodes drop out)
  recovery       = first month in (shock, shock+24] with R40 >= pre-shock value;
                   censored at +24 or at episode death

Writes:
  validation/results/a8_shock_r40_curves.csv   policy x rel_month mean/CI (episode as unit)
  validation/results/a8_shock_recovery.csv     policy x shock_month + overall recovery stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "validation/results"
SHOCKS = [24, 48, 72]
REL = range(-6, 25)
WINDOW = 24

frames = []
a2 = pd.read_csv(ROOT / "validation/agents/decision_log.csv",
                 usecols=["policy", "seed", "month", "rule_of_40"])
a2["source"] = "A2"
frames.append(a2)
for pol in ["boardroom", "oracle_v1", "oracle_v3", "oracle_v3_no_memory"]:
    m = pd.read_csv(RES / "a3" / f"monthly_{pol}.csv",
                    usecols=["policy", "seed", "month", "rule_of_40"])
    m["source"] = "A3"
    frames.append(m)
df = pd.concat(frames, ignore_index=True)

curve_rows, rec_rows = [], []
for (source, policy), g in df.groupby(["source", "policy"]):
    series = {seed: sg.set_index("month").rule_of_40 for seed, sg in g.groupby("seed")}
    # ---- event-time curves: average events within episode, then across episodes
    per_episode = {rel: [] for rel in REL}
    for seed, s in series.items():
        for rel in REL:
            vals = [s.get(sm + rel) for sm in SHOCKS if (sm + rel) in s.index and sm - 1 in s.index]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                per_episode[rel].append(np.mean(vals))
    for rel in REL:
        v = np.array(per_episode[rel])
        if not len(v):
            continue
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        curve_rows.append(dict(source=source, policy=policy, rel_month=rel,
                               n_episodes=len(v), mean_r40=v.mean(),
                               ci95_lo=v.mean() - 1.96 * se, ci95_hi=v.mean() + 1.96 * se))
    # ---- recovery per event
    for seed, s in series.items():
        for sm in SHOCKS:
            if sm not in s.index or (sm - 1) not in s.index:
                continue  # episode dead before the shock
            pre = s.loc[sm - 1]
            months = np.nan
            recovered = False
            for m in range(sm + 1, sm + WINDOW + 1):
                if m not in s.index:
                    break  # died inside the window -> censored
                if s.loc[m] >= pre:
                    recovered, months = True, m - sm
                    break
            rec_rows.append(dict(source=source, policy=policy, seed=seed,
                                 shock_month=sm, pre_shock_r40=pre,
                                 recovered=recovered, months_to_recover=months))

pd.DataFrame(curve_rows).to_csv(RES / "a8_shock_r40_curves.csv", index=False)
rec = pd.DataFrame(rec_rows)

summary = (rec.groupby(["source", "policy"])
              .agg(n_events=("recovered", "size"),
                   recovery_rate=("recovered", "mean"),
                   median_months_recovered=("months_to_recover", "median"))
              .reset_index())
by_shock = (rec.groupby(["source", "policy", "shock_month"])
               .agg(n_events=("recovered", "size"), recovery_rate=("recovered", "mean"),
                    median_months_recovered=("months_to_recover", "median"))
               .reset_index())
out = pd.concat([summary.assign(shock_month="all"), by_shock], ignore_index=True)
out.to_csv(RES / "a8_shock_recovery.csv", index=False)

print("A8 overall (R40 recovery within 24 months of a scheduled shock):")
print(summary.round(3).to_string(index=False))
