"""Summarize the Phase 4 oracle_v3-at-real-scale run, paired vs boardroom.

For the same 8 HOLDOUT companies and seeds 0-9, re-runs the no-oracle
boardroom arm under identical v2 flags (deterministic_rng gives both arms the
identical world at equal seed - fixed draw count per step) and reports the
per-episode paired difference in 4q growth, Wilcoxon, and survival. Appends
the summary to calibration_report.md section 6 material (printed; numbers
transcribed there) and writes oracle_v3_real_scale_v2_summary.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import ROOT, load_panel, pick_init, run_company_arm  # noqa: E402
from p4_oracle_holdout import SEEDS, V2_ENV, pick_oracle_companies  # noqa: E402

RESULTS = ROOT / "validation/results"


def main() -> None:
    oc = pd.read_csv(RESULTS / "oracle_v3_real_scale_v2.csv")
    panel = load_panel()
    rows = []
    for ticker in pick_oracle_companies():
        row, _ = pick_init(panel[panel.ticker == ticker])
        out = run_company_arm(row, "boardroom", SEEDS,
                              extra_env_config=dict(V2_ENV),
                              corridor="scale_aware")
        # per-seed growths align with seeds 0..9 minus deaths; rebuild per-seed
        # via rollout order: run_company_arm appends in seed order for survivors
        # - safer: rerun explicitly per seed
        rows.append((ticker, out))
    # explicit per-seed rerun for exact pairing
    from backtest_lib import build_state, rollout
    paired = []
    for ticker, _ in rows:
        row, _ = pick_init(panel[panel.ticker == ticker])
        state = build_state(row)
        scale = state.mrr / 50_000.0
        for seed in SEEDS:
            b = rollout(row, state, "boardroom", seed, scale,
                        extra_env_config=dict(V2_ENV), corridor="scale_aware")
            o = oc[(oc.ticker == ticker) & (oc.seed == seed)]
            paired.append(dict(
                ticker=ticker, seed=seed,
                boardroom_growth=b["growth"],
                boardroom_died=int(b["died"]),
                oracle_growth=float(o.growth.iloc[0]) if len(o) and pd.notna(o.growth.iloc[0]) else np.nan,
                oracle_died=int(o.died.iloc[0]) if len(o) else np.nan,
                oracle_llm_calls=int(o.llm_calls.iloc[0]) if len(o) else 0))
    df = pd.DataFrame(paired)
    df.to_csv(RESULTS / "oracle_v3_real_scale_v2_summary.csv", index=False)

    both = df.dropna(subset=["boardroom_growth", "oracle_growth"])
    diff = (both.oracle_growth - both.boardroom_growth).to_numpy()
    wil = stats.wilcoxon(diff) if len(diff) > 5 else None
    print(f"episodes: {len(df)} | both-survived pairs: {len(both)}")
    print(f"oracle survival: {1 - df.oracle_died.mean():.0%} | "
          f"boardroom survival: {1 - df.boardroom_died.mean():.0%}")
    print(f"median oracle growth {both.oracle_growth.median():+.1%} vs "
          f"boardroom {both.boardroom_growth.median():+.1%}")
    print(f"paired diff (oracle-boardroom): median {np.median(diff):+.1%}, "
          f"positive in {(diff > 0).mean():.0%} of pairs"
          + (f", Wilcoxon p={wil.pvalue:.2g}" if wil else ""))
    print(f"per-company medians:")
    print(both.groupby('ticker')[['boardroom_growth', 'oracle_growth']]
          .median().to_string(float_format='%.3f'))
    print(f"total LLM calls: {df.oracle_llm_calls.sum()}")


if __name__ == "__main__":
    main()
