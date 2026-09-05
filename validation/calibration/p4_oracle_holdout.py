"""Phase 4: oracle_v3 (Ollama llama3.1:8b) on HOLDOUT at real scale, v2 physics.

First-ever oracle-at-real-scale numbers. 8 HOLDOUT companies chosen
deterministically (sorted by init revenue, evenly spaced indices - spans the
scale range), 10 matched seeds (0-9, pairable with the p4 backtest's
boardroom arm), 12 months, v2 flags (fitted curve, financing, scale-aware
corridor). One Oracle instance per company: episodes are sequential and
memory accrues within the company run, mirroring the recorded research
design (disclosed as such). Research-default prompt (no burn context, no
churn benchmark) so the brief channel is the same one A3/A4 characterized.

Writes validation/results/oracle_v3_real_scale_v2.csv (per-episode rows).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import (ROOT, build_state, load_panel, load_split,  # noqa: E402
                          pick_init, rollout)

from agents.proposal_agents import (CFOProposalAgent, CMOProposalAgent,  # noqa: E402
                                    CPOProposalAgent)
from boardroom.boardroom import Boardroom  # noqa: E402
from oracle.oracle import Oracle  # noqa: E402

RESULTS = ROOT / "validation/results"
SEEDS = range(10)
N_COMPANIES = 8
V2_ENV = {"marketing_curve": "v2", "financing_enabled": True,
          "competitive_entry": "scale_neutral"}


def pick_oracle_companies() -> list[str]:
    s = load_split()
    ho = s[s.split == "HOLDOUT"].sort_values("init_revenue").reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(ho) - 1, N_COMPANIES).round().astype(int))
    return ho.loc[idx, "ticker"].tolist()


def main() -> None:
    panel = load_panel()
    tickers = pick_oracle_companies()
    print("oracle companies:", tickers, flush=True)
    rows = []
    out_path = RESULTS / "oracle_v3_real_scale_v2.csv"
    for ticker in tickers:
        row, _ = pick_init(panel[panel.ticker == ticker])
        state = build_state(row)
        scale = state.mrr / 50_000.0
        corridor = "scale_aware"
        oracle = Oracle(mode="oracle_v3", run_id=f"p4_oracle_{ticker}")
        board = Boardroom(
            [CFOProposalAgent(corridor=corridor),
             CMOProposalAgent(corridor=corridor),
             CPOProposalAgent(corridor=corridor)],
            use_oracle=True, oracle_mode="oracle_v3", oracle_instance=oracle,
            corridor=corridor)
        for seed in SEEDS:
            t0 = time.time()
            board.start_episode(seed)
            out = rollout(row, state, "boardroom", seed, scale,
                          extra_env_config=dict(V2_ENV), board=board,
                          corridor=corridor, collect_trace=True)
            stats = board.get_episode_stats()
            rows.append(dict(
                physics_version="v2", policy="oracle_v3", ticker=ticker,
                seed=seed, growth=out["growth"], died=int(out["died"]),
                months_survived=out["months_survived"],
                llm_calls=stats.get("llm_calls", 0),
                cache_hits=stats.get("cache_hits", 0),
                n_raises=sum(1 for t in out["trace"] if t["financing_raise"] > 0),
                wall_s=round(time.time() - t0, 1)))
            print(f"{ticker} seed {seed}: growth="
                  f"{out['growth'] if out['growth'] is not None else 'DIED'} "
                  f"llm={rows[-1]['llm_calls']} cache={rows[-1]['cache_hits']} "
                  f"({rows[-1]['wall_s']}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
