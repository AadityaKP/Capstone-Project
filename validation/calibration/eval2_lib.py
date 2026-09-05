"""Round-2 runner helpers (R2-2). Mapping flag lives here, not in the engine:
`mapping_version` only changes what state the backtest hands the physics.

Defaults reproduce round-1 behaviour exactly (mapping_version="v1" gives the
round-1 CAC/LTV identity; financing_model defaults to "rescue" inside the
engine). tests/test_round2.py asserts both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, build_state, physics_patches, rollout  # noqa: E402


def load_eval2_states() -> pd.DataFrame:
    return pd.read_csv(CAL_DIR / "eval2_states.csv")


def load_hazard() -> dict:
    return json.loads((CAL_DIR / "financing_hazard_r2.json").read_text())


def build_state_eval2(row, mapping_version: str = "v1"):
    """`row` is an eval2_states.csv row (itertuples). v2 overrides CAC with the
    company-specific value; LTV stays price/churn exactly as v1, so LTV:CAC
    varies across companies (PROTOCOL_round2.md)."""
    if mapping_version not in {"v1", "v2"}:
        raise ValueError(f"unknown mapping_version: {mapping_version!r}")
    state = build_state(row, float(row.price_assumed))
    if mapping_version == "v2":
        if row.cac_v2 is None or (isinstance(row.cac_v2, float) and np.isnan(row.cac_v2)):
            raise ValueError(f"{row.ticker}: no cac_v2 available")
        state.cac = float(row.cac_v2)
    return state


def eval2_env_config(financing: bool, financing_model: str = "rescue",
                     hazard: dict | None = None) -> dict:
    cfg = {"marketing_curve": "v2", "competitive_entry": "scale_neutral",
           "financing_enabled": bool(financing)}
    if financing and financing_model == "opportunistic":
        hz = hazard or load_hazard()
        cfg["financing_model"] = "opportunistic"
        cfg["financing_hazard"] = {"bins": hz["bins"], "h": hz["h"], "K": hz["K"]}
    return cfg


def run_eval2_company_arm(row, arm: str, seeds,
                          mapping_version: str = "v1",
                          financing: bool = True,
                          financing_model: str = "rescue",
                          hazard: dict | None = None,
                          corridor_for_agents: str | None = None) -> dict:
    """Mirror of backtest_lib.run_company_arm on an eval2 row."""
    state = build_state_eval2(row, mapping_version)
    scale = state.mrr / 50_000.0
    corridor = (corridor_for_agents if corridor_for_agents is not None
                else ("scale_aware" if arm in ("heuristic", "boardroom") else "legacy"))
    cfg = eval2_env_config(financing, financing_model, hazard)
    growths, deaths, n_raises, months = [], 0, 0, []
    with physics_patches():
        for seed in seeds:
            out = rollout(row, state, arm, seed, scale,
                          extra_env_config=dict(cfg), corridor=corridor,
                          collect_trace=True)
            months.append(out["months_survived"])
            n_raises += sum(1 for t in out["trace"] if t["financing_raise"] > 0)
            if out["died"]:
                deaths += 1
            else:
                growths.append(out["growth"])
    return dict(ticker=row.ticker, arm=arm, price=float(row.price_assumed),
                mapping_version=mapping_version, financing=financing,
                financing_model=financing_model,
                growths=growths, deaths=deaths, n_seeds=len(list(seeds)),
                n_financing_raises=n_raises,
                median_months_survived=float(np.median(months)),
                median_growth=float(np.median(growths)) if growths else np.nan)
