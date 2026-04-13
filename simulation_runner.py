import sys
import os
import random
from copy import deepcopy
import numpy as np
import pandas as pd

print("Starting simulation runner...", flush=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.startup_env import StartupEnv
from agents.adapter import ActionAdapter
from config import sim_config

from agents.baseline_agents import merge_actions
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from oracle.action_modifier import NoOpActionModifier
from oracle.oracle import Oracle

class BaselineJointAgent:
    """
    Acts as a container for the C-Suite agents, merging their decisions.
    """
    def get_action(self, state):
        return merge_actions(state)

class RandomBundleAgent:
    """
    A 'Dumb' Agent that makes random ActionBundles.
    """
    def get_action(self, state):
        return {
            "marketing": {
                "spend": random.uniform(1000, 20000),
                "channel": random.choice(["ppc", "brand"])
            },
            "hiring": {
                "hires": random.randint(0, 2),
                "cost_per_employee": random.uniform(8000, 12000)
            },
            "product": {
                "r_and_d_spend": random.uniform(1000, 10000)
            },
            "pricing": {
                "price_change_pct": random.uniform(-0.05, 0.05)
            }
        }

from boardroom.boardroom import Boardroom
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent

class BoardroomAgent:
    def __init__(
        self,
        oracle_mode="none",
        oracle_frequency=3,
        enable_action_modifier=True,
        enable_memory_retrieval=True,
        oracle_instance=None,
        action_modifier_instance=None,
    ):
        self.boardroom = Boardroom([
            CFOProposalAgent(),
            CMOProposalAgent(),
            CPOProposalAgent(),
        ],
            use_oracle=(oracle_mode != "none"),
            oracle_frequency=oracle_frequency,
            oracle_mode=oracle_mode,
            oracle_instance=oracle_instance,
            action_modifier_instance=action_modifier_instance,
            enable_action_modifier=enable_action_modifier,
            enable_memory_retrieval=enable_memory_retrieval,
        )

    def start_episode(self, episode_seed):
        self.boardroom.start_episode(episode_seed)

    def get_action(self, state):
        return self.boardroom.decide(state)

    def get_episode_stats(self):
        return self.boardroom.get_episode_stats()

    def set_shock_label(self, shock_label):
        self.boardroom.set_shock_label(shock_label)

    def get_last_brief(self):
        brief = self.boardroom.get_last_brief()
        return brief.model_dump(mode="json") if brief is not None else None

    def get_last_decision_trace(self):
        return self.boardroom.get_last_decision_trace()


def _default_oracle_stats() -> dict:
    return {
        "oracle_refresh_requests": 0,
        "cadence_refreshes": 0,
        "event_refreshes": 0,
        "cache_hits": 0,
        "llm_calls": 0,
    }


def _is_real_shock(shock_label) -> bool:
    return bool(shock_label) and shock_label != "NO_SHOCK"


def _safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def _safe_median(values):
    return float(np.median(values)) if values else np.nan


def _collect_retrieval_rows(
    policy: str,
    episode_index: int,
    episode_seed: int,
    month: int,
    decision_trace: dict | None,
) -> list[dict]:
    rows = []
    trace = decision_trace or {}
    for retrieval_rank, memory in enumerate(trace.get("retrieved_memories") or [], start=1):
        metadata = memory.get("metadata") or {}
        realized_outcome = metadata.get("realized_outcome")
        if realized_outcome == "GROWTH":
            outcome_bucket = "POSITIVE"
        elif realized_outcome == "DECLINE":
            outcome_bucket = "NEGATIVE"
        else:
            outcome_bucket = "NEUTRAL"

        rows.append(
            {
                "policy": policy,
                "episode": episode_index,
                "seed": episode_seed,
                "month": month,
                "refresh_reason": trace.get("refresh_reason"),
                "brief_source": trace.get("brief_source"),
                "shock_label": trace.get("shock_label"),
                "retrieval_rank": retrieval_rank,
                "document": memory.get("document"),
                "source_month": metadata.get("source_month"),
                "stored_global_month": metadata.get("stored_global_month"),
                "realized_outcome": realized_outcome,
                "outcome_bucket": outcome_bucket,
                "memory_weight": memory.get("memory_weight"),
                "similarity_score": memory.get("similarity_score"),
                "recency_factor": memory.get("recency_factor"),
            }
        )
    return rows


def _build_agent_for_policy(policy: str, oracle_frequency: int, oracle_overrides: dict | None = None):
    oracle_overrides = oracle_overrides or {}

    if policy == "heuristic":
        return BaselineJointAgent()
    if policy == "random":
        return RandomBundleAgent()
    if policy == "boardroom":
        return BoardroomAgent(oracle_mode="none")
    if policy == "boardroom_oracle":
        return BoardroomAgent(oracle_mode="oracle_v1", oracle_frequency=oracle_frequency, **oracle_overrides)
    if policy == "oracle_v1_no_modifier":
        return BoardroomAgent(
            oracle_mode="oracle_v1",
            oracle_frequency=oracle_frequency,
            enable_action_modifier=False,
            action_modifier_instance=NoOpActionModifier(),
            **oracle_overrides,
        )
    if policy in {"oracle_v1", "oracle_v2", "oracle_v3"}:
        return BoardroomAgent(oracle_mode=policy, oracle_frequency=oracle_frequency, **oracle_overrides)
    if policy == "oracle_v3_no_memory":
        return BoardroomAgent(
            oracle_mode="oracle_v3",
            oracle_frequency=oracle_frequency,
            enable_memory_retrieval=False,
            oracle_instance=Oracle(mode="oracle_v3", memory_store=None, enable_memory_retrieval=False),
            **oracle_overrides,
        )
    if policy == "oracle_v4":
        return BoardroomAgent(
            oracle_mode="oracle_v4",
            oracle_frequency=oracle_frequency,
            **oracle_overrides,
        )
    if policy == "oracle_v4_causal":
        return BoardroomAgent(
            oracle_mode="oracle_v4_causal",
            oracle_frequency=oracle_frequency,
            **oracle_overrides,
        )
    if policy == "oracle_v3_hetero":
        from agents.llm_client import create_llm_client
        agents = [
            CFOProposalAgent(
                llm_client=create_llm_client("openai", "o4-mini"), use_llm=True
            ),
            CMOProposalAgent(
                llm_client=create_llm_client("anthropic", "claude-sonnet-4-5-20251001"),
                use_llm=True
            ),
            CPOProposalAgent(
                llm_client=create_llm_client("anthropic", "claude-sonnet-4-5-20251001"),
                use_llm=True
            ),
        ]
        return Boardroom(
            agents=agents,
            use_oracle=True,
            oracle_mode="oracle_v3",
            oracle_frequency=oracle_frequency,
            **oracle_overrides,
        )
    raise ValueError(f"Unknown policy: {policy}")

def run_simulation(
    policy: str = "heuristic",
    num_episodes: int = 100,
    seed_start: int = 0,
    oracle_frequency: int = 3,
    oracle_overrides: dict | None = None,
    return_action_trace: bool = False,
    return_monthly_trace: bool = False,
    return_retrieval_trace: bool = False,
):
    print(f"Running {num_episodes} episodes with Policy: {policy} (Seeds {seed_start}-{seed_start+num_episodes-1})...")
    
    env = StartupEnv()
    agent = _build_agent_for_policy(policy, oracle_frequency, oracle_overrides=oracle_overrides)
    
    results = []
    action_trace = []
    monthly_trace = []
    retrieval_trace = []
    
    for i in range(num_episodes):
        episode_seed = seed_start + i
        
        obs, _ = env.reset(seed=episode_seed)
        if hasattr(agent, "start_episode"):
            agent.start_episode(episode_seed)
        if hasattr(agent, "set_shock_label"):
            agent.set_shock_label(None)
        
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        rule_40_history = []
        post_shock_rule40_window = []
        shock_events = []
        pending_recoveries = []
        previous_rule_40 = np.nan
        
        while not (terminated or truncated):
            current_month = env.state.months_elapsed
            raw_action = agent.get_action(env.state)
            decision_trace = agent.get_last_decision_trace() if hasattr(agent, "get_last_decision_trace") else None
            
            clean_action = ActionAdapter.translate_action(raw_action)
            if return_action_trace:
                action_trace.append(
                    {
                        "episode": i,
                        "seed": episode_seed,
                        "policy": policy,
                        "month": current_month,
                        "action": deepcopy(clean_action),
                        "brief": agent.get_last_brief() if hasattr(agent, "get_last_brief") else None,
                        "decision_trace": deepcopy(decision_trace),
                    }
                )

            if return_retrieval_trace:
                retrieval_trace.extend(
                    _collect_retrieval_rows(
                        policy=policy,
                        episode_index=i,
                        episode_seed=episode_seed,
                        month=current_month,
                        decision_trace=decision_trace,
                    )
                )
            
            obs, reward, terminated, truncated, info = env.step(clean_action)
            if hasattr(agent, "set_shock_label"):
                agent.set_shock_label(info.get("shock_label"))

            current_rule_40 = info.get("rule_of_40", 0)
            if 25 <= current_month <= 60:
                post_shock_rule40_window.append(current_rule_40)

            shock_label = info.get("shock_label", "NO_SHOCK")
            if _is_real_shock(shock_label):
                shock_event = {
                    "shock_month": current_month,
                    "shock_label": shock_label,
                    "pre_shock_rule_40": previous_rule_40,
                    "recovered": False,
                    "recovery_month": np.nan,
                    "recovery_time_months": np.nan,
                }
                shock_events.append(shock_event)
                if not np.isnan(previous_rule_40):
                    pending_recoveries.append(shock_event)

            still_pending = []
            for pending in pending_recoveries:
                if current_month > pending["shock_month"] and current_rule_40 >= pending["pre_shock_rule_40"]:
                    pending["recovered"] = True
                    pending["recovery_month"] = current_month
                    pending["recovery_time_months"] = current_month - pending["shock_month"]
                else:
                    still_pending.append(pending)
            pending_recoveries = still_pending
            previous_rule_40 = current_rule_40

            if return_monthly_trace:
                state_snapshot = info.get("state", {})
                monthly_trace.append(
                    {
                        "episode": i,
                        "seed": episode_seed,
                        "policy": policy,
                        "month": current_month,
                        "reward": reward,
                        "rule_of_40": info.get("rule_of_40"),
                        "shock_label": info.get("shock_label", "NO_SHOCK"),
                        "terminated": terminated,
                        "truncated": truncated,
                        "mrr": state_snapshot.get("mrr"),
                        "cash": state_snapshot.get("cash"),
                        "innovation_factor": state_snapshot.get("innovation_factor"),
                        "unemployment": state_snapshot.get("unemployment"),
                        "months_in_depression": state_snapshot.get("months_in_depression"),
                        "brief": agent.get_last_brief() if hasattr(agent, "get_last_brief") else None,
                        "decision_trace": deepcopy(decision_trace),
                    }
                )
            
            total_reward += reward
            steps += 1
            rule_40_history.append(current_rule_40)

        state = env.state
        
        avg_rule_40 = np.mean(rule_40_history) if rule_40_history else 0
        months_above_40 = sum(1 for x in rule_40_history if x >= 40)
        pct_above_40 = (months_above_40 / len(rule_40_history)) * 100 if rule_40_history else 0
        recovery_times = [event["recovery_time_months"] for event in shock_events if not np.isnan(event["recovery_time_months"])]
        recovered_shock_count = sum(1 for event in shock_events if event["recovered"])
        
        ltv_cac = state.ltv / state.cac if state.cac > 0 else 0
        oracle_stats = _default_oracle_stats()
        if hasattr(agent, "get_episode_stats"):
            oracle_stats.update(agent.get_episode_stats())
        
        result = {
            "episode": i,
            "seed": episode_seed,
            "policy": policy,
            "steps": steps,
            "final_mrr": state.mrr,
            "final_cash": state.cash,
            "final_cac": state.cac,
            "final_ltv": state.ltv,
            "final_ltv_cac": ltv_cac,
            "final_headcount": state.headcount,
            "final_valuation_multiple": state.valuation_multiple,
            "final_unemployment": state.unemployment,
            "final_innovation_factor": state.innovation_factor,
            "depression_months": state.months_in_depression,
            "cause": "Bankruptcy" if terminated else "Time Limit",
            "total_reward": total_reward,
            "avg_rule_40": avg_rule_40,
            "pct_above_40": pct_above_40,
            "shock_count": len(shock_events),
            "recovered_shock_count": recovered_shock_count,
            "recovered_shock_rate_pct": (recovered_shock_count / len(shock_events) * 100.0) if shock_events else np.nan,
            "mean_recovery_time_months": _safe_mean(recovery_times),
            "median_recovery_time_months": _safe_median(recovery_times),
            "post_shock_avg_rule40_25_60": _safe_mean(post_shock_rule40_window),
            **oracle_stats,
        }
        if hasattr(agent, "boardroom") and hasattr(agent.boardroom, "oracle"):
            agent.boardroom.oracle.end_episode(episode_metrics=result)
        results.append(result)

        print(
            "EPISODE_END | "
            f"policy={policy} | episode={i} | seed={episode_seed} | cause={result['cause']} | "
            f"months={steps} | final_mrr={state.mrr:,.0f} | final_cash={state.cash:,.0f} | "
            f"oracle_refreshes={result['oracle_refresh_requests']} | cadence={result['cadence_refreshes']} | "
            f"events={result['event_refreshes']} | cache_hits={result['cache_hits']} | llm_calls={result['llm_calls']}"
        )

    df = pd.DataFrame(results)
    
    print(f"\n--- Simulation Summary ({policy}) ---")
    print(f"Survival Rate: {(df['cause'] == 'Time Limit').mean():.2%}")
    print(f"Avg Duration: {df['steps'].mean():.1f} months")
    print(f"Avg Final MRR: ${df['final_mrr'].mean():,.2f}")
    print(f"Avg Rule of 40: {df['avg_rule_40'].mean():.1f}")
    print(f"Avg Innovation Factor: {df['final_innovation_factor'].mean():.2f}")
    print(f"Avg Unemployment: {df['final_unemployment'].mean():.1f}%")
    
    if return_action_trace and return_monthly_trace and return_retrieval_trace:
        return df, {
            "action_trace": action_trace,
            "monthly_trace": monthly_trace,
            "retrieval_trace": retrieval_trace,
        }
    if return_action_trace and return_monthly_trace:
        return df, {
            "action_trace": action_trace,
            "monthly_trace": monthly_trace,
        }
    if return_action_trace and return_retrieval_trace:
        return df, {
            "action_trace": action_trace,
            "retrieval_trace": retrieval_trace,
        }
    if return_action_trace:
        return df, action_trace
    if return_monthly_trace and return_retrieval_trace:
        return df, {
            "monthly_trace": monthly_trace,
            "retrieval_trace": retrieval_trace,
        }
    if return_monthly_trace:
        return df, monthly_trace
    if return_retrieval_trace:
        return df, retrieval_trace
    return df

if __name__ == "__main__":
    run_simulation(policy="oracle_v3", num_episodes=5)
