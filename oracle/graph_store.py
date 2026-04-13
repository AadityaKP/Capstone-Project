"""
Neo4j integration for oracle_v4_causal.

Writes shock events, decisions, and outcomes as a causal graph.
Queries historical shock patterns to augment Oracle prompt context.

Gracefully degrades (self.enabled = False) if Neo4j is unavailable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from oracle.schemas import (
    CausalChainSummary,
    GraphContext,
    GraphShockRecord,
    OracleBrief,
    StateSnapshot,
)

load_dotenv()

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class CausalGraphStore:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.enabled = False
        self.driver = None

        if not NEO4J_AVAILABLE:
            print("[CausalGraphStore] neo4j package not installed. Graph store disabled.")
            return

        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self._ensure_indexes()
            self.enabled = True
            print("[CausalGraphStore] Connected to Neo4j.")
        except Exception as exc:
            print(f"[CausalGraphStore] Neo4j unavailable, disabling: {exc}")
            self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def write_episode(self, episode_metrics: Dict[str, Any]) -> None:
        """Called by Oracle.end_episode(). Creates or updates an Episode node."""

        if not self.enabled:
            return

        cypher = """
        MERGE (e:Episode {episode_id: $episode_id})
        SET e.seed = $seed,
            e.policy = $policy,
            e.final_mrr = $final_mrr,
            e.final_cash = $final_cash,
            e.steps = $steps,
            e.cause = $cause,
            e.total_reward = $total_reward
        """
        params = {
            "episode_id": int(episode_metrics.get("seed", episode_metrics.get("episode", -1))),
            "seed": int(episode_metrics.get("seed", -1)),
            "policy": episode_metrics.get("policy", "unknown"),
            "final_mrr": float(episode_metrics.get("final_mrr", 0)),
            "final_cash": float(episode_metrics.get("final_cash", 0)),
            "steps": int(episode_metrics.get("steps", 0)),
            "cause": episode_metrics.get("cause", "unknown"),
            "total_reward": float(episode_metrics.get("total_reward", 0)),
        }
        self._run(cypher, params)

    def write_shock_event(
        self,
        episode_id: int,
        shock_label: str,
        month: int,
        pre_state: StateSnapshot,
        decision: Dict[str, Any],
        brief: Optional[OracleBrief],
    ) -> None:
        """
        Called by Boardroom when a hard shock is detected.
        Creates Shock, StateSnapshot, and Decision nodes and their edges.
        """

        if not self.enabled:
            return

        shock_type = shock_label.split(":")[0].strip() if shock_label else "UNKNOWN"

        from oracle.context import get_churn_tier, get_innovation_tier, get_mrr_tier

        mrr_tier = get_mrr_tier(pre_state.mrr)
        churn_tier = get_churn_tier(pre_state.avg_churn)
        innov_tier = get_innovation_tier(pre_state.innovation)

        cypher = """
        MERGE (e:Episode {episode_id: $episode_id})
        MERGE (sh:Shock {episode_id: $episode_id, shock_month: $shock_month})
        SET sh.shock_type = $shock_type,
            sh.shock_label = $shock_label,
            sh.mrr_at_shock = $mrr_at_shock,
            sh.mrr_tier = $mrr_tier
        MERGE (e)-[:HAD_SHOCK]->(sh)
        MERGE (snap:StateSnapshot {episode_id: $episode_id, month: $shock_month})
        SET snap.mrr = $mrr_at_shock,
            snap.avg_churn = $avg_churn,
            snap.innovation = $innovation,
            snap.mrr_tier = $mrr_tier,
            snap.churn_tier = $churn_tier,
            snap.innov_tier = $innov_tier
        MERGE (snap)-[:PRECEDED]->(sh)
        MERGE (d:Decision {episode_id: $episode_id, month: $shock_month})
        SET d.marketing_spend = $marketing_spend,
            d.rd_spend = $rd_spend,
            d.hires = $hires,
            d.price_change_pct = $price_change_pct,
            d.brief_risk_level = $brief_risk_level,
            d.brief_growth_outlook = $brief_growth_outlook,
            d.brief_confidence = $brief_confidence
        MERGE (sh)-[:FOLLOWED_BY]->(d)
        """

        marketing = decision.get("marketing", {})
        product = decision.get("product", {})
        hiring = decision.get("hiring", {})
        pricing = decision.get("pricing", {})

        params = {
            "episode_id": episode_id,
            "shock_month": month,
            "shock_type": shock_type,
            "shock_label": shock_label,
            "mrr_at_shock": float(pre_state.mrr),
            "avg_churn": float(pre_state.avg_churn),
            "innovation": float(pre_state.innovation),
            "mrr_tier": mrr_tier,
            "churn_tier": churn_tier,
            "innov_tier": innov_tier,
            "marketing_spend": float(marketing.get("spend", 0)),
            "rd_spend": float(product.get("r_and_d_spend", 0)),
            "hires": int(hiring.get("hires", 0)),
            "price_change_pct": float(pricing.get("price_change_pct", 0)),
            "brief_risk_level": brief.risk_level.value if brief else "UNKNOWN",
            "brief_growth_outlook": brief.growth_outlook.value if brief else "UNKNOWN",
            "brief_confidence": float(brief.confidence) if brief else 0.5,
        }
        self._run(cypher, params)

    def write_outcome(
        self,
        episode_id: int,
        shock_month: int,
        outcome_metrics: Dict[str, Any],
    ) -> None:
        """
        Called 6 months after a shock (mirroring memory maturation).
        Creates an Outcome node and edges from Shock and Decision.
        """

        if not self.enabled:
            return

        cypher = """
        MATCH (sh:Shock {episode_id: $episode_id, shock_month: $shock_month})
        MATCH (d:Decision {episode_id: $episode_id, month: $shock_month})
        MERGE (o:Outcome {episode_id: $episode_id, shock_month: $shock_month})
        SET o.recovery_months = $recovery_months,
            o.recovered = $recovered,
            o.post_shock_rule40 = $post_shock_rule40,
            o.mrr_change_pct = $mrr_change_pct
        MERGE (sh)-[:CAUSED]->(o)
        MERGE (d)-[:PRODUCED]->(o)
        """
        params = {
            "episode_id": episode_id,
            "shock_month": shock_month,
            "recovery_months": outcome_metrics.get("recovery_months"),
            "recovered": bool(outcome_metrics.get("recovered", False)),
            "post_shock_rule40": float(outcome_metrics.get("post_shock_rule40", 0)),
            "mrr_change_pct": float(outcome_metrics.get("mrr_change_pct", 0)),
        }
        self._run(cypher, params)

    def query_similar_shocks(
        self,
        shock_type: str,
        mrr_tier: str,
        n: int = 5,
    ) -> List[GraphShockRecord]:
        """Returns top-n historical shocks of the same type in the same MRR tier."""

        if not self.enabled:
            return []

        cypher = """
        MATCH (sh:Shock {shock_type: $shock_type})
        WHERE sh.mrr_tier = $mrr_tier
        MATCH (sh)-[:FOLLOWED_BY]->(d:Decision)
        MATCH (sh)-[:CAUSED]->(o:Outcome)
        RETURN sh, d, o
        ORDER BY o.mrr_change_pct DESC
        LIMIT $n
        """
        records = self._query(
            cypher,
            {"shock_type": shock_type, "mrr_tier": mrr_tier, "n": n},
        )

        results = []
        for row in records:
            shock = row["sh"]
            decision = row["d"]
            outcome = row["o"]
            results.append(
                GraphShockRecord(
                    episode_id=shock.get("episode_id", -1),
                    shock_type=shock.get("shock_type", shock_type),
                    shock_month=shock.get("shock_month", 0),
                    mrr_tier=shock.get("mrr_tier", mrr_tier),
                    brief_risk_level=decision.get("brief_risk_level", "UNKNOWN"),
                    marketing_spend=float(decision.get("marketing_spend", 0)),
                    rd_spend=float(decision.get("rd_spend", 0)),
                    hires=int(decision.get("hires", 0)),
                    recovery_months=outcome.get("recovery_months"),
                    recovered=bool(outcome.get("recovered", False)),
                    post_shock_rule40=float(outcome.get("post_shock_rule40", 0)),
                    mrr_change_pct=float(outcome.get("mrr_change_pct", 0)),
                )
            )
        return results

    def query_causal_chain(self, shock_type: str) -> Optional[CausalChainSummary]:
        """
        Returns aggregated statistics for a shock type across all episodes.
        Groups by brief_risk_level to identify which risk classification led to
        the fastest recovery.
        """

        if not self.enabled:
            return None

        aggregate_cypher = """
        MATCH (sh:Shock {shock_type: $shock_type})-[:CAUSED]->(o:Outcome)
        RETURN
            count(o) AS total,
            avg(o.recovery_months) AS mean_recovery,
            avg(CASE WHEN o.recovered THEN 1.0 ELSE 0.0 END) AS recovery_rate,
            avg(o.post_shock_rule40) AS mean_rule40
        """
        aggregate = self._query(aggregate_cypher, {"shock_type": shock_type})
        if not aggregate or aggregate[0]["total"] == 0:
            return None

        row = aggregate[0]
        total = int(row["total"])
        mean_recovery = float(row["mean_recovery"] or 0)
        recovery_rate = float(row["recovery_rate"] or 0)
        mean_rule40 = float(row["mean_rule40"] or 0)

        breakdown_cypher = """
        MATCH (sh:Shock {shock_type: $shock_type})-[:FOLLOWED_BY]->(d:Decision)
        MATCH (sh)-[:CAUSED]->(o:Outcome)
        WHERE o.recovery_months IS NOT NULL
        RETURN d.brief_risk_level AS risk_level, avg(o.recovery_months) AS avg_recovery
        ORDER BY avg_recovery ASC
        """
        breakdown = self._query(breakdown_cypher, {"shock_type": shock_type})
        best_risk = breakdown[0]["risk_level"] if breakdown else None
        worst_risk = breakdown[-1]["risk_level"] if len(breakdown) > 1 else None

        return CausalChainSummary(
            shock_type=shock_type,
            total_occurrences=total,
            mean_recovery_months=mean_recovery,
            recovery_rate=recovery_rate,
            mean_post_shock_rule40=mean_rule40,
            best_risk_level=best_risk,
            worst_risk_level=worst_risk,
        )

    def build_graph_context(
        self,
        shock_type: Optional[str],
        mrr_tier: str,
    ) -> GraphContext:
        """
        Convenience wrapper called by Oracle.get_context() when a shock is active.
        Returns an empty GraphContext if no shock or no data.
        """

        if not shock_type or shock_type == "NO_SHOCK" or not self.enabled:
            return GraphContext()

        similar = self.query_similar_shocks(shock_type, mrr_tier)
        summary = self.query_causal_chain(shock_type)

        return GraphContext(
            similar_shocks=similar,
            causal_summary=summary,
            active_shock_type=shock_type,
        )

    def _ensure_indexes(self) -> None:
        indexes = [
            "CREATE INDEX shock_type_idx IF NOT EXISTS FOR (s:Shock) ON (s.shock_type)",
            "CREATE INDEX shock_mrr_tier_idx IF NOT EXISTS FOR (s:Shock) ON (s.mrr_tier)",
            "CREATE INDEX episode_policy_idx IF NOT EXISTS FOR (e:Episode) ON (e.policy)",
            "CREATE INDEX outcome_episode_idx IF NOT EXISTS FOR (o:Outcome) ON (o.episode_id)",
        ]
        for index in indexes:
            try:
                self._run(index, {})
            except Exception:
                pass

    def _run(self, cypher: str, params: Dict[str, Any]) -> None:
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                session.run(cypher, params)
        except Exception as exc:
            print(f"[CausalGraphStore] Write failed: {exc}")

    def _query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params)
                return [dict(record) for record in result]
        except Exception as exc:
            print(f"[CausalGraphStore] Query failed: {exc}")
            return []
