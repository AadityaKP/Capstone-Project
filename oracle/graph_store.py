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
    CausalGraphContext,
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
            self._ensure_seed_edges()
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

    def query_role_causal_context(
        self,
        stress_node: str,
        role: str,
        limit: int = 3,
    ) -> CausalGraphContext:
        """Return a concise role-filtered causal chain around a stress node."""

        empty_context = CausalGraphContext(
            role=role,
            stress_node=stress_node,
            chain_summary="",
            confidence=0.0,
        )
        if not self.enabled:
            return empty_context

        query_limit = max(limit * 6, limit)
        cypher = """
        MATCH (root)
        WHERE root.name = $stress_node
        MATCH p = (root)-[*1..3]-(target)
        UNWIND relationships(p) AS rel
        WITH startNode(rel) AS s, rel, endNode(rel) AS o, min(length(p)) AS distance
        WITH
            coalesce(s.name, s.id, labels(s)[0]) AS subject,
            type(rel) AS predicate,
            coalesce(o.name, o.id, labels(o)[0]) AS object,
            coalesce(rel.confidence, 0.5) AS confidence,
            distance
        RETURN subject, predicate, object, confidence
        ORDER BY distance ASC, confidence DESC, subject, predicate, object
        LIMIT $query_limit
        """
        rows = self._query(
            cypher,
            {"stress_node": stress_node, "query_limit": query_limit},
        )

        triples_with_confidence: list[tuple[list[str], float]] = []
        for row in rows:
            subject = str(row.get("subject") or "")
            predicate = str(row.get("predicate") or "")
            obj = str(row.get("object") or "")
            if not subject or not predicate or not obj:
                continue
            confidence = float(row.get("confidence") or 0.5)
            triples_with_confidence.append(([subject, predicate, obj], confidence))

        selected = self._select_role_triples(role, triples_with_confidence, limit)
        if not selected:
            selected = triples_with_confidence[:limit]
        if not selected:
            return empty_context

        raw_triples = [triple for triple, _ in selected]
        confidence = sum(score for _, score in selected) / len(selected)
        chain_summary = "; ".join(
            f"{subj} -{pred}-> {obj}" for subj, pred, obj in raw_triples
        )
        root_cause_node = raw_triples[0][2] if raw_triples else None

        return CausalGraphContext(
            role=role,
            stress_node=stress_node,
            chain_summary=chain_summary,
            root_cause_node=root_cause_node,
            confidence=max(0.0, min(1.0, confidence)),
            raw_triples=raw_triples,
        )

    def write_action_outcome(
        self,
        action: Dict[str, Any],
        kpi_delta: Dict[str, float],
        confidence: float = 0.6,
        stress_node: Optional[str] = None,
        episode_id: Optional[int] = None,
        month: Optional[int] = None,
    ) -> None:
        """Write closed-loop action-to-KPI evidence for causal proposal learning."""

        if not self.enabled:
            return

        action_name = self._action_pattern_name(action)
        stress_name = stress_node or "Steady_State"
        for metric, delta in kpi_delta.items():
            if delta is None:
                continue
            delta_value = float(delta)
            delta_name = self._kpi_delta_name(metric, delta_value)
            positive = 1 if self._is_positive_delta(metric, delta_value) else 0
            confidence_increment = 0.05 if positive else -0.03

            cypher = """
            MERGE (s:Stress {name: $stress_name})
            MERGE (a:ActionPattern {name: $action_name})
            SET a.last_episode_id = $episode_id,
                a.last_month = $month
            MERGE (k:KPIDelta {name: $delta_name})
            SET k.metric = $metric
            MERGE (s)-[:OBSERVED_WITH]->(a)
            MERGE (a)-[r:MAY_CAUSE]->(k)
            SET r.observations = coalesce(r.observations, 0) + 1,
                r.positive_observations = coalesce(r.positive_observations, 0) + $positive,
                r.last_delta = $delta_value,
                r.last_episode_id = $episode_id,
                r.last_month = $month,
                r.confidence = CASE
                    WHEN coalesce(r.confidence, $base_confidence) + $confidence_increment > 0.95 THEN 0.95
                    WHEN coalesce(r.confidence, $base_confidence) + $confidence_increment < 0.05 THEN 0.05
                    ELSE coalesce(r.confidence, $base_confidence) + $confidence_increment
                END
            """
            params = {
                "stress_name": stress_name,
                "action_name": action_name,
                "delta_name": delta_name,
                "metric": metric,
                "delta_value": delta_value,
                "positive": positive,
                "base_confidence": float(confidence),
                "confidence_increment": confidence_increment,
                "episode_id": int(episode_id or 0),
                "month": int(month or 0),
            }
            self._run(cypher, params)

            promote_cypher = """
            MATCH (a:ActionPattern {name: $action_name})-[r:MAY_CAUSE]->(k:KPIDelta {name: $delta_name})
            WHERE r.confidence >= 0.85 AND r.positive_observations >= 3
            MERGE (a)-[c:CONFIRMED_CAUSE]->(k)
            SET c.confidence = r.confidence,
                c.observations = r.observations,
                c.positive_observations = r.positive_observations,
                c.last_episode_id = $episode_id,
                c.last_month = $month
            """
            self._run(promote_cypher, params)

    def _ensure_indexes(self) -> None:
        indexes = [
            "CREATE INDEX shock_type_idx IF NOT EXISTS FOR (s:Shock) ON (s.shock_type)",
            "CREATE INDEX shock_mrr_tier_idx IF NOT EXISTS FOR (s:Shock) ON (s.mrr_tier)",
            "CREATE INDEX episode_policy_idx IF NOT EXISTS FOR (e:Episode) ON (e.policy)",
            "CREATE INDEX outcome_episode_idx IF NOT EXISTS FOR (o:Outcome) ON (o.episode_id)",
            "CREATE INDEX stress_name_idx IF NOT EXISTS FOR (s:Stress) ON (s.name)",
            "CREATE INDEX action_pattern_name_idx IF NOT EXISTS FOR (a:ActionPattern) ON (a.name)",
            "CREATE INDEX kpi_delta_name_idx IF NOT EXISTS FOR (k:KPIDelta) ON (k.name)",
            "CREATE INDEX causal_lever_name_idx IF NOT EXISTS FOR (c:CausalLever) ON (c.name)",
        ]
        for index in indexes:
            try:
                self._run(index, {})
            except Exception:
                pass

    def _ensure_seed_edges(self) -> None:
        """Seed sparse causal coverage for stress states before live evidence exists."""

        seed_edges = [
            {
                "stress_name": "Cash_Shortage",
                "lever_name": "Runway_Depletion",
                "role": "CFO",
                "confidence": 0.82,
                "description": "Cash shortage raises runway risk and requires cash discipline.",
            },
            {
                "stress_name": "Cash_Shortage",
                "lever_name": "Hiring_Freeze_Recommended",
                "role": "CFO",
                "confidence": 0.76,
                "description": "Cash shortage should limit new hiring costs.",
            },
            {
                "stress_name": "Cash_Shortage",
                "lever_name": "Marketing_Spend_Cut",
                "role": "CMO",
                "confidence": 0.68,
                "description": "Cash shortage should reduce discretionary marketing spend.",
            },
            {
                "stress_name": "Cash_Shortage",
                "lever_name": "Product_Investment_Delay",
                "role": "CPO",
                "confidence": 0.60,
                "description": "Cash shortage can require delaying product investment.",
            },
            {
                "stress_name": "Churn_Spike",
                "lever_name": "Tech_Debt_Remediation",
                "role": "CPO",
                "confidence": 0.72,
                "description": "Churn spikes often trace back to product quality and tech debt issues.",
            },
            {
                "stress_name": "Churn_Spike",
                "lever_name": "Acquisition_Channel_Reallocation",
                "role": "CMO",
                "confidence": 0.62,
                "description": "Churn spikes call for reallocating acquisition spend toward better-fit channels.",
            },
            {
                "stress_name": "Churn_Spike",
                "lever_name": "Cost_Structure_Review",
                "role": "CFO",
                "confidence": 0.58,
                "description": "Churn spikes erode revenue and warrant a review of the cost structure.",
            },
            {
                "stress_name": "Steady_State",
                "lever_name": "Margin_Optimization_Review",
                "role": "CFO",
                "confidence": 0.60,
                "description": "Steady state is a good time to tighten margins and cost discipline.",
            },
            {
                "stress_name": "Steady_State",
                "lever_name": "Growth_Channel_Diversification",
                "role": "CMO",
                "confidence": 0.65,
                "description": "Steady state allows experimentation with new acquisition channels.",
            },
            {
                "stress_name": "Steady_State",
                "lever_name": "Product_Quality_Investment",
                "role": "CPO",
                "confidence": 0.70,
                "description": "Steady state is a good time to invest in product quality and reduce future churn risk.",
            },
            {
                "stress_name": "CAC_Pressure",
                "lever_name": "Pricing_Power_Assessment",
                "role": "CFO",
                "confidence": 0.72,
                "description": "High CAC pressure calls for a pricing review to improve unit economics.",
            },
            {
                "stress_name": "CAC_Pressure",
                "lever_name": "CAC_Reduction_Campaign",
                "role": "CMO",
                "confidence": 0.65,
                "description": "CAC pressure requires shifting acquisition mix toward higher-ROI channels.",
            },
            {
                "stress_name": "CAC_Pressure",
                "lever_name": "Product_Retention_Investment",
                "role": "CPO",
                "confidence": 0.58,
                "description": "Improving product retention raises LTV and offsets a high CAC.",
            },
            {
                "stress_name": "Growth_Stall",
                "lever_name": "Growth_Acceleration_Push",
                "role": "CMO",
                "confidence": 0.72,
                "description": "Growth stall demands an aggressive acquisition and MRR expansion push.",
            },
            {
                "stress_name": "Growth_Stall",
                "lever_name": "Innovation_Pipeline_Build",
                "role": "CPO",
                "confidence": 0.65,
                "description": "Growth stall calls for investing in the product innovation pipeline.",
            },
            {
                "stress_name": "Growth_Stall",
                "lever_name": "Burn_Rate_Reduction",
                "role": "CFO",
                "confidence": 0.58,
                "description": "Growth stall requires burn discipline to extend runway while growth recovers.",
            },
        ]
        cypher = """
        UNWIND $edges AS edge
        MERGE (s:Stress {name: edge.stress_name})
        MERGE (l:CausalLever {name: edge.lever_name})
        SET l.role = edge.role,
            l.description = edge.description
        MERGE (s)-[r:MAY_CAUSE]->(l)
        ON CREATE SET
            r.confidence = edge.confidence,
            r.seed = true,
            r.observations = 0
        ON MATCH SET
            r.seed = coalesce(r.seed, true),
            r.confidence = CASE
                WHEN coalesce(r.observations, 0) = 0 THEN edge.confidence
                ELSE coalesce(r.confidence, edge.confidence)
            END
        """
        try:
            self._run(cypher, {"edges": seed_edges})
        except Exception as exc:
            print(f"[CausalGraphStore] Seed edge setup failed: {exc}")

    @staticmethod
    def _select_role_triples(
        role: str,
        triples_with_confidence: list[tuple[list[str], float]],
        limit: int,
    ) -> list[tuple[list[str], float]]:
        keywords_by_role = {
            "CFO": {
                "cash",
                "runway",
                "burn",
                "cost",
                "hiring",
                "price",
                "pricing",
                "capital",
                "margin",
            },
            "CMO": {
                "cac",
                "marketing",
                "ad",
                "roi",
                "growth",
                "mrr",
                "acquisition",
                "channel",
                "lead",
            },
            "CPO": {
                "churn",
                "retention",
                "product",
                "quality",
                "innovation",
                "tech",
                "debt",
                "nrr",
                "r_and_d",
                "rd",
            },
        }
        keywords = keywords_by_role.get(role, set())
        selected: list[tuple[list[str], float]] = []
        for triple, confidence in triples_with_confidence:
            # Skip the subject: it's the query's anchor (the stress node, for
            # distance-1 triples) and identical across every candidate, so
            # matching it tells us nothing about which role the triple
            # belongs to - it only causes spurious matches when the stress
            # node's name happens to share a substring with a role keyword
            # (e.g. "cac_pressure" -> CMO's "cac", "steady_state" -> CMO's "ad").
            text = " ".join(triple[1:]).lower()
            if any(keyword in text for keyword in keywords):
                selected.append((triple, confidence))
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _action_pattern_name(action: Dict[str, Any]) -> str:
        marketing = action.get("marketing", {})
        hiring = action.get("hiring", {})
        product = action.get("product", {})
        pricing = action.get("pricing", {})

        marketing_spend = float(marketing.get("spend", 0.0) or 0.0)
        rd_spend = float(product.get("r_and_d_spend", 0.0) or 0.0)
        hires = int(hiring.get("hires", 0) or 0)
        price_change = float(pricing.get("price_change_pct", 0.0) or 0.0)

        spend_bucket = "high" if marketing_spend >= 20_000 else "mid" if marketing_spend >= 5_000 else "low"
        rd_bucket = "high" if rd_spend >= 20_000 else "mid" if rd_spend >= 5_000 else "low"
        price_bucket = "up" if price_change > 0.01 else "down" if price_change < -0.01 else "flat"

        return (
            f"marketing_{spend_bucket}|rd_{rd_bucket}|"
            f"hires_{hires}|price_{price_bucket}"
        )

    @staticmethod
    def _kpi_delta_name(metric: str, delta_value: float) -> str:
        if abs(delta_value) < 1e-9:
            direction = "Flat"
        elif delta_value > 0:
            direction = "Up"
        else:
            direction = "Down"
        return f"{metric}_{direction}"

    @staticmethod
    def _is_positive_delta(metric: str, delta_value: float) -> bool:
        metric_lower = metric.lower()
        if "churn" in metric_lower:
            return delta_value < 0
        return delta_value > 0

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
