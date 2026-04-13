from oracle.schemas import ExpectedOutcome, OracleBrief, URGENCY_MAPPING


class WeightAdapter:
    def adjust_weights(self, base_weights: dict, brief: OracleBrief, oracle_mode: str = "oracle_v1") -> dict:
        """
        Merges base weights with Oracle-driven adjustments.
        Applies exponential smoothing to prevent unstable policy jumps.
        """
        memory_aware_modes = {"oracle_v3", "oracle_v4", "oracle_v4_causal"}

        # Map signals numerically
        innov_signal = URGENCY_MAPPING.get(brief.innovation_urgency.value, 0.5)
        eff_signal = URGENCY_MAPPING.get(brief.efficiency_pressure.value, 0.5)
        growth_signal = URGENCY_MAPPING.get(brief.growth_outlook.value, 0.5)
        macro_signal = URGENCY_MAPPING.get(brief.macro_condition.value, 0.5)
        
        # Dampen signal with confidence scalar
        base_k = 0.3 if oracle_mode in memory_aware_modes else 0.2
        k = base_k * brief.confidence
        
        target_weights = {
            "efficiency": base_weights.get("efficiency", 0.3) + (k * eff_signal),
            "growth": base_weights.get("growth", 0.2) + (k * growth_signal),
            "innovation": base_weights.get("innovation", 0.4) + (k * innov_signal),
            "macro": base_weights.get("macro", 0.1) + (k * macro_signal) 
        }

        if oracle_mode in memory_aware_modes and brief.expected_outcome == ExpectedOutcome.DECLINE:
            target_weights["innovation"] += 0.05
            target_weights["efficiency"] += 0.05
        
        # Normalize target mathematically
        total = sum(target_weights.values())
        target_weights = {k: v / total for k, v in target_weights.items()}
        
        # Smooth weight updates: 70% old, 30% new
        final_weights = {}
        for key in base_weights:
            final_weights[key] = (0.7 * base_weights[key]) + (0.3 * target_weights.get(key, base_weights[key]))
            
        # Final safety normalization
        final_total = sum(final_weights.values())
        return {k: v / final_total for k, v in final_weights.items()}
