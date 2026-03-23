from oracle.schemas import OracleBrief, URGENCY_MAPPING

class WeightAdapter:
    def adjust_weights(self, base_weights: dict, brief: OracleBrief) -> dict:
        """
        Merges base weights with Oracle-driven adjustments.
        Applies exponential smoothing to prevent unstable policy jumps.
        """
        # Map signals numerically
        innov_signal = URGENCY_MAPPING.get(brief.innovation_urgency.value, 0.5)
        eff_signal = URGENCY_MAPPING.get(brief.efficiency_pressure.value, 0.5)
        growth_signal = URGENCY_MAPPING.get(brief.growth_outlook.value, 0.5)
        macro_signal = URGENCY_MAPPING.get(brief.macro_condition.value, 0.5)
        
        # Dampen signal with confidence scalar
        k = 0.2 * brief.confidence
        
        target_weights = {
            "efficiency": base_weights.get("efficiency", 0.3) + (k * eff_signal),
            "growth": base_weights.get("growth", 0.2) + (k * growth_signal),
            "innovation": base_weights.get("innovation", 0.4) + (k * innov_signal),
            "macro": base_weights.get("macro", 0.1) + (k * macro_signal) 
        }
        
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
