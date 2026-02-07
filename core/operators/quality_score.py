from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator

@register_operator
class QualityScoreOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="quality_score",
            display_name="Final Quality Score",
            category="quality",
            description="Computes the weighted sum of all quality metrics.",
            min_value=0.0,
            max_value=100.0,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        if not ctx.config:
            return OperatorResult(error="Missing application config")
            
        # These come from shared_data or are passed in params
        # Actually, it's better if this operator runs LAST and sees other metrics
        # But Operators are currently independent.
        # Maybe QualityScore should be a special function in registry or pipeline?
        # Or it can look at metrics already in shared_data.
        
        weights = {
            "sharpness": getattr(ctx.config, "quality_weights_sharpness", 0),
            "edge_strength": getattr(ctx.config, "quality_weights_edge_strength", 0),
            "contrast": getattr(ctx.config, "quality_weights_contrast", 0),
            "brightness": getattr(ctx.config, "quality_weights_brightness", 0),
            "entropy": getattr(ctx.config, "quality_weights_entropy", 0),
            "niqe": getattr(ctx.config, "quality_weights_niqe", 0),
        }
        
        # Pull normalized (0-1) scores from shared_data
        # We expect other operators to have put them there
        scores = ctx.shared_data.get("normalized_metrics", {})
        
        quality_sum = 0.0
        for name, weight in weights.items():
            score = scores.get(name, 0.0)
            quality_sum += score * (weight / 100.0)
            
        return OperatorResult(metrics={"quality_score": float(quality_sum * 100.0)})
