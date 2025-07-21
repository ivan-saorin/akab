"""Handler for revealing Level 3 experiment results"""
import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


async def reveal_experiment_handler(
    experiment_id: str,
    response_builder,
    reference_manager
) -> Dict[str, Any]:
    """Reveal experiment results if statistical significance is reached"""
    try:
        # Import here to avoid circular imports
        from ....server_fastmcp import SCRAMBLED_MODELS
        from ....core.vault import CampaignVault
        from ....core.laboratory import LABORATORY

        
        # Load experiment from vault
        vault = CampaignVault()
        campaign = await vault.load_campaign(experiment_id)
        
        if not campaign:
            return response_builder.error(f"Experiment '{experiment_id}' not found")
        
        if campaign.level != 3:
            return response_builder.error(f"'{experiment_id}' is not a Level 3 experiment")
        
        # Check if experiment has been executed
        if campaign.status != "completed":
            return response_builder.error(
                f"Experiment must be completed before reveal. Current status: {campaign.status}"
            )
        
        # Get results
        results = campaign.results
        if not results or "variants" not in results:
            return response_builder.error("No results found for this experiment")
        
        # Extract variant scores for statistical analysis
        variant_results = {}
        for variant_id, variant_data in results["variants"].items():
            if "raw_results" in variant_data:
                # Collect scores from all iterations
                scores = []
                for result in variant_data["raw_results"]:
                    if "metadata" in result and "composite_score" in result["metadata"]:
                        scores.append(result["metadata"]["composite_score"])
                if scores:
                    variant_results[variant_id] = scores
        
        if len(variant_results) < 2:
            return response_builder.error("Insufficient data for statistical analysis")
        
        # Check minimum iterations
        min_iterations = campaign.metadata.get("success_criteria", {}).get("minimum_iterations", 30)
        iterations_per_variant = {k: len(v) for k, v in variant_results.items()}
        min_completed = min(iterations_per_variant.values()) if iterations_per_variant else 0
        
        if min_completed < min_iterations:
            return response_builder.success(
                data={
                    "status": "insufficient_data",
                    "iterations_completed": min_completed,
                    "iterations_required": min_iterations,
                    "iterations_by_variant": iterations_per_variant,
                    "message": f"Need {min_iterations - min_completed} more iterations for statistical power"
                },
                message="Experiment not ready for reveal",
                suggestions=[
                    response_builder.suggest_next(
                        "akab_execute_campaign",
                        "Continue executing experiment",
                        {"campaign_id": experiment_id, "iterations": min_iterations - min_completed}
                    ),
                    response_builder.suggest_next(
                        "akab_diagnose_experiment",
                        "Diagnose why experiment isn't converging",
                        {"experiment_id": experiment_id}
                    )
                ]
            )
        
        # Run statistical significance test
        significance_result = LABORATORY.check_experiment_significance(
            variant_results,
            significance_level=campaign.metadata.get("success_criteria", {}).get("significance_level", 0.05),
            effect_size_threshold=campaign.metadata.get("success_criteria", {}).get("effect_size_threshold", 0.2)
        )
        
        # Check if we've reached significance
        if not significance_result.get("significant", False):
            return response_builder.success(
                data={
                    "status": "not_significant",
                    "iterations_completed": min_completed,
                    "p_value": significance_result.get("p_value", "N/A"),
                    "effect_size": significance_result.get("effect_size", "N/A"),
                    "message": "Statistical significance not reached",
                    "details": significance_result.get("details", "")
                },
                message="Experiment has not reached statistical significance",
                suggestions=[
                    response_builder.suggest_next(
                        "akab_execute_campaign",
                        "Run more iterations",
                        {"campaign_id": experiment_id, "iterations": 20}
                    ),
                    response_builder.suggest_next(
                        "akab_diagnose_experiment",
                        "Analyze why significance wasn't reached",
                        {"experiment_id": experiment_id}
                    )
                ]
            )
        
        # REVEAL TIME! Map scrambled IDs back to real models
        mappings = {}
        model_performance = {}
        
        for variant in campaign.variants:
            scrambled_id = variant.get("constraints", {}).get("scrambled_id")
            if scrambled_id and scrambled_id in SCRAMBLED_MODELS:
                real_model_id = SCRAMBLED_MODELS[scrambled_id]
                mappings[scrambled_id] = real_model_id
                
                # Get performance data
                variant_data = results["variants"].get(variant.get("id"), {})
                model_performance[real_model_id] = {
                    "scrambled_id": scrambled_id,
                    "mean_score": variant_data.get("average", 0),
                    "iterations": len(variant_results.get(variant.get("id"), [])),
                    "trimmed_mean": np.mean(np.sort(variant_results.get(variant.get("id"), []))[1:-1]) if len(variant_results.get(variant.get("id"), [])) > 2 else variant_data.get("average", 0)
                }
        
        # Find the winner
        winner_scrambled = significance_result.get("best_variant")
        winner_real = None
        if winner_scrambled:
            for variant in campaign.variants:
                if variant.get("id") == winner_scrambled:
                    scrambled_id = variant.get("constraints", {}).get("scrambled_id")
                    if scrambled_id in mappings:
                        winner_real = mappings[scrambled_id]
                        break
        
        # Update campaign with revealed status
        campaign.metadata["revealed"] = True
        campaign.metadata["reveal_timestamp"] = datetime.utcnow().isoformat()
        campaign.metadata["model_mappings"] = mappings
        await vault.save_campaign(campaign)
        
        return response_builder.success(
            data={
                "status": "revealed",
                "experiment_id": experiment_id,
                "winner": winner_real,
                "mappings": mappings,
                "model_performance": model_performance,
                "statistics": {
                    "p_value": significance_result.get("p_value"),
                    "effect_size": significance_result.get("effect_size"),
                    "confidence": f"{(1 - significance_result.get('p_value', 0.05)) * 100:.1f}%",
                    "test_used": significance_result.get("test", "Mann-Whitney U")
                },
                "hypothesis": campaign.metadata.get("hypothesis"),
                "conclusion": f"Model {winner_real} performed significantly better"
            },
            message="Experiment results revealed! Statistical significance achieved.",
            suggestions=[
                response_builder.suggest_next(
                    "akab_analyze_results",
                    "Get detailed analysis of the experiment",
                    {"campaign_id": experiment_id}
                ),
                response_builder.suggest_next(
                    "akab_cost_report",
                    "View cost breakdown by model",
                    {"campaign_id": experiment_id}
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Error revealing experiment: {e}", exc_info=True)
        return response_builder.error(f"Failed to reveal experiment: {str(e)}")
