"""Handler for diagnosing Level 3 experiments"""
import logging
from typing import Dict, Any, List
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


async def diagnose_experiment_handler(
    experiment_id: str,
    force_reveal: bool,
    response_builder,
    reference_manager
) -> Dict[str, Any]:
    """Diagnose why an experiment hasn't reached significance"""
    try:
        # Import here to avoid circular imports
        from ....server import SCRAMBLED_MODELS
        from ....core.vault import CampaignVault

        
        # Load experiment
        vault = CampaignVault()
        campaign = await vault.get_campaign(experiment_id)
        
        if not campaign:
            return response_builder.error(f"Experiment '{experiment_id}' not found")
        
        if campaign.level != 3:
            return response_builder.error(f"'{experiment_id}' is not a Level 3 experiment")
        
        # Get results
        results = campaign.results
        if not results or "variants" not in results:
            return response_builder.error("No results found. Execute the experiment first.")
        
        # Analyze each variant's performance
        variant_analysis = {}
        all_scores = []
        prompt_performance = {}
        
        for variant_id, variant_data in results["variants"].items():
            if "raw_results" not in variant_data:
                continue
                
            scores = []
            prompt_scores = {}
            
            for result in variant_data["raw_results"]:
                if "metadata" in result and "composite_score" in result["metadata"]:
                    score = result["metadata"]["composite_score"]
                    scores.append(score)
                    all_scores.append(score)
                    
                    # Track by prompt
                    prompt = result.get("prompt", "unknown")[:50]  # First 50 chars
                    if prompt not in prompt_scores:
                        prompt_scores[prompt] = []
                    prompt_scores[prompt].append(score)
            
            if scores:
                variant_analysis[variant_id] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "variance": np.var(scores),
                    "cv": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                    "n": len(scores),
                    "range": (min(scores), max(scores))
                }
                
                # Aggregate prompt performance
                for prompt, prompt_vals in prompt_scores.items():
                    if prompt not in prompt_performance:
                        prompt_performance[prompt] = {}
                    prompt_performance[prompt][variant_id] = {
                        "mean": np.mean(prompt_vals),
                        "std": np.std(prompt_vals),
                        "n": len(prompt_vals)
                    }
        
        # Calculate pairwise effect sizes
        effect_sizes = {}
        variant_ids = list(variant_analysis.keys())
        
        for i in range(len(variant_ids)):
            for j in range(i + 1, len(variant_ids)):
                v1, v2 = variant_ids[i], variant_ids[j]
                
                # Get scores for each variant
                scores1 = [r["metadata"]["composite_score"] 
                          for r in results["variants"][v1]["raw_results"]
                          if "metadata" in r and "composite_score" in r["metadata"]]
                scores2 = [r["metadata"]["composite_score"]
                          for r in results["variants"][v2]["raw_results"] 
                          if "metadata" in r and "composite_score" in r["metadata"]]
                
                if scores1 and scores2:
                    # Cohen's d
                    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                    if pooled_std > 0:
                        d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                        effect_sizes[f"{v1} vs {v2}"] = abs(d)
        
        # Identify issues
        issues = []
        recommendations = []
        
        # Check for high variance
        high_variance_variants = [
            v for v, stats in variant_analysis.items() 
            if stats["cv"] > 0.5  # Coefficient of variation > 50%
        ]
        if high_variance_variants:
            issues.append(f"High variance in: {', '.join(high_variance_variants)}")
            recommendations.append("Consider running more iterations to reduce variance")
        
        # Check for small effect sizes
        small_effects = [pair for pair, d in effect_sizes.items() if d < 0.2]
        if small_effects and len(small_effects) == len(effect_sizes):
            issues.append("All effect sizes are small (d < 0.2)")
            recommendations.append("Models may be too similar - consider testing more diverse variants")
        
        # Check for problematic prompts
        high_variance_prompts = []
        for prompt, variant_data in prompt_performance.items():
            variances = [data["std"] for data in variant_data.values() if data["n"] > 1]
            if variances and np.mean(variances) > np.mean([v["std"] for v in variant_analysis.values()]) * 1.5:
                high_variance_prompts.append(prompt)
        
        if high_variance_prompts:
            issues.append(f"High variance prompts: {len(high_variance_prompts)}")
            recommendations.append("Consider revising or removing high-variance prompts")
        
        # Check sample size
        min_n = min(stats["n"] for stats in variant_analysis.values())
        if min_n < 30:
            issues.append(f"Insufficient data: only {min_n} iterations per variant")
            recommendations.append(f"Run at least {30 - min_n} more iterations")
        
        # Prepare diagnosis
        diagnosis = {
            "status": "diagnosis_complete",
            "experiment_id": experiment_id,
            "issues_found": len(issues),
            "issues": issues,
            "recommendations": recommendations,
            "variant_analysis": variant_analysis,
            "effect_sizes": effect_sizes,
            "prompt_analysis": {
                "total_prompts": len(prompt_performance),
                "high_variance_prompts": len(high_variance_prompts)
            },
            "convergence": {
                "trending": "Not converging" if issues else "Converging well",
                "estimated_iterations_needed": max(30 - min_n, 0) if min_n < 30 else 
                                               (50 if small_effects else 20)
            }
        }
        
        # Handle force reveal
        if force_reveal:
            diagnosis["warning"] = "PROTOCOL BREACH: Force reveal breaks Level 3 blinding"
            diagnosis["revealed_mappings"] = {}
            
            for variant in campaign.variants:
                scrambled_id = variant.get("constraints", {}).get("scrambled_id")
                if scrambled_id and scrambled_id in SCRAMBLED_MODELS:
                    real_model_id = SCRAMBLED_MODELS[scrambled_id]
                    diagnosis["revealed_mappings"][scrambled_id] = real_model_id
        
        return response_builder.success(
            data=diagnosis,
            message=f"Diagnosis complete: {len(issues)} issues found",
            suggestions=[
                response_builder.suggest_next(
                    "akab_execute_campaign",
                    f"Run {diagnosis['convergence']['estimated_iterations_needed']} more iterations",
                    campaign_id=experiment_id, 
                    iterations=diagnosis['convergence']['estimated_iterations_needed']
                ),
                response_builder.suggest_next(
                    "akab_reveal_experiment",
                    "Try to reveal results",
                    experiment_id=experiment_id
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Error diagnosing experiment: {e}", exc_info=True)
        return response_builder.error(f"Failed to diagnose experiment: {str(e)}")
