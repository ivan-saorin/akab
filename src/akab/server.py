"""akab MCP Server Implementation"""

import os
import json
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from substrate import (
    SubstrateMCP, 
    ExecutionRequest,
    SamplingManager,
    ValidationError
)
from .campaigns import CampaignManager, Campaign
from .laboratory import Laboratory
from .blinded_hermes import BlindedHermes
from .clear_hermes import ClearHermes
from .judge import QualityJudge

# Configure logging
logger = logging.getLogger(__name__)


class AkabServer(SubstrateMCP):
    """A/B Testing Framework MCP Server"""
    
    def __init__(self):
        super().__init__(
            name="akab",
            version="1.0.0",
            description="A/B Testing Framework with scientific rigor",
            instructions="Use akab for pure A/B testing with blinded execution and statistical analysis."
        )
        
        self.data_dir = os.getenv("DATA_DIR", "./data")
        self.krill_dir = os.getenv("KRILL_DIR", "../krill/data")
        
        # Initialize components
        self.campaign_manager = CampaignManager(self.data_dir, self.krill_dir)
        self.laboratory = Laboratory()
        self.hermes = BlindedHermes(self.data_dir, self.krill_dir)  # For Level 2/3
        self.clear_hermes = ClearHermes()  # For Level 1
        self.sampling_manager = SamplingManager()
        self.quality_judge = QualityJudge(self.clear_hermes)  # Use clear hermes for judge
        
        # Valid providers and models
        self.valid_providers = ["anthropic", "openai"]
        self.default_models = {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4"
        }
        
        # Model size mappings
        self.model_sizes = {
            "anthropic": {
                "xs": "claude-3-haiku-20240307",
                "s": "claude-3-5-haiku-20241022", 
                "m": "claude-3-5-sonnet-20241022",
                "l": "claude-3-5-sonnet-20241022",  # Same as m for now
                "xl": "claude-3-opus-20240229"
            },
            "openai": {
                "xs": "gpt-3.5-turbo",
                "s": "gpt-4o-mini",
                "m": "gpt-4",
                "l": "gpt-4-turbo",
                "xl": "gpt-4-turbo"  # Same as l for now
            }
        }
        
        # Initialize Level 3 scrambled models (fire-and-forget)
        self.scrambled_models = self._initialize_scrambling()
        
        # Register tools
        self._register_tools()
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities"""
        return {
            "tools": [
                "akab_quick_compare",
                "akab_create_campaign", 
                "akab_execute_campaign",
                "akab_analyze_results",
                "akab_list_campaigns",
                "akab_cost_report",
                "akab_sampling_callback",
                "akab_unlock",
                "akab_list_scrambled_models",
                "akab_create_experiment",
                "akab_reveal_experiment",
                "akab_diagnose_experiment"
            ],
            "features": [
                "Blinded A/B testing",
                "Statistical analysis with trimmed means",
                "Campaign management",
                "Cost tracking",
                "Krill integration for results",
                "Intelligent constraint suggestions",
                "Progress tracking for long operations",
                "Three-level testing architecture",
                "Success criteria for campaigns",
                "Campaign unlocking (Level 2)",
                "Fire-and-forget model scrambling (Level 3)",
                "Scientific experiments with hypothesis testing"
            ],
            "providers": self.valid_providers,
            "default_models": self.default_models
        }
    
    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.tool(name="akab_quick_compare")
        async def quick_compare(ctx, prompt: str, providers: List[str] = None, 
                              constraints: Dict[str, Any] = None):
            """Quick comparison of prompt across multiple providers"""
            if providers is None:
                providers = ["anthropic", "openai"]
            
            return await self.quick_compare(prompt, providers, constraints)
        
        @self.tool(name="akab_create_campaign")
        async def create_campaign(ctx, name: str, description: str, variants: List[Dict[str, Any]],
                                success_criteria: Dict[str, Any] = None):
            """Create new A/B testing campaign with optional success criteria"""
            return await self.create_campaign(name, description, variants, success_criteria)
        
        @self.tool(name="akab_execute_campaign")
        async def execute_campaign(ctx, campaign_id: str, iterations: int = 1):
            """Execute A/B testing campaign"""
            return await self.execute_campaign(campaign_id, iterations)
        
        @self.tool(name="akab_analyze_results")
        async def analyze_results(ctx, campaign_id: str):
            """Analyze campaign results with statistical rigor"""
            return await self.analyze_results(campaign_id)
        
        @self.tool(name="akab_list_campaigns")
        async def list_campaigns(ctx, status: str = None):
            """List all A/B testing campaigns"""
            return await self.list_campaigns(status)
        
        @self.tool(name="akab_cost_report")
        async def cost_report(ctx, campaign_id: str = None):
            """Get cost report for campaigns"""
            return await self.cost_report(campaign_id)
        
        @self.tool(name="akab_sampling_callback")
        async def sampling_callback(ctx, request_id: str, response: str):
            """Handle sampling responses from Claude"""
            return await self.sampling_manager.handle_callback(request_id, response)
        
        @self.tool(name="akab_unlock")
        async def akab_unlock(ctx, id: str):
            """Unlock campaign or experiment to reveal mappings and archive"""
            return await self.unlock(id)
        
        @self.tool(name="akab_list_scrambled_models")
        async def list_scrambled_models(ctx):
            """List available scrambled model IDs for Level 3 experiments"""
            return await self.list_scrambled_models()
        
        @self.tool(name="akab_create_experiment")
        async def create_experiment(ctx, name: str, description: str, hypothesis: str,
                                  variants: List[str], prompts: List[str],
                                  iterations_per_prompt: int = 10,
                                  success_criteria: Dict[str, Any] = None):
            """Create Level 3 scientific experiment with complete blinding"""
            return await self.create_experiment(name, description, hypothesis, variants,
                                              prompts, iterations_per_prompt, success_criteria)
        
        @self.tool(name="akab_reveal_experiment")
        async def reveal_experiment(ctx, experiment_id: str):
            """Reveal Level 3 experiment results after statistical significance"""
            return await self.reveal_experiment(experiment_id)
        
        @self.tool(name="akab_diagnose_experiment")
        async def diagnose_experiment(ctx, experiment_id: str, force_reveal: bool = False):
            """Diagnose why a Level 3 experiment hasn't reached significance"""
            return await self.diagnose_experiment(experiment_id, force_reveal)
    
    async def quick_compare(self, prompt: str, providers: List[str], 
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Level 1: Quick comparison with NO blinding, immediate results.
        For debugging, exploration, and human judgment.
        """
        
        # Validate providers
        invalid_providers = [p for p in providers if p not in self.valid_providers]
        if invalid_providers:
            raise ValidationError(
                f"Unknown providers: {invalid_providers}",
                field="providers",
                suggestions=self.valid_providers
            )
        
        # If no constraints provided, offer intelligent assistance
        if not constraints and self.sampling_manager.should_request_sampling("constraints"):
            sampling_request = self.sampling_manager.create_request(
                f"User wants to A/B test this prompt: '{prompt[:100]}...' "
                f"across {providers}. Suggest appropriate constraints "
                "(max_tokens, temperature, etc.) for a fair comparison. "
                "Consider the nature of the prompt and provide specific values.",
                context={"prompt": prompt, "providers": providers}
            )
            
            return self.create_response(
                data={
                    "status": "awaiting_constraints",
                    "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "providers": providers
                },
                message="Would you like constraint suggestions for this comparison?",
                _sampling_request=sampling_request
            )
        
        # Execute Level 1 comparison with progress tracking
        async with self.progress_context("quick_compare") as progress:
            await progress(0.1, "Starting Level 1 quick comparison...")
            
            # Execute tests directly without campaign storage
            results_by_provider = {}
            errors_by_provider = {}
            
            total_providers = len(providers)
            for idx, provider in enumerate(providers):
                await progress(
                    0.1 + (0.8 * idx / total_providers),
                    f"Testing {provider}..."
                )
                
                try:
                    # Create execution request
                    model = self.default_models.get(provider)
                    request = ExecutionRequest(
                        prompt=prompt,
                        model_id=model,
                        model_name=model,
                        parameters={
                            "provider": provider,
                            "level": 1,  # Mark as Level 1
                            "quick_compare": True
                        },
                        constraints=constraints or {"max_tokens": 1000, "temperature": 0.7}
                    )
                    
                    # Execute with ClearHermes (no blinding)
                    result = await self.clear_hermes.execute(request)
                    
                    if result.response:
                        results_by_provider[provider] = {
                            "response": result.response,
                            "model": model,
                            "execution_time": result.execution_time,
                            "tokens_used": result.tokens_used,
                            "cost": result.cost,
                            "metadata": result.metadata
                        }
                    else:
                        errors_by_provider[provider] = result.error or "No response"
                        
                except Exception as e:
                    errors_by_provider[provider] = str(e)
            
            await progress(0.9, "Preparing results...")
            
            # Format Level 1 results - NO WINNER SELECTION
            response_data = {
                "level": 1,
                "prompt": prompt,
                "results": results_by_provider,
                "errors": errors_by_provider,
                "constraints_used": constraints or {"max_tokens": 1000, "temperature": 0.7}
            }
            
            # Add comparison summary
            if results_by_provider:
                summary = {
                    "providers_tested": list(results_by_provider.keys()),
                    "fastest": min(results_by_provider.items(), 
                                  key=lambda x: x[1]["execution_time"])[0],
                    "cheapest": min(results_by_provider.items(), 
                                   key=lambda x: x[1]["cost"])[0]
                }
                response_data["summary"] = summary
            
            await progress(1.0, "Complete!")
            
            # Return Level 1 response with clear provider names
            return self.create_response(
                data=response_data,
                message="Level 1 quick comparison complete. Judge the results yourself!",
                annotations={
                    "priority": 0.8,
                    "tone": "informative",
                    "visualization": "side_by_side_comparison"
                }
            )
    
    async def create_campaign(self, name: str, description: str, 
                            variants: List[Dict[str, Any]],
                            success_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create new Level 2 campaign with success criteria"""
        try:
            # Validate variants
            if len(variants) < 2:
                raise ValidationError(
                    "A/B test requires at least 2 variants",
                    field="variants",
                    suggestions=["Add more variants to compare"]
                )
            
            required_fields = ["id", "provider", "model", "prompt"]
            for variant in variants:
                missing = [f for f in required_fields if f not in variant]
                if missing:
                    raise ValidationError(
                        f"Variant '{variant.get('id', 'unknown')}' missing fields: {missing}",
                        field="variant",
                        suggestions=required_fields
                    )
                
                if variant["provider"] not in self.valid_providers:
                    raise ValidationError(
                        f"Unknown provider '{variant['provider']}' in variant '{variant['id']}'",
                        field="provider",
                        suggestions=self.valid_providers
                    )
            
            # Validate success criteria if provided
            if success_criteria:
                # Ensure required fields
                if "primary_metric" not in success_criteria:
                    success_criteria["primary_metric"] = "execution_time"
                if "direction" not in success_criteria:
                    success_criteria["direction"] = "minimize"
                if "evaluation_method" not in success_criteria:
                    success_criteria["evaluation_method"] = "statistical"
            
            campaign = await self.campaign_manager.create_campaign(
                name, description, variants, success_criteria, level=2
            )
            
            return self.create_response(
                data={
                    "campaign": campaign.to_dict(),
                    "status": "created"
                },
                message=f"Campaign '{name}' created successfully"
            )
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def reveal_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Reveal Level 3 experiment results after verification"""
        try:
            campaign = await self.campaign_manager.get_campaign(experiment_id)
            
            if not campaign:
                raise ValidationError(
                    f"Experiment not found: {experiment_id}",
                    field="experiment_id"
                )
            
            # Verify it's a Level 3 experiment
            campaign_level = getattr(campaign, 'level', 2)
            is_experiment = campaign.metadata.get("is_experiment", False)
            
            if campaign_level != 3 and not is_experiment:
                return self.create_error_response(
                    f"This is a Level {campaign_level} campaign, not a Level 3 experiment"
                )
            
            # Check if already revealed
            if campaign.metadata.get("revealed", False):
                return self.create_response(
                    data={
                        "experiment_id": experiment_id,
                        "status": "already_revealed",
                        "revealed_at": campaign.metadata.get("revealed_at"),
                        "winner": campaign.metadata.get("winner"),
                        "mappings": campaign.metadata.get("revealed_mappings")
                    },
                    message="Experiment already revealed"
                )
            
            # Check completion criteria
            success_criteria = campaign.success_criteria or {}
            min_iterations = success_criteria.get("minimum_iterations", 30)
            total_tests = len(campaign.results)
            
            if total_tests < min_iterations:
                return self.create_error_response(
                    f"Experiment incomplete: {total_tests}/{min_iterations} tests completed",
                    suggestions=[f"Run {min_iterations - total_tests} more tests"]
                )
            
            # Perform quality evaluation if enabled
            variant_quality_scores = {}
            if success_criteria.get("evaluation_method") == "llm_judge":
                try:
                    # Group results by variant for quality evaluation
                    prompts = campaign.metadata.get("prompts", [])
                    quality_scores = await self.quality_judge.evaluate_campaign_results(
                        campaign.results, prompts
                    )
                    variant_quality_scores = quality_scores
                except Exception as e:
                    # Log but don't fail the reveal
                    print(f"Quality evaluation failed: {e}")
            
            # Calculate significance with quality scores or execution times
            variant_results = {}
            for variant in campaign.variants:
                variant_id = variant["id"]
                
                if variant_quality_scores and variant_id in variant_quality_scores:
                    # Use quality scores if available
                    variant_results[variant_id] = variant_quality_scores[variant_id]
                else:
                    # Fall back to execution times
                    variant_times = [
                        r["execution_time"] for r in campaign.results 
                        if r.get("variant") == variant_id and r.get("success", False)
                    ]
                    if variant_times:
                        variant_results[variant_id] = variant_times
            
            # Check statistical significance
            significance_level = success_criteria.get("significance_level", 0.05)
            effect_size_threshold = success_criteria.get("effect_size_threshold", 0.2)
            
            significance_check = self.laboratory.check_experiment_significance(
                variant_results,
                significance_level,
                effect_size_threshold
            )
            
            if not significance_check["significant"]:
                return self.create_error_response(
                    f"No statistically significant difference found: {significance_check['reason']}",
                    suggestions=[
                        "Run more iterations",
                        "Check if variants are sufficiently different",
                        "Consider adjusting significance criteria"
                    ],
                    data={
                        "comparisons": significance_check.get("comparisons", []),
                        "current_results": {k: len(v) for k, v in variant_results.items()}
                    }
                )
            
            # Reveal the mappings!
            revealed_mappings = {}
            for variant_id in variant_results.keys():
                if variant_id in self.scrambled_models:
                    model_info = self.scrambled_models[variant_id]
                    revealed_mappings[variant_id] = {
                        "provider": model_info["provider"],
                        "size": model_info["size"],
                        "model": model_info["model"]
                    }
            
            # Mark as revealed
            campaign.metadata["revealed"] = True
            campaign.metadata["revealed_at"] = time.time()
            campaign.metadata["winner"] = significance_check["best_variant"]
            campaign.metadata["revealed_mappings"] = revealed_mappings
            campaign.metadata["significance_results"] = significance_check
            
            await self.campaign_manager._save_campaign(campaign)
            
            # Prepare detailed results
            winner_mapping = revealed_mappings.get(significance_check["best_variant"], {})
            
            return self.create_response(
                data={
                    "experiment_id": experiment_id,
                    "hypothesis": campaign.metadata.get("hypothesis"),
                    "winner": {
                        "scrambled_id": significance_check["best_variant"],
                        "provider": winner_mapping.get("provider"),
                        "model": winner_mapping.get("model"),
                        "size": winner_mapping.get("size")
                    },
                    "all_mappings": revealed_mappings,
                    "statistical_results": significance_check,
                    "total_tests": total_tests,
                    "evaluation_method": success_criteria.get("evaluation_method", "execution_time")
                },
                message=f"Experiment complete! Winner: {winner_mapping.get('provider', 'Unknown')} {winner_mapping.get('size', '')} ({significance_check['best_variant']})",
                annotations={
                    "priority": 0.95,
                    "tone": "celebratory",
                    "visualization": "experiment_results"
                }
            )
            
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def diagnose_experiment(self, experiment_id: str, force_reveal: bool = False) -> Dict[str, Any]:
        """Diagnose why a Level 3 experiment hasn't reached significance"""
        try:
            campaign = await self.campaign_manager.get_campaign(experiment_id)
            
            if not campaign:
                raise ValidationError(
                    f"Experiment not found: {experiment_id}",
                    field="experiment_id"
                )
            
            # Verify it's a Level 3 experiment
            campaign_level = getattr(campaign, 'level', 2)
            is_experiment = campaign.metadata.get("is_experiment", False)
            
            if campaign_level != 3 and not is_experiment:
                return self.create_error_response(
                    f"This is a Level {campaign_level} campaign, not a Level 3 experiment"
                )
            
            # Check if already revealed
            if campaign.metadata.get("revealed", False) and not force_reveal:
                return self.create_response(
                    data={
                        "experiment_id": experiment_id,
                        "status": "already_revealed",
                        "note": "Experiment was already revealed. Use force_reveal=True to re-diagnose."
                    }
                )
            
            # Gather diagnostic data
            success_criteria = campaign.success_criteria or {}
            total_tests = len(campaign.results)
            prompts = campaign.metadata.get("prompts", [])
            
            # Group results by variant for analysis
            variant_results = {}
            variant_response_samples = {}
            for variant in campaign.variants:
                variant_id = variant["id"]
                variant_data = [
                    r for r in campaign.results 
                    if r.get("variant") == variant_id
                ]
                
                # Get execution times
                variant_times = [
                    r["execution_time"] for r in variant_data
                    if r.get("success", False)
                ]
                if variant_times:
                    variant_results[variant_id] = variant_times
                
                # Get sample responses for diagnostic
                sample_responses = []
                for prompt_idx in range(min(3, len(prompts))):  # First 3 prompts
                    prompt_responses = [
                        r for r in variant_data 
                        if r.get("prompt_index") == prompt_idx and r.get("success", False)
                    ]
                    if prompt_responses and prompt_responses[0].get("response"):
                        sample_responses.append({
                            "prompt_index": prompt_idx,
                            "prompt_preview": prompts[prompt_idx][:100] + "...",
                            "response_preview": prompt_responses[0]["response"][:200] + "...",
                            "response_length": len(prompt_responses[0]["response"])
                        })
                variant_response_samples[variant_id] = sample_responses
            
            # Perform statistical analysis
            significance_level = success_criteria.get("significance_level", 0.05)
            effect_size_threshold = success_criteria.get("effect_size_threshold", 0.2)
            
            significance_check = self.laboratory.check_experiment_significance(
                variant_results,
                significance_level,
                effect_size_threshold
            )
            
            # Calculate detailed diagnostics
            diagnostics = {
                "experiment_status": {
                    "total_tests_run": total_tests,
                    "tests_per_variant": {k: len(v) for k, v in variant_results.items()},
                    "success_rate": sum(1 for r in campaign.results if r.get("success", False)) / max(1, total_tests),
                    "prompts_tested": len(prompts)
                },
                "statistical_analysis": significance_check,
                "issues_detected": []
            }
            
            # Diagnose common issues
            issues = diagnostics["issues_detected"]
            
            # 1. Insufficient sample size
            min_iterations = success_criteria.get("minimum_iterations", 30)
            if total_tests < min_iterations:
                issues.append({
                    "type": "insufficient_data",
                    "severity": "high",
                    "description": f"Only {total_tests}/{min_iterations} tests completed",
                    "recommendation": f"Run at least {min_iterations - total_tests} more tests"
                })
            
            # 2. High variance in results
            for variant_id, times in variant_results.items():
                if times:
                    stats = self.laboratory.calculate_statistics(times)
                    cv = stats["std_dev"] / stats["mean"] if stats["mean"] > 0 else 0
                    if cv > 0.5:  # Coefficient of variation > 50%
                        issues.append({
                            "type": "high_variance",
                            "severity": "medium",
                            "variant": variant_id,
                            "coefficient_of_variation": round(cv, 3),
                            "description": f"High variance in {variant_id} results (CV={cv:.1%})",
                            "recommendation": "Consider more iterations or check for outliers"
                        })
            
            # 3. Models too similar
            if significance_check.get("comparisons"):
                all_effect_sizes = [
                    abs(comp["effect_size"]["cohens_d"]) 
                    for comp in significance_check["comparisons"]
                ]
                max_effect = max(all_effect_sizes) if all_effect_sizes else 0
                
                if max_effect < 0.2:
                    issues.append({
                        "type": "models_too_similar",
                        "severity": "high",
                        "max_effect_size": round(max_effect, 3),
                        "description": "Models performing too similarly (negligible effect sizes)",
                        "recommendation": "Test more diverse model sizes or different prompt types"
                    })
            
            # 4. Prompt-specific issues
            prompt_performance = {}
            for prompt_idx in range(len(prompts)):
                prompt_results = [r for r in campaign.results if r.get("prompt_index") == prompt_idx]
                if prompt_results:
                    success_rate = sum(1 for r in prompt_results if r.get("success", False)) / len(prompt_results)
                    prompt_performance[prompt_idx] = {
                        "prompt_preview": prompts[prompt_idx][:100] + "...",
                        "success_rate": success_rate,
                        "total_tests": len(prompt_results)
                    }
                    
                    if success_rate < 0.8:
                        issues.append({
                            "type": "prompt_failures",
                            "severity": "medium",
                            "prompt_index": prompt_idx,
                            "success_rate": round(success_rate, 3),
                            "description": f"Low success rate for prompt {prompt_idx}",
                            "recommendation": "Review prompt for clarity or constraints"
                        })
            
            diagnostics["prompt_performance"] = prompt_performance
            
            # Add response samples for qualitative review
            diagnostics["response_samples"] = variant_response_samples
            
            # Force reveal handling
            if force_reveal:
                # Add warning about breaking scientific protocol
                diagnostics["force_reveal_warning"] = (
                    "WARNING: Force revealing without significance breaks the double-blind protocol. "
                    "Results may be biased by knowledge of model identities."
                )
                
                # Reveal mappings
                revealed_mappings = {}
                for variant_id in variant_results.keys():
                    if variant_id in self.scrambled_models:
                        model_info = self.scrambled_models[variant_id]
                        revealed_mappings[variant_id] = {
                            "provider": model_info["provider"],
                            "size": model_info["size"],
                            "model": model_info["model"]
                        }
                
                diagnostics["force_revealed_mappings"] = revealed_mappings
                
                # Mark as diagnostically revealed
                campaign.metadata["diagnostic_reveal"] = True
                campaign.metadata["diagnostic_reveal_at"] = time.time()
                await self.campaign_manager._save_campaign(campaign)
            
            # Generate summary
            summary = "Experiment diagnostic complete. "
            if not significance_check["significant"]:
                summary += "No significant differences found. "
            
            critical_issues = [i for i in issues if i["severity"] == "high"]
            if critical_issues:
                summary += f"{len(critical_issues)} critical issues detected."
            else:
                summary += "Consider running more iterations or testing more diverse models."
            
            return self.create_response(
                data=diagnostics,
                message=summary,
                annotations={
                    "priority": 0.8,
                    "tone": "analytical",
                    "visualization": "diagnostic_report"
                }
            )
            
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def execute_campaign(self, campaign_id: str, iterations: int) -> Dict[str, Any]:
        """Execute campaign with progress tracking"""
        try:
            campaign = await self.campaign_manager.get_campaign(campaign_id)
            
            if not campaign:
                raise ValidationError(
                    f"Campaign not found: {campaign_id}",
                    field="campaign_id",
                    suggestions=["Check campaign ID", "List campaigns with akab_list_campaigns"]
                )
            
            async with self.progress_context(f"campaign_{campaign_id}") as progress:
                result = await self._execute_campaign_with_progress(
                    campaign_id, iterations, progress, 0, 1
                )
                return result
                
        except ValidationError:
            raise
        except Exception as e:
            # Offer error recovery assistance
            if self.sampling_manager.should_request_sampling("error_recovery"):
                sampling_request = self.sampling_manager.create_request(
                    f"Campaign execution failed with error: {str(e)}. "
                    "How should we proceed? Options include retrying, "
                    "skipping failed variants, or adjusting parameters.",
                    context={"error": str(e), "campaign_id": campaign_id}
                )
                
                return self.create_error_response(
                    error=str(e),
                    suggestions=["retry", "skip_failed_variants", "adjust_parameters"],
                    _sampling_request=sampling_request
                )
            else:
                return self.create_error_response(str(e))
    
    async def _execute_campaign_with_progress(self, campaign_id: str, iterations: int,
                                            progress_callback: Callable, 
                                            start_progress: float, end_progress: float) -> Dict[str, Any]:
        """Execute campaign with sub-progress tracking"""
        campaign = await self.campaign_manager.get_campaign(campaign_id)
        
        # Check if this is a Level 3 experiment
        is_experiment = campaign.metadata.get("is_experiment", False)
        campaign_level = getattr(campaign, 'level', 2)
        
        if is_experiment or campaign_level == 3:
            # Handle Level 3 experiments differently
            return await self._execute_experiment_with_progress(
                campaign, iterations, progress_callback, start_progress, end_progress
            )
        
        # Original Level 2 campaign execution
        # Create blinding mapping
        variant_ids = [v["id"] for v in campaign.variants]
        blinding_map = self.laboratory.create_blind_mapping(variant_ids, campaign_id)
        
        # Calculate progress increments
        total_tests = len(campaign.variants) * iterations
        progress_range = end_progress - start_progress
        completed = 0
        
        # Execute all tests with concurrency limit
        all_tasks = []
        task_metadata = []
        
        for iteration in range(iterations):
            for variant in campaign.variants:
                task = self._execute_single_test(
                    campaign_id, variant, iteration, blinding_map
                )
                all_tasks.append(task)
                task_metadata.append({
                    "iteration": iteration,
                    "variant_id": variant["id"],
                    "blinded_id": blinding_map[variant["id"]]
                })
        
        # Execute with concurrency limit
        results = await self._gather_with_limit(all_tasks, limit=10)
        
        # Process results and update progress
        successful_results = []
        for i, (result, metadata) in enumerate(zip(results, task_metadata)):
            completed += 1
            current_progress = start_progress + (completed / total_tests) * progress_range
            
            await progress_callback(
                current_progress,
                f"Completed {completed}/{total_tests} tests "
                f"(iteration {metadata['iteration'] + 1}/{iterations})"
            )
            
            if not isinstance(result, Exception):
                successful_results.append({
                    "variant": metadata["blinded_id"],
                    "iteration": metadata["iteration"],
                    "success": True
                })
            else:
                successful_results.append({
                    "variant": metadata["blinded_id"],
                    "iteration": metadata["iteration"],
                    "success": False,
                    "error": str(result)
                })
        
        return self.create_response(
            data={
                "campaign_id": campaign_id,
                "iterations": iterations,
                "total_tests": total_tests,
                "successful_tests": sum(1 for r in successful_results if r["success"]),
                "results": successful_results
            },
            message=f"Executed {total_tests} tests across {iterations} iterations"
        )
    
    async def _execute_experiment_with_progress(self, campaign: Campaign, iterations: int,
                                              progress_callback: Callable,
                                              start_progress: float, end_progress: float) -> Dict[str, Any]:
        """Execute Level 3 experiment with multiple prompts"""
        campaign_id = campaign.id
        prompts = campaign.metadata.get("prompts", [])
        iterations_per_prompt = campaign.metadata.get("iterations_per_prompt", iterations)
        
        # For experiments, iterations parameter is ignored in favor of iterations_per_prompt
        actual_iterations = min(iterations, iterations_per_prompt)
        
        # Create blinding mapping (for Level 3, this is just for tracking)
        variant_ids = [v["id"] for v in campaign.variants]
        blinding_map = {v_id: v_id for v_id in variant_ids}  # No additional blinding for L3
        
        # Calculate total tests
        total_tests = len(campaign.variants) * len(prompts) * actual_iterations
        progress_range = end_progress - start_progress
        completed = 0
        
        # Execute all tests
        all_tasks = []
        task_metadata = []
        
        for prompt_idx, prompt in enumerate(prompts):
            for iteration in range(actual_iterations):
                for variant in campaign.variants:
                    # Get actual model info from scrambled ID
                    model_info = self.scrambled_models.get(variant["id"], {})
                    
                    # Create a modified variant with the current prompt
                    variant_with_prompt = {
                        **variant,
                        "prompt": prompt,
                        "provider": model_info.get("provider", "unknown"),
                        "model": model_info.get("model", variant["id"]),
                        "model_name": model_info.get("model", variant["id"])
                    }
                    
                    task = self._execute_single_test(
                        campaign_id, variant_with_prompt, iteration, blinding_map,
                        extra_metadata={"prompt_index": prompt_idx}
                    )
                    all_tasks.append(task)
                    task_metadata.append({
                        "iteration": iteration,
                        "variant_id": variant["id"],
                        "blinded_id": variant["id"],  # No additional blinding
                        "prompt_index": prompt_idx
                    })
        
        # Execute with concurrency limit
        results = await self._gather_with_limit(all_tasks, limit=5)  # Lower limit for experiments
        
        # Process results
        successful_results = []
        for i, (result, metadata) in enumerate(zip(results, task_metadata)):
            completed += 1
            current_progress = start_progress + (completed / total_tests) * progress_range
            
            await progress_callback(
                current_progress,
                f"Completed {completed}/{total_tests} tests "
                f"(prompt {metadata['prompt_index'] + 1}/{len(prompts)})"
            )
            
            if not isinstance(result, Exception):
                successful_results.append({
                    "variant": metadata["blinded_id"],
                    "iteration": metadata["iteration"],
                    "prompt_index": metadata["prompt_index"],
                    "success": True
                })
            else:
                successful_results.append({
                    "variant": metadata["blinded_id"],
                    "iteration": metadata["iteration"],
                    "prompt_index": metadata["prompt_index"],
                    "success": False,
                    "error": str(result)
                })
        
        return self.create_response(
            data={
                "experiment_id": campaign_id,
                "iterations": actual_iterations,
                "prompts_tested": len(prompts),
                "total_tests": total_tests,
                "successful_tests": sum(1 for r in successful_results if r["success"]),
                "results": successful_results
            },
            message=f"Executed {total_tests} tests: {len(campaign.variants)} models × {len(prompts)} prompts × {actual_iterations} iterations"
        )
    
    async def _execute_single_test(self, campaign_id: str, variant: Dict[str, Any],
                                  iteration: int, blinding_map: Dict[str, str],
                                  extra_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single test variant"""
        try:
            # Create execution request
            request = ExecutionRequest(
                prompt=variant["prompt"],
                model_id=variant["model"],
                model_name=variant.get("model_name", variant["model"]),
                parameters={
                    "provider": variant["provider"],
                    "campaign_id": campaign_id,
                    "variant_id": variant["id"],
                    "iteration": iteration,
                    **(extra_metadata or {})
                },
                constraints={
                    "temperature": variant.get("constraints", {}).get("temperature", variant.get("temperature", 0.7)),
                    "max_tokens": variant.get("constraints", {}).get("max_tokens", variant.get("max_tokens", 1000))
                }
            )
            
            # Execute with blinding
            result = await self.hermes.execute(request)
            
            # Store result WITH ACTUAL RESPONSE!
            result_data = {
                "variant": blinding_map[variant["id"]],
                "iteration": iteration,
                "response": result.response,  # CRITICAL: Store the actual response!
                "response_sanitized": result.response,  # Already sanitized by BlindedHermes
                "response_length": len(result.response) if result.response else 0,
                "execution_time": result.execution_time,
                "tokens_used": result.tokens_used,
                "cost": result.cost,
                "success": not bool(result.error),
                "error": result.error,
                "timestamp": time.time(),
                "metadata": result.metadata
            }
            
            # Add extra metadata if provided
            if extra_metadata:
                result_data.update(extra_metadata)
            
            await self.campaign_manager.add_result(campaign_id, result_data)
            
            return result
            
        except Exception as e:
            # Store failed result
            result_data = {
                "variant": blinding_map[variant["id"]],
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
            
            if extra_metadata:
                result_data.update(extra_metadata)
                
            await self.campaign_manager.add_result(campaign_id, result_data)
            return e
    
    async def _gather_with_limit(self, tasks: List, limit: int = 10):
        """Execute tasks with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def bounded_task(task):
            async with semaphore:
                try:
                    return await task
                except Exception as e:
                    return e
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        return await asyncio.gather(*bounded_tasks)
    
    async def analyze_results(self, campaign_id: str) -> Dict[str, Any]:
        """Analyze campaign results with rich annotations"""
        try:
            campaign = await self.campaign_manager.get_campaign(campaign_id)
            
            if not campaign:
                raise ValidationError(
                    f"Campaign not found: {campaign_id}",
                    field="campaign_id",
                    suggestions=["Check campaign ID", "List campaigns with akab_list_campaigns"]
                )
            
            analysis = await self._analyze_results_internal(campaign)
            
            # Determine annotations based on results
            if analysis.get("clear_winner"):
                annotations = {
                    "priority": 0.9,
                    "tone": "confident",
                    "visualization": "winner_highlight"
                }
                message = f"Clear winner: {analysis['winner']} - {analysis['conclusion']}"
            elif analysis.get("insufficient_data"):
                annotations = {
                    "priority": 0.5,
                    "tone": "cautious",
                    "visualization": "need_more_data"
                }
                message = "Insufficient data for conclusive results"
            else:
                annotations = {
                    "priority": 0.7,
                    "tone": "analytical",
                    "visualization": "detailed_comparison"
                }
                message = analysis.get("conclusion", "Analysis complete")
            
            # Mark as completed if enough results
            if len(campaign.results) >= len(campaign.variants) * 10:
                await self.campaign_manager.complete_campaign(campaign_id)
            
            return self.create_response(
                data=analysis,
                message=message,
                annotations=annotations
            )
            
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def unlock(self, id: str) -> Dict[str, Any]:
        """Unlock campaign or experiment to reveal mappings with krill archiving"""
        try:
            # Try to get as campaign first
            campaign = await self.campaign_manager.get_campaign(id)
            
            if not campaign:
                raise ValidationError(
                    f"Campaign or experiment not found: {id}",
                    field="id"
                )
            
            # Get campaign level and check if it's an experiment
            campaign_level = getattr(campaign, 'level', 2)
            is_experiment = campaign.metadata.get("is_experiment", False)
            
            # Handle based on type
            if campaign_level == 3 or is_experiment:
                # This is a Level 3 experiment
                return await self._unlock_experiment(campaign)
            elif campaign_level == 2:
                # This is a regular Level 2 campaign
                return await self._unlock_campaign(campaign)
            else:
                # Level 1 campaigns don't need unlocking
                return self.create_response(
                    data={"error": f"Level {campaign_level} campaigns don't support unlocking"},
                    success=False
                )
            
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def _unlock_campaign(self, campaign: Campaign) -> Dict[str, Any]:
        """Unlock a Level 2 campaign"""
        campaign_id = campaign.id
        
        # Check if already unlocked
        if campaign.metadata.get('unlocked', False):
            return self.create_response(
                data={"error": "Campaign is already unlocked"},
                success=False
            )
        
        # Get blinding mappings before unlocking
        variant_ids = [v["id"] for v in campaign.variants]
        blinding_map = self.laboratory.create_blind_mapping(variant_ids, campaign_id)
        
        # Create archive in krill
        archive_metadata = {}
        try:
            import shutil
            from pathlib import Path
            
            # Create campaign archive directory in krill
            krill_archive_dir = Path(self.krill_dir) / "archive" / campaign_id
            krill_archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Get campaign directory path
            campaign_source_dir = Path(self.data_dir) / "campaigns" / "standard"
            campaign_file = campaign_source_dir / f"{campaign_id}.json"
            
            # Copy blinded state (before unlock)
            blinded_dir = krill_archive_dir / "blinded"
            blinded_dir.mkdir(exist_ok=True)
            
            if campaign_file.exists():
                shutil.copy2(campaign_file, blinded_dir / "campaign.json")
            
            # Also copy results if they exist
            results_dir = Path(self.data_dir) / "results" / campaign_id
            if results_dir.exists():
                shutil.copytree(results_dir, blinded_dir / "results", dirs_exist_ok=True)
            
            # Now unlock the campaign
            campaign.metadata["unlocked"] = True
            campaign.metadata["unlocked_at"] = time.time()
            campaign.metadata["blinding_map"] = blinding_map
            
            # Save the updated campaign
            await self.campaign_manager._save_campaign(campaign)
            
            # Copy clear state (after unlock)
            clear_dir = krill_archive_dir / "clear"
            clear_dir.mkdir(exist_ok=True)
            
            # Copy the now-unlocked campaign file
            shutil.copy2(campaign_file, clear_dir / "campaign.json")
            
            # Copy results again for clear version
            if results_dir.exists():
                shutil.copytree(results_dir, clear_dir / "results", dirs_exist_ok=True)
            
            # Create archive metadata
            archive_metadata = {
                "campaign_id": campaign_id,
                "campaign_name": campaign.name,
                "level": 2,
                "unlocked_at": campaign.metadata["unlocked_at"],
                "blinding_map": blinding_map,
                "archive_created": time.time(),
                "paths": {
                    "blinded": str(blinded_dir),
                    "clear": str(clear_dir)
                }
            }
            
            # Save archive metadata
            with open(krill_archive_dir / "metadata.json", 'w') as f:
                json.dump(archive_metadata, f, indent=2)
            
        except Exception as archive_error:
            # Log but don't fail the unlock
            logger.warning(f"Krill archive failed: {archive_error}", exc_info=True)
            # Store error in metadata for client to see
            archive_metadata = {
                "error": str(archive_error),
                "error_type": type(archive_error).__name__
            }
        
        # Create provider mapping details for response
        mappings = {}
        for variant in campaign.variants:
            variant_id = variant["id"]
            blinded_id = blinding_map[variant_id]
            mappings[variant_id] = {
                "blinded_id": blinded_id,
                "provider": variant["provider"],
                "model": variant["model"],
                "prompt": variant["prompt"][:50] + "..." if len(variant["prompt"]) > 50 else variant["prompt"]
            }
        
        response_data = {
            "id": campaign_id,
            "name": campaign.name,
            "type": "campaign",
            "level": 2,
            "mappings": mappings,
            "blinding_map": blinding_map,
            "unlocked_at": campaign.metadata["unlocked_at"]
        }
        
        # Add archive info
        if "error" not in archive_metadata:
            response_data["archive"] = {
                "status": "success",
                "location": f"/krill/archive/{campaign_id}",
                "created_at": archive_metadata["archive_created"]
            }
        else:
            response_data["archive"] = {
                "status": "failed",
                "error": archive_metadata["error"]
            }
        
        return self.create_response(
            data=response_data,
            message="Campaign unlocked! Provider mappings revealed and archived to krill."
        )
    
    async def _unlock_experiment(self, experiment: Campaign) -> Dict[str, Any]:
        """Unlock a Level 3 experiment (only if complete)"""
        experiment_id = experiment.id
        
        # Check if already revealed/unlocked
        if experiment.metadata.get("revealed", False):
            # Experiment was revealed, now unlock it
            return await self._perform_experiment_unlock(experiment)
        else:
            # Experiment not yet revealed
            success_criteria = experiment.success_criteria or {}
            min_iterations = success_criteria.get("minimum_iterations", 30)
            total_tests = len(experiment.results)
            
            if total_tests < min_iterations:
                return self.create_error_response(
                    f"Cannot unlock ongoing experiment: {total_tests}/{min_iterations} tests completed",
                    suggestions=[
                        f"Complete {min_iterations - total_tests} more tests",
                        "Use akab_reveal_experiment when ready",
                        "Use akab_diagnose_experiment to check status"
                    ]
                )
            else:
                return self.create_error_response(
                    "Experiment complete but not revealed. Use akab_reveal_experiment first.",
                    suggestions=[
                        "Run akab_reveal_experiment to check statistical significance",
                        "If no significance found, use akab_diagnose_experiment"
                    ]
                )
    
    async def _perform_experiment_unlock(self, experiment: Campaign) -> Dict[str, Any]:
        """Actually unlock and archive a revealed experiment"""
        experiment_id = experiment.id
        
        # Check if already unlocked
        if experiment.metadata.get("unlocked", False):
            return self.create_response(
                data={"error": "Experiment is already unlocked"},
                success=False
            )
        
        # Get all the mappings (from reveal and scrambling)
        revealed_mappings = experiment.metadata.get("revealed_mappings", {})
        winner = experiment.metadata.get("winner")
        
        # Create archive in krill
        archive_metadata = {}
        try:
            import shutil
            from pathlib import Path
            
            # Create experiment archive directory in krill
            krill_archive_dir = Path(self.krill_dir) / "archive" / experiment_id
            krill_archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Get experiment directory path
            experiment_source_dir = Path(self.data_dir) / "campaigns" / "experiments"
            experiment_file = experiment_source_dir / f"{experiment_id}.json"
            
            # Copy blinded state (before adding full mappings)
            blinded_dir = krill_archive_dir / "blinded"
            blinded_dir.mkdir(exist_ok=True)
            
            # Create a blinded version without revealed mappings
            blinded_experiment = experiment.to_dict()
            blinded_experiment["metadata"].pop("revealed_mappings", None)
            blinded_experiment["metadata"].pop("winner", None)
            
            with open(blinded_dir / "experiment.json", 'w') as f:
                json.dump(blinded_experiment, f, indent=2)
            
            # Copy results
            results_dir = Path(self.data_dir) / "results" / experiment_id
            if results_dir.exists():
                shutil.copytree(results_dir, blinded_dir / "results", dirs_exist_ok=True)
            
            # Now unlock the experiment
            experiment.metadata["unlocked"] = True
            experiment.metadata["unlocked_at"] = time.time()
            
            # Save the updated experiment
            await self.campaign_manager._save_campaign(experiment)
            
            # Copy clear state (with all mappings)
            clear_dir = krill_archive_dir / "clear"
            clear_dir.mkdir(exist_ok=True)
            
            # Copy the full experiment with mappings
            if experiment_file.exists():
                shutil.copy2(experiment_file, clear_dir / "experiment.json")
            
            # Copy results again
            if results_dir.exists():
                shutil.copytree(results_dir, clear_dir / "results", dirs_exist_ok=True)
            
            # Create comprehensive archive metadata
            archive_metadata = {
                "experiment_id": experiment_id,
                "experiment_name": experiment.name,
                "level": 3,
                "hypothesis": experiment.metadata.get("hypothesis"),
                "revealed_at": experiment.metadata.get("revealed_at"),
                "unlocked_at": experiment.metadata["unlocked_at"],
                "winner": winner,
                "revealed_mappings": revealed_mappings,
                "total_tests": len(experiment.results),
                "archive_created": time.time(),
                "paths": {
                    "blinded": str(blinded_dir),
                    "clear": str(clear_dir)
                }
            }
            
            # Save archive metadata
            with open(krill_archive_dir / "metadata.json", 'w') as f:
                json.dump(archive_metadata, f, indent=2)
            
        except Exception as archive_error:
            # Log but don't fail the unlock
            logger.warning(f"Krill archive failed: {archive_error}", exc_info=True)
            archive_metadata = {"error": str(archive_error)}
        
        response_data = {
            "id": experiment_id,
            "name": experiment.name,
            "type": "experiment",
            "level": 3,
            "hypothesis": experiment.metadata.get("hypothesis"),
            "winner": winner,
            "revealed_mappings": revealed_mappings,
            "unlocked_at": experiment.metadata["unlocked_at"]
        }
        
        # Add archive info
        if "error" not in archive_metadata:
            response_data["archive"] = {
                "status": "success",
                "location": f"/krill/archive/{experiment_id}",
                "created_at": archive_metadata["archive_created"]
            }
        else:
            response_data["archive"] = {
                "status": "failed",
                "error": archive_metadata["error"]
            }
        
        return self.create_response(
            data=response_data,
            message="Experiment unlocked! All mappings revealed and archived to krill."
        )
    
    def create_error_response(self, error: str, suggestions: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Create an error response with optional suggestions"""
        return self.create_response(
            success=False,
            error=error,
            suggestions=suggestions,
            **kwargs
        )
    
    async def _analyze_results_internal(self, campaign: Campaign) -> Dict[str, Any]:
        """Internal analysis logic with Level 3 support"""
        # Check campaign level
        campaign_level = getattr(campaign, 'level', 2)
        is_experiment = campaign.metadata.get("is_experiment", False)
        
        # Get blinding map (for L3, variant IDs are already scrambled)
        variant_ids = [v["id"] for v in campaign.variants]
        if campaign_level == 3 or is_experiment:
            # For Level 3, no additional blinding - IDs are already scrambled
            blinding_map = {v_id: v_id for v_id in variant_ids}
            reverse_map = blinding_map
        else:
            # Level 1/2 use regular blinding
            blinding_map = self.laboratory.create_blind_mapping(variant_ids, campaign.id)
            reverse_map = {v: k for k, v in blinding_map.items()}
        
        # Group results by variant
        variant_results = {}
        for variant_id in variant_ids:
            blinded_id = blinding_map[variant_id]
            variant_results[variant_id] = [
                r for r in campaign.results if r.get("variant") == blinded_id
            ]
        
        # Calculate statistics for each variant
        variant_stats = {}
        for variant_id, results in variant_results.items():
            if results:
                # Extract metrics
                execution_times = [r["execution_time"] for r in results if r.get("success", False)]
                costs = [r["cost"] for r in results if r.get("success", False)]
                tokens = [r["tokens_used"] for r in results if r.get("success", False)]
                
                # For Level 3, also group by prompt
                if campaign_level == 3 or is_experiment:
                    prompt_stats = {}
                    prompts = campaign.metadata.get("prompts", [])
                    for prompt_idx in range(len(prompts)):
                        prompt_results = [r for r in results if r.get("prompt_index") == prompt_idx]
                        if prompt_results:
                            prompt_times = [r["execution_time"] for r in prompt_results if r.get("success", False)]
                            if prompt_times:
                                prompt_stats[prompt_idx] = {
                                    "n": len(prompt_times),
                                    "mean": sum(prompt_times) / len(prompt_times),
                                    "prompt_preview": prompts[prompt_idx][:50] + "..."
                                }
                
                if execution_times:
                    stats = self.laboratory.calculate_statistics(execution_times)
                    variant_stats[variant_id] = {
                        "variant_id": variant_id,
                        "total_tests": len(results),
                        "successful_tests": len(execution_times),
                        "failure_rate": (len(results) - len(execution_times)) / len(results),
                        "execution_time": stats,
                        "cost": {
                            "mean": sum(costs) / len(costs) if costs else 0,
                            "total": sum(costs)
                        },
                        "tokens": {
                            "mean": sum(tokens) / len(tokens) if tokens else 0,
                            "total": sum(tokens)
                        }
                    }
                    
                    # Add prompt-specific stats for Level 3
                    if campaign_level == 3 or is_experiment:
                        variant_stats[variant_id]["prompt_stats"] = prompt_stats
        
        # Determine winner based on success criteria
        winner = None
        conclusion = ""
        
        # Get success criteria
        success_criteria = getattr(campaign, 'success_criteria', None)
        if success_criteria and len(variant_stats) >= 2:
            primary_metric = success_criteria.get("primary_metric", "execution_time")
            direction = success_criteria.get("direction", "minimize")
            evaluation_method = success_criteria.get("evaluation_method", "statistical")
            
            # For Level 3 with quality evaluation
            if (campaign_level == 3 or is_experiment) and evaluation_method == "llm_judge":
                # Try to get quality scores
                try:
                    prompts = campaign.metadata.get("prompts", [])
                    quality_scores = await self.quality_judge.evaluate_campaign_results(
                        campaign.results, prompts
                    )
                    
                    # Add quality scores to variant stats
                    for variant_id, scores in quality_scores.items():
                        if variant_id in variant_stats and scores:
                            quality_stats = self.laboratory.calculate_statistics(scores)
                            variant_stats[variant_id]["quality_score"] = quality_stats
                            variant_stats[variant_id]["quality_scores_raw"] = scores
                    
                    # Sort by quality score
                    sorted_variants = sorted(
                        [(k, v) for k, v in variant_stats.items() if "quality_score" in v],
                        key=lambda x: x[1]["quality_score"]["trimmed_mean"],
                        reverse=True  # Higher quality is better
                    )
                    
                    if sorted_variants:
                        winner = sorted_variants[0][0]
                        best_quality = sorted_variants[0][1]["quality_score"]["trimmed_mean"]
                        worst_quality = sorted_variants[-1][1]["quality_score"]["trimmed_mean"]
                        
                        if worst_quality > 0:
                            improvement = ((best_quality - worst_quality) / worst_quality) * 100
                            conclusion = f"{winner} has {improvement:.1f}% higher quality score"
                        else:
                            conclusion = f"{winner} has the highest quality score: {best_quality:.1f}"
                except Exception as e:
                    # Fall back to execution time
                    print(f"Quality evaluation failed in analysis: {e}")
                    primary_metric = "execution_time"
            
            # Default metric-based selection
            if not winner and primary_metric == "execution_time":
                sorted_variants = sorted(
                    variant_stats.items(),
                    key=lambda x: x[1]["execution_time"]["trimmed_mean"],
                    reverse=(direction == "maximize")
                )
                
                if sorted_variants:
                    winner = sorted_variants[0][0]
                    best_value = sorted_variants[0][1]["execution_time"]["trimmed_mean"]
                    worst_value = sorted_variants[-1][1]["execution_time"]["trimmed_mean"]
                    
                    if direction == "minimize":
                        improvement = ((worst_value - best_value) / worst_value) * 100
                        conclusion = f"{winner} is {improvement:.1f}% faster"
                    else:
                        improvement = ((best_value - worst_value) / worst_value) * 100
                        conclusion = f"{winner} is {improvement:.1f}% better"
        
        # Prepare return data
        analysis_data = {
            "variant_results": variant_stats,
            "winner": winner,
            "clear_winner": winner is not None,
            "conclusion": conclusion,
            "insufficient_data": len(campaign.results) < len(campaign.variants) * 3,
            "success_criteria": success_criteria
        }
        
        # Handle blinding based on level
        if campaign_level == 1:
            # Level 1: No blinding, always show mappings
            analysis_data["blinding_map"] = blinding_map
        elif campaign_level == 2:
            # Level 2: Only show if unlocked
            if hasattr(campaign, 'metadata') and campaign.metadata.get('unlocked', False):
                analysis_data["blinding_map"] = blinding_map
            else:
                analysis_data["blinding_locked"] = True
        elif campaign_level == 3 or is_experiment:
            # Level 3: Never show mappings until experiment complete
            analysis_data["experiment_blinded"] = True
            analysis_data["level"] = 3
            
            # Check if experiment meets completion criteria
            total_tests = len(campaign.results)
            min_iterations = success_criteria.get("minimum_iterations", 30) if success_criteria else 30
            
            if total_tests >= min_iterations:
                analysis_data["completion_status"] = {
                    "minimum_iterations_met": True,
                    "total_tests": total_tests,
                    "required_tests": min_iterations,
                    "ready_for_reveal": True
                }
            else:
                analysis_data["completion_status"] = {
                    "minimum_iterations_met": False,
                    "total_tests": total_tests,
                    "required_tests": min_iterations,
                    "tests_remaining": min_iterations - total_tests
                }
        
        return analysis_data
    
    async def list_campaigns(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List campaigns with summaries"""
        try:
            campaigns = await self.campaign_manager.list_campaigns(status)
            
            # Create summaries
            summaries = []
            for campaign in campaigns:
                summaries.append({
                    "id": campaign.id,
                    "name": campaign.name,
                    "status": campaign.status,
                    "level": getattr(campaign, 'level', 2),
                    "created_at": campaign.created_at,
                    "variants": len(campaign.variants),
                    "results": len(campaign.results),
                    "estimated_cost": self._estimate_campaign_cost(campaign)
                })
            
            return self.create_response(
                data={
                    "campaigns": summaries,
                    "total": len(summaries),
                    "by_status": {
                        "active": sum(1 for c in summaries if c["status"] == "active"),
                        "completed": sum(1 for c in summaries if c["status"] == "completed"),
                        "cancelled": sum(1 for c in summaries if c["status"] == "cancelled")
                    },
                    "by_level": {
                        "level_1": sum(1 for c in summaries if c["level"] == 1),
                        "level_2": sum(1 for c in summaries if c["level"] == 2),
                        "level_3": sum(1 for c in summaries if c["level"] == 3)
                    }
                },
                message=f"Found {len(summaries)} campaigns"
            )
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def cost_report(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate detailed cost report"""
        try:
            if campaign_id:
                campaign = await self.campaign_manager.get_campaign(campaign_id)
                if not campaign:
                    raise ValidationError(
                        f"Campaign not found: {campaign_id}",
                        field="campaign_id"
                    )
                campaigns = [campaign]
            else:
                campaigns = await self.campaign_manager.list_campaigns()
            
            total_cost = 0.0
            campaign_reports = []
            
            for campaign in campaigns:
                report = self._calculate_campaign_costs(campaign)
                campaign_reports.append(report)
                total_cost += report["total_cost"]
            
            return self.create_response(
                data={
                    "total_cost": round(total_cost, 4),
                    "campaigns": campaign_reports,
                    "cost_by_provider": self._aggregate_costs_by_provider(campaign_reports)
                },
                message=f"Total cost across {len(campaigns)} campaigns: ${total_cost:.4f}"
            )
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
    
    def _estimate_campaign_cost(self, campaign: Campaign) -> float:
        """Estimate campaign cost from results"""
        total_cost = sum(r.get("cost", 0) for r in campaign.results if r.get("success", False))
        return round(total_cost, 4)
    
    def _calculate_campaign_costs(self, campaign: Campaign) -> Dict[str, Any]:
        """Calculate detailed costs for a campaign"""
        variant_costs = {}
        total_cost = 0.0
        
        # Get blinding map
        variant_ids = [v["id"] for v in campaign.variants]
        blinding_map = self.laboratory.create_blind_mapping(variant_ids, campaign.id)
        reverse_map = {v: k for k, v in blinding_map.items()}
        
        for result in campaign.results:
            if result.get("success", False) and "cost" in result:
                cost = result["cost"]
                total_cost += cost
                
                # Find original variant
                blinded_id = result.get("variant")
                if blinded_id in reverse_map:
                    variant_id = reverse_map[blinded_id]
                    variant_costs[variant_id] = variant_costs.get(variant_id, 0) + cost
        
        return {
            "campaign_id": campaign.id,
            "campaign_name": campaign.name,
            "total_cost": round(total_cost, 4),
            "variant_costs": {k: round(v, 4) for k, v in variant_costs.items()},
            "cost_per_iteration": round(total_cost / max(1, len(campaign.results)), 4)
        }
    
    def _aggregate_costs_by_provider(self, campaign_reports: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate costs by provider across campaigns"""
        # This would need campaign variant info to properly aggregate
        # For now, return placeholder
        return {
            "anthropic": sum(r["total_cost"] * 0.6 for r in campaign_reports),
            "openai": sum(r["total_cost"] * 0.4 for r in campaign_reports)
        }
    
    def _initialize_scrambling(self) -> Dict[str, Dict[str, Any]]:
        """Initialize or load fire-and-forget model scrambling for Level 3"""
        import hashlib
        import random
        from datetime import datetime
        
        # Ensure scrambling directory exists
        scrambling_dir = os.path.join(self.krill_dir, "scrambling")
        os.makedirs(scrambling_dir, exist_ok=True)
        
        # Look for existing session scrambling
        existing_files = [f for f in os.listdir(scrambling_dir) if f.startswith("session_")]
        
        if existing_files:
            # Use the most recent scrambling
            latest_file = sorted(existing_files)[-1]
            filepath = os.path.join(scrambling_dir, latest_file)
            with open(filepath, 'r') as f:
                return json.load(f)
        
        # Create new scrambling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scrambled = {}
        
        # Generate scrambled IDs for all provider/size combinations
        for provider in self.valid_providers:
            for size in ["xs", "s", "m", "l", "xl"]:
                # Generate deterministic but unguessable ID
                seed = f"{provider}_{size}_{timestamp}_{random.randint(1000, 9999)}"
                model_id = "model_" + hashlib.sha256(seed.encode()).hexdigest()[:6]
                
                scrambled[model_id] = {
                    "provider": provider,
                    "size": size,
                    "model": self.model_sizes[provider][size],
                    "created_at": timestamp
                }
        
        # Save scrambling
        filepath = os.path.join(scrambling_dir, f"session_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(scrambled, f, indent=2)
        
        return scrambled
    
    async def list_scrambled_models(self) -> Dict[str, Any]:
        """List available scrambled model IDs for Level 3 experiments"""
        try:
            # Group by approximate capability
            grouped = {
                "small": [],
                "medium": [],
                "large": []
            }
            
            for model_id, info in self.scrambled_models.items():
                size = info["size"]
                if size in ["xs", "s"]:
                    grouped["small"].append(model_id)
                elif size in ["m"]:
                    grouped["medium"].append(model_id)
                else:  # l, xl
                    grouped["large"].append(model_id)
            
            return self.create_response(
                data={
                    "scrambled_models": grouped,
                    "total_models": len(self.scrambled_models),
                    "usage": "Use these model IDs for Level 3 experiments only",
                    "warning": "Model mappings are irreversibly scrambled until experiment completion"
                },
                message=f"Found {len(self.scrambled_models)} scrambled models for Level 3 experiments"
            )
        except Exception as e:
            return self.create_error_response(str(e))
    
    async def create_experiment(self, name: str, description: str, hypothesis: str,
                              variants: List[str], prompts: List[str],
                              iterations_per_prompt: int = 10,
                              success_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create Level 3 scientific experiment with complete blinding"""
        try:
            # Validate all variants are scrambled model IDs
            invalid_variants = [v for v in variants if v not in self.scrambled_models]
            if invalid_variants:
                available = list(self.scrambled_models.keys())[:5]  # Show first 5
                raise ValidationError(
                    f"Invalid scrambled model IDs: {invalid_variants}",
                    field="variants",
                    suggestions=["Use akab_list_scrambled_models to see available IDs"] + available
                )
            
            # Level 3 requires enhanced success criteria
            if not success_criteria:
                success_criteria = {
                    "primary_metric": "quality_score",
                    "secondary_metrics": ["execution_time", "cost"],
                    "direction": "maximize",
                    "evaluation_method": "llm_judge",
                    "significance_level": 0.05,
                    "effect_size_threshold": 0.2,
                    "minimum_iterations": 30
                }
            else:
                # Ensure required fields for Level 3
                if "significance_level" not in success_criteria:
                    success_criteria["significance_level"] = 0.05
                if "minimum_iterations" not in success_criteria:
                    success_criteria["minimum_iterations"] = 30
            
            # Create experiment structure
            experiment_id = f"experiment_{int(time.time() * 1000)}"
            
            # Create campaign-like structure for compatibility
            # But with experiment-specific fields
            campaign_variants = []
            for variant_id in variants:
                model_info = self.scrambled_models[variant_id]
                campaign_variants.append({
                    "id": variant_id,
                    "provider": "unknown",  # Hide provider
                    "model": variant_id,  # Use scrambled ID as model
                    "prompts": prompts,  # Multiple prompts
                    "iterations_per_prompt": iterations_per_prompt
                })
            
            # Store as Level 3 campaign
            campaign = await self.campaign_manager.create_campaign(
                name=name,
                description=f"{description}\n\nHypothesis: {hypothesis}",
                variants=campaign_variants,
                success_criteria=success_criteria,
                level=3
            )
            
            # Add experiment-specific metadata
            campaign.metadata["hypothesis"] = hypothesis
            campaign.metadata["experiment_id"] = experiment_id
            campaign.metadata["prompts"] = prompts
            campaign.metadata["iterations_per_prompt"] = iterations_per_prompt
            campaign.metadata["total_iterations"] = len(prompts) * iterations_per_prompt * len(variants)
            campaign.metadata["is_experiment"] = True  # Flag for special handling
            
            await self.campaign_manager._save_campaign(campaign)
            
            return self.create_response(
                data={
                    "experiment_id": campaign.id,
                    "name": name,
                    "hypothesis": hypothesis,
                    "variants": variants,
                    "prompts_count": len(prompts),
                    "iterations_per_prompt": iterations_per_prompt,
                    "total_tests": campaign.metadata["total_iterations"],
                    "success_criteria": success_criteria,
                    "status": "created"
                },
                message=f"Level 3 experiment '{name}' created with {len(variants)} models and {len(prompts)} prompts"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            return self.create_error_response(str(e))
