"""Comparison engine for AKAB - handles A/B testing logic."""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, List, Optional

# Import substrate components
try:
    from substrate import ComparisonResult, Timer, estimate_progress
except ImportError:
    try:
        from .substrate import ComparisonResult, Timer, estimate_progress
    except ImportError:
        from .substrate_stub import ComparisonResult, Timer, estimate_progress

from .providers import ProviderManager

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """Engine for comparing AI outputs across providers."""
    
    def __init__(self, provider_manager: ProviderManager):
        """Initialize comparison engine.
        
        Args:
            provider_manager: Provider manager instance
        """
        self.provider_manager = provider_manager
        
    async def compare(
        self,
        prompt: str,
        providers: List[str],
        *,
        parameters: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[ComparisonResult]:
        """Compare prompt across multiple providers.
        
        Args:
            prompt: The prompt to test
            providers: List of provider names
            parameters: Optional prompt parameters to interpolate
            constraints: Optional constraints (max_tokens, temperature, etc.)
            progress_callback: Optional progress callback
            
        Returns:
            List of comparison results
        """
        # Prepare prompt with parameters
        final_prompt = self._prepare_prompt(prompt, parameters or {})
        
        # Extract constraints
        max_tokens = constraints.get("max_tokens", 1000) if constraints else 1000
        temperature = constraints.get("temperature", 0.7) if constraints else 0.7
        
        results = []
        total_providers = len(providers)
        
        # Run comparisons concurrently
        tasks = []
        for i, provider_name in enumerate(providers):
            task = self._run_single_comparison(
                provider_name,
                final_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                progress_info=(i, total_providers, progress_callback)
            )
            tasks.append(task)
            
        # Gather results
        comparison_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, (provider_name, result) in enumerate(zip(providers, comparison_results)):
            if isinstance(result, Exception):
                # Handle provider errors
                results.append(
                    ComparisonResult(
                        provider=provider_name,
                        response="",
                        latency_ms=0,
                        error=str(result),
                        metadata={"error_type": type(result).__name__}
                    )
                )
            else:
                results.append(result)
                
        # Final progress update
        if progress_callback:
            await progress_callback(0.95, "Comparisons complete")
            
        return results
        
    async def _run_single_comparison(
        self,
        provider_name: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        progress_info: tuple
    ) -> ComparisonResult:
        """Run comparison for a single provider.
        
        Args:
            provider_name: Name of the provider
            prompt: The prompt to test
            max_tokens: Maximum tokens
            temperature: Temperature setting
            progress_info: Progress tracking info
            
        Returns:
            Comparison result
        """
        i, total, progress_callback = progress_info
        
        try:
            # Get provider
            provider = await self.provider_manager.get_provider(provider_name)
            
            # Update progress
            if progress_callback:
                progress = estimate_progress(i, total, 0.1, 0.8)
                await progress_callback(
                    progress,
                    f"Testing {provider_name}..."
                )
                
            # Run completion with timing
            with Timer() as timer:
                response = await provider.complete(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
            # Extract response data
            text = response.get("text", "")
            usage = response.get("usage", {})
            
            # Calculate cost
            config = self.provider_manager.get_provider_config(provider_name)
            cost = self.provider_manager.estimate_cost(
                provider_name,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0)
            )
            
            return ComparisonResult(
                provider=provider_name,
                response=text,
                latency_ms=timer.elapsed_ms,
                tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                cost_estimate=cost,
                metadata={
                    "model": config.model,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            
        except Exception as e:
            logger.error(f"Error in {provider_name} comparison: {e}")
            raise
            
    def _prepare_prompt(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Prepare prompt by interpolating parameters.
        
        Args:
            prompt: Prompt template
            parameters: Parameters to interpolate
            
        Returns:
            Prepared prompt
        """
        if not parameters:
            return prompt
            
        # Simple template substitution
        # In production, use a proper template engine
        result = prompt
        for key, value in parameters.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
            
        return result
        
    def analyze_results(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze comparison results.
        
        Args:
            results: List of comparison results
            
        Returns:
            Analysis dict with metrics and insights
        """
        if not results:
            return {"error": "No results to analyze"}
            
        # Filter successful results
        successful = [r for r in results if not r.error]
        
        if not successful:
            return {
                "error": "All providers failed",
                "failures": [
                    {"provider": r.provider, "error": r.error}
                    for r in results
                ]
            }
            
        # Calculate metrics
        metrics = {
            "fastest_provider": min(successful, key=lambda r: r.latency_ms).provider,
            "cheapest_provider": min(
                successful, 
                key=lambda r: r.cost_estimate or float('inf')
            ).provider,
            "average_latency_ms": sum(r.latency_ms for r in successful) / len(successful),
            "total_cost": sum(r.cost_estimate or 0 for r in successful),
            "response_lengths": {
                r.provider: len(r.response)
                for r in successful
            },
            "success_rate": len(successful) / len(results),
        }
        
        # Determine winner (simple heuristic - can be customized)
        # Score based on: speed (40%), cost (40%), response quality (20%)
        scores = {}
        
        for result in successful:
            speed_score = 1.0 - (result.latency_ms / max(r.latency_ms for r in successful))
            cost_score = 1.0 - (
                (result.cost_estimate or 0) / 
                max(r.cost_estimate or 0.001 for r in successful)
            )
            # Simple quality metric - response length normalized
            quality_score = min(1.0, len(result.response) / 500)
            
            scores[result.provider] = (
                speed_score * 0.4 +
                cost_score * 0.4 +
                quality_score * 0.2
            )
            
        winner = max(scores.items(), key=lambda x: x[1])[0]
        
        # Generate summary
        summary_parts = [
            f"{winner} performed best overall.",
            f"Fastest: {metrics['fastest_provider']} "
            f"({min(r.latency_ms for r in successful):.0f}ms).",
            f"Cheapest: {metrics['cheapest_provider']} "
            f"(${min(r.cost_estimate or 0 for r in successful):.4f}).",
        ]
        
        if len(successful) < len(results):
            failed_count = len(results) - len(successful)
            summary_parts.append(f"{failed_count} provider(s) failed.")
            
        return {
            "winner": winner,
            "metrics": metrics,
            "scores": scores,
            "summary": " ".join(summary_parts),
            "details": {
                "successful_providers": [r.provider for r in successful],
                "failed_providers": [r.provider for r in results if r.error],
            }
        }
        
    async def compare_batch(
        self,
        prompts: List[Dict[str, Any]],
        providers: List[str],
        *,
        constraints: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, List[ComparisonResult]]:
        """Compare multiple prompts across providers.
        
        Args:
            prompts: List of prompt dicts with 'prompt' and optional 'parameters'
            providers: List of provider names
            constraints: Optional constraints
            progress_callback: Optional progress callback
            
        Returns:
            Dict mapping prompt IDs to comparison results
        """
        results = {}
        total_prompts = len(prompts)
        
        for i, prompt_data in enumerate(prompts):
            prompt_id = prompt_data.get("id", f"prompt_{i}")
            prompt = prompt_data["prompt"]
            parameters = prompt_data.get("parameters", {})
            
            # Update progress
            if progress_callback:
                progress = estimate_progress(i, total_prompts)
                await progress_callback(
                    progress,
                    f"Testing prompt {i+1}/{total_prompts}"
                )
                
            # Run comparison for this prompt
            prompt_results = await self.compare(
                prompt,
                providers,
                parameters=parameters,
                constraints=constraints
            )
            
            results[prompt_id] = prompt_results
            
        return results
