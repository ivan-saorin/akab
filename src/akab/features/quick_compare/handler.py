"""Quick Compare Business Logic"""
import asyncio
import time
import yaml
from typing import Dict, Any, List, Optional
from substrate.shared.models import get_model_registry
from substrate.shared.api import ClearHermes


class QuickCompareHandler:
    """Handler for Level 1 quick comparisons"""
    
    def __init__(self, response_builder, reference_manager):
        self.response_builder = response_builder  # From substrate
        self.reference_manager = reference_manager  # From substrate
        self.hermes = ClearHermes()  # From substrate!
    
    async def compare(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute quick comparison across providers"""
        
        # Default constraints
        constraints = constraints or {}
        max_tokens = constraints.get("max_tokens", 1000)
        temperature = constraints.get("temperature", 0.7)
        models_per_provider = constraints.get("models_per_provider", 1)
        
        # Get available providers
        configured_providers = self.hermes.get_configured_providers()
        
        # Use requested providers or all configured ones
        if providers:
            # Validate requested providers
            providers = [p for p in providers if p in configured_providers]
            if not providers:
                return self.response_builder.error(
                    f"None of the requested providers are configured. Available: {configured_providers}"
                )
        else:
            providers = configured_providers
        
        if not providers:
            return self.response_builder.error(
                "No providers configured. Please set API keys for at least one provider."
            )
        
        # Select models (default to medium size)
        models = []
        registry = get_model_registry()
        for provider in providers:
            # Try different sizes in order of preference
            for size in ["m", "l", "s", "xl"]:
                model_id = f"{provider}_{size}"
                model = registry.get(model_id)
                if model:
                    models.append(model)
                    break
                    
        if not models:
            return self.response_builder.error(
                "No models found for the configured providers"
            )
        
        # Execute comparisons in parallel
        tasks = []
        for model in models:
            task = self._execute_single_test(
                model, prompt, max_tokens, temperature
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        comparison_data = {}
        successful_results = []
        failed_results = []
        
        for model, result in zip(models, results):
            model_id = model.identifier
            if isinstance(result, Exception):
                failed_results.append(model_id)
                comparison_data[model_id] = {
                    "error": str(result),
                    "provider": model.provider.value,
                    "model": model.api_name
                }
            else:
                successful_results.append((model_id, result))
                comparison_data[model_id] = {
                    "response": result["content"],
                    "tokens": result["tokens"],
                    "latency": result["latency"],
                    "provider": result["provider"],
                    "model": result["model"]
                }
        
        # Find best performer (by speed/token ratio)
        best_performer = None
        best_score = -1
        
        for model_id, result in successful_results:
            # Simple scoring: tokens per second
            score = result["tokens"] / max(result["latency"], 0.1)
            if score > best_score:
                best_score = score
                best_performer = model_id
        
        # Save results for future reference
        timestamp = int(time.time())
        ref_result = await self.reference_manager.create_ref(
            f"quick_compare/{timestamp}",
            yaml.dump({
                "prompt": prompt,
                "constraints": constraints,
                "results": comparison_data,
                "best_performer": best_performer,
                "summary": {
                    "tested_models": len(models),
                    "successful": len(successful_results),
                    "failed": len(failed_results)
                }
            }),
            metadata={"type": "quick_compare", "timestamp": timestamp}
        )
        ref = ref_result["ref"]
        
        # Build response with suggestions
        return self.response_builder.success(
            data={
                "comparison_results": comparison_data,
                "best_performer": best_performer,
                "reference": ref
            },
            message=f"Compared {len(models)} models. {len(successful_results)} succeeded.",
            suggestions=self._build_suggestions(
                prompt, best_performer, successful_results, failed_results
            )
        )
    
    async def _execute_single_test(
        self, model, prompt: str, max_tokens: int, temperature: float
    ) -> Dict[str, Any]:
        """Execute a single model test"""
        return await self.hermes.complete(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def _build_suggestions(
        self, prompt: str, best_performer: str, 
        successful_results: List, failed_results: List
    ) -> List[Dict[str, Any]]:
        """Build contextual suggestions based on results"""
        suggestions = []
        
        # If we found a best performer, suggest enhancement
        if best_performer:
            suggestions.append(
                self.response_builder.suggest_next(
                    "synapse_enhance_prompt",
                    f"Enhance prompt for {best_performer}",
                    prompt=prompt,
                    model=best_performer
                )
            )
        
        # Suggest creating a campaign for deeper testing
        if len(successful_results) > 1:
            suggestions.append(
                self.response_builder.suggest_next(
                    "akab_create_campaign",
                    "Create campaign for statistical testing",
                    base_prompt=prompt,
                    models=[{"provider": r[0].split("_")[0], "size": r[0].split("_")[1]} 
                           for r in successful_results[:3]]  # Top 3 models
                )
            )
        
        # If some providers failed, suggest checking API keys
        if failed_results:
            failed_providers = list(set(m.split("_")[0] for m in failed_results))
            suggestions.append(
                self.response_builder.suggest_next(
                    "atlas_documentation",
                    f"Check setup for failed providers: {', '.join(failed_providers)}",
                    doc_type="setup-guide"
                )
            )
        
        return suggestions
