"""Blinded API wrapper for Level 2/3 testing - AKAB-specific"""
import logging
import time
from typing import Dict, Any, Optional
from substrate.shared.api import ClearHermes
from substrate.shared.models import get_model_registry

logger = logging.getLogger(__name__)


class BlindedHermes:
    """Blinded API calls for Level 2/3 testing
    
    This is AKAB-specific functionality that provides blinding
    by mapping variant IDs to actual models. It delegates the
    actual API calls to substrate's ClearHermes.
    """
    
    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        """Initialize with variant-to-model mapping
        
        Args:
            mapping: Dict mapping variant IDs to model identifiers
                    e.g., {"variant_a": "anthropic_m", "variant_b": "openai_l"}
        """
        self.mapping = mapping or {}
        self.clear_hermes = ClearHermes()  # Delegate to substrate
        self.call_log = []  # Track calls for debugging
    
    def set_mapping(self, mapping: Dict[str, str]):
        """Update the variant mapping"""
        self.mapping = mapping
        logger.info(f"Updated mapping with {len(mapping)} variants")
    
    async def complete(
        self, 
        variant_id: str, 
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make blinded completion call
        
        Args:
            variant_id: Blinded variant identifier (e.g., "variant_a")
            prompt: The prompt to send
            **kwargs: Additional parameters for the API call
            
        Returns:
            API response WITHOUT revealing the actual model used
        """
        # Resolve variant to actual model
        model_id = self.mapping.get(variant_id)
        if not model_id:
            raise ValueError(
                f"Unknown variant: {variant_id}. "
                f"Available variants: {list(self.mapping.keys())}"
            )
        
        # Parse model ID (format: provider_size)
        try:
            registry = get_model_registry()
            model = registry.get(model_id)
            if not model:
                raise ValueError(f"Model not found in registry: {model_id}")
        except ValueError as e:
            raise ValueError(f"Invalid model ID format: {model_id}") from e
        
        # Log the call (for campaign tracking)
        self.call_log.append({
            "variant_id": variant_id,
            "timestamp": time.time(),
            "prompt_length": len(prompt)
        })
        
        # Use substrate's ClearHermes for actual API call
        result = await self.clear_hermes.complete(model, prompt, **kwargs)
        
        # Remove any model-identifying information from response
        sanitized_result = {
            "content": result["content"],
            "tokens": result["tokens"],
            "latency": result["latency"],
            "variant_id": variant_id  # Include variant ID instead of model
        }
        
        return sanitized_result
    
    async def batch_complete_blinded(
        self,
        requests: list[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> list[Dict[str, Any]]:
        """Execute multiple blinded completion requests
        
        Args:
            requests: List of dicts with keys: variant_id, prompt, etc.
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            List of results with variant_ids (not actual models)
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_complete(request):
            async with semaphore:
                variant_id = request.pop("variant_id")
                try:
                    return await self.complete(variant_id, **request)
                except Exception as e:
                    return {
                        "error": str(e),
                        "variant_id": variant_id
                    }
        
        tasks = [limited_complete(req.copy()) for req in requests]
        return await asyncio.gather(*tasks)
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get statistics about blinded calls made"""
        if not self.call_log:
            return {"total_calls": 0, "variants_used": []}
        
        from collections import Counter
        variant_counts = Counter(call["variant_id"] for call in self.call_log)
        
        return {
            "total_calls": len(self.call_log),
            "variants_used": list(variant_counts.keys()),
            "calls_per_variant": dict(variant_counts),
            "first_call": self.call_log[0]["timestamp"],
            "last_call": self.call_log[-1]["timestamp"]
        }



