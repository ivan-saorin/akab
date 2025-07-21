"""Campaign Creation Handler - Full Implementation"""
from typing import Dict, Any, List, Optional
import uuid
import time
from substrate.shared.models import MODEL_REGISTRY
from ....core.vault import CampaignVault, Campaign


class CampaignCreateHandler:
    """Handler for creating Level 2 campaigns"""
    
    def __init__(self, response_builder, reference_manager):
        self.response_builder = response_builder
        self.reference_manager = reference_manager
        self.vault = CampaignVault()
        
        # Model size mappings (same as old AKAB)
        self.model_sizes = {
            "anthropic": {
                "xs": "claude-3-haiku-20240307",
                "s": "claude-3-5-haiku-20241022", 
                "m": "claude-3-5-sonnet-20241022",
                "l": "claude-3-5-sonnet-20241022",
                "xl": "claude-3-opus-20240229"
            },
            "openai": {
                "xs": "gpt-3.5-turbo",
                "s": "gpt-4o-mini",
                "m": "gpt-4",
                "l": "gpt-4-turbo",
                "xl": "gpt-4-turbo"
            },
            "google": {
                "xs": "gemini-2.0-flash-exp",
                "s": "gemini-2.0-flash-exp",
                "m": "gemini-1.5-pro-002",
                "l": "gemini-1.5-pro-002",
                "xl": "gemini-exp-1206"
            },
            "groq": {
                "xs": "llama-3.3-70b-versatile",
                "s": "llama-3.3-70b-versatile",
                "m": "mixtral-8x7b-32768",
                "l": "llama-3.1-70b-versatile",
                "xl": "llama-3.3-70b-versatile"
            }
        }
    
    async def create(
        self,
        name: str,
        description: str,
        variants: Optional[List[Dict[str, Any]]] = None,
        base_prompt: Optional[str] = None,
        models: Optional[List[Dict[str, str]]] = None,
        enhancement_config: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new campaign with two modes:
        
        Mode 1: Direct variants (full control)
        Mode 2: Auto-generate from base prompt
        """
        
        # Validate inputs
        if not variants and not base_prompt:
            return self.response_builder.error(
                "Must provide either 'variants' (Mode 1) or 'base_prompt' (Mode 2)"
            )
        
        # Generate campaign ID
        campaign_id = f"campaign_{int(time.time() * 1000)}"
        
        # Mode 2: Auto-generate variants from base prompt
        if variants is None and base_prompt is not None and models is not None:
            variants = await self._generate_variants_from_base(
                base_prompt, models, enhancement_config
            )
        
        # Validate and transform variants
        transformed_variants = []
        for idx, variant in enumerate(variants):
            # Ensure variant has an ID
            if "id" not in variant:
                variant["id"] = f"variant_{idx}"
            
            # Handle model nickname transformation
            if "size" in variant and variant.get("provider") in self.model_sizes:
                provider = variant["provider"]
                size = variant["size"]
                # Look up the actual model name
                variant["model"] = self.model_sizes[provider].get(size, 
                    self.model_sizes[provider].get("m"))  # Default to medium
            
            # Validate required fields
            required_fields = ["id", "provider", "model", "prompt"]
            missing = [f for f in required_fields if f not in variant]
            if missing:
                return self.response_builder.error(
                    f"Variant '{variant.get('id', 'unknown')}' missing fields: {missing}"
                )
            
            transformed_variants.append(variant)
        
        # Set default success criteria if not provided
        if success_criteria is None:
            success_criteria = {
                "evaluation_method": "statistical",
                "primary_metric": "execution_time",
                "direction": "minimize"
            }
        
        # Create campaign object
        campaign = Campaign(
            id=campaign_id,
            name=name,
            description=description,
            variants=transformed_variants,
            created_at=time.time(),
            status="created",
            success_criteria=success_criteria,
            level=2,  # Level 2 campaign
            metadata={
                "created_via": "fastmcp",
                "enhancement_config": enhancement_config if base_prompt else None,
                "base_prompt": base_prompt if base_prompt else None
            }
        )
        
        # Create variant mapping for blinding (Level 2)
        variant_mapping = {}
        for variant in transformed_variants:
            # Create blinded ID
            blinded_id = f"blinded_{uuid.uuid4().hex[:8]}"
            variant_mapping[variant["id"]] = blinded_id
        
        campaign.variant_mapping = variant_mapping
        
        # Store campaign in vault
        await self.vault.store_campaign(campaign)
        
        # Prepare response data
        response_data = {
            "campaign_id": campaign_id,
            "name": name,
            "status": "created",
            "level": 2,
            "variants_count": len(transformed_variants),
            "success_criteria": success_criteria
        }
        
        # Add enhanced campaign info if using base_prompt mode
        if base_prompt is not None:
            enhanced_count = sum(1 for v in transformed_variants if "enhanced" in v["id"])
            baseline_count = sum(1 for v in transformed_variants if "baseline" in v["id"])
            
            response_data.update({
                "enhanced_variants": enhanced_count,
                "baseline_variants": baseline_count,
                "multi_turn_enabled": any(v.get("multi_turn", False) for v in transformed_variants),
                "target_tokens": enhancement_config.get("target_tokens") if enhancement_config else None
            })
        
        return self.response_builder.success(
            data=response_data,
            message=f"Campaign '{name}' created with {len(transformed_variants)} variants",
            suggestions=[
                self.response_builder.suggest_next(
                    "akab_execute_campaign",
                    "Execute the campaign",
                    campaign_id=campaign_id,
                    iterations=5
                ),
                self.response_builder.suggest_next(
                    "akab_list_campaigns",
                    "View all campaigns"
                )
            ]
        )
    
    async def _generate_variants_from_base(
        self,
        base_prompt: str,
        models: List[Dict[str, str]],
        enhancement_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate variants from base prompt and model specifications"""
        
        enhancement_config = enhancement_config or {}
        variants = []
        
        # Default configuration
        multi_turn = enhancement_config.get("multi_turn", False)
        target_tokens = enhancement_config.get("target_tokens")
        include_baseline = enhancement_config.get("include_baseline", True)
        enhance = enhancement_config.get("enhance", False)
        
        for model_spec in models:
            provider = model_spec.get("provider")
            size = model_spec.get("size", "m")
            
            # Get actual model name
            if provider in self.model_sizes:
                model_name = self.model_sizes[provider].get(size, 
                    self.model_sizes[provider].get("m"))
            else:
                # Unknown provider, skip
                continue
            
            # Create baseline variant
            if include_baseline:
                baseline_variant = {
                    "id": f"{provider}_{size}_baseline",
                    "provider": provider,
                    "model": model_name,
                    "prompt": base_prompt,
                    "multi_turn": multi_turn,
                    "target_tokens": target_tokens
                }
                variants.append(baseline_variant)
            
            # Create enhanced variant if requested
            if enhance:
                # TODO: When Synapse is available, actually enhance the prompt
                # For now, just add a simple enhancement
                enhanced_prompt = self._apply_simple_enhancement(base_prompt)
                
                enhanced_variant = {
                    "id": f"{provider}_{size}_enhanced",
                    "provider": provider,
                    "model": model_name,
                    "prompt": enhanced_prompt,
                    "multi_turn": multi_turn,
                    "target_tokens": target_tokens,
                    "metadata": {
                        "enhanced": True,
                        "enhancement_strategy": enhancement_config.get("strategy", "auto")
                    }
                }
                variants.append(enhanced_variant)
        
        return variants
    
    def _apply_simple_enhancement(self, prompt: str) -> str:
        """Apply a simple enhancement pattern (placeholder for Synapse integration)"""
        # TODO: This will be replaced with actual Synapse enhancement
        enhancement = """You are an expert assistant with deep knowledge and analytical capabilities.
        
Task: """
        
        return enhancement + prompt
