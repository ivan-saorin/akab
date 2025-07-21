"""Campaign Creation Tool Registration"""
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from .handler import CampaignCreateHandler


def register_create_campaign_tools(
    server: FastMCP,
    response_builder,
    reference_manager
) -> List[dict]:
    """Register campaign creation tools"""
    
    # Create handler with substrate components
    handler = CampaignCreateHandler(response_builder, reference_manager)
    
    @server.tool()
    async def akab_create_campaign(
        name: str,
        description: str,
        variants: Optional[List[Dict[str, Any]]] = None,
        base_prompt: Optional[str] = None,
        models: Optional[List[Dict[str, str]]] = None,
        enhancement_config: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new A/B testing campaign with flexible configuration
        
        Level 2 testing - Blinded execution with unlock capability.
        
        Two modes supported:
        
        Mode 1: Direct Variants (Full Control)
        Specify each variant explicitly:
        ```
        variants=[{
            "id": "variant_a",
            "provider": "anthropic",
            "size": "m",
            "prompt": "Custom prompt for this variant",
            "multi_turn": true,
            "target_tokens": 5000
        }]
        ```
        
        Mode 2: Auto-Generate from Base (Quick Setup)
        Generate variants from base prompt:
        ```
        base_prompt="Your prompt here",
        models=[
            {"provider": "anthropic", "size": "m"},
            {"provider": "openai", "size": "l"}
        ],
        enhancement_config={
            "enhance": true,  # Use Synapse enhancement
            "include_baseline": true,
            "multi_turn": true,
            "target_tokens": 5000
        }
        ```
        
        Args:
            name: Campaign name
            description: Campaign description
            variants: Direct variant specifications (Mode 1)
            base_prompt: Base prompt for auto-generation (Mode 2)
            models: Models for auto-generation (Mode 2)
            enhancement_config: Enhancement options (Mode 2)
            success_criteria: How to determine winner:
                            - metric: "tokens_per_second" | "quality_score" | "cost_efficiency"
                            - threshold: minimum improvement required
                            
        Returns:
            Campaign creation result with ID and next steps
        """
        return await handler.create(
            name=name,
            description=description,
            variants=variants,
            base_prompt=base_prompt,
            models=models,
            enhancement_config=enhancement_config,
            success_criteria=success_criteria
        )
    
    return [{
        "name": "akab_create_campaign",
        "description": "Create flexible A/B testing campaign"
    }]
