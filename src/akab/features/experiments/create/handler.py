"""Handler for creating Level 3 experiments"""
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


async def create_experiment_handler(
    name: str,
    description: str,
    hypothesis: str,
    variants: List[str],
    prompts: List[str],
    iterations_per_prompt: int,
    success_criteria: Optional[Dict[str, Any]],
    response_builder,
    reference_manager
) -> Dict[str, Any]:
    """Create a Level 3 scientific experiment"""
    try:
        # Import here to avoid circular imports
        from ....server_fastmcp import SCRAMBLED_MODELS
        from ....core.vault import CampaignVault
        from ....core.vault import Campaign
        from enum import Enum
        
        # Define CampaignStatus locally
        class CampaignStatus(str, Enum):
            CREATED = "created"
            RUNNING = "running"
            COMPLETED = "completed"
            CANCELLED = "cancelled"
        
        # Validate scrambled IDs exist
        invalid_variants = [v for v in variants if v not in SCRAMBLED_MODELS]
        if invalid_variants:
            return response_builder.error(
                f"Invalid scrambled model IDs: {invalid_variants}. "
                "Use akab_list_scrambled_models to get valid IDs."
            )
        
        # Validate we have at least 2 variants
        if len(variants) < 2:
            return response_builder.error(
                "Level 3 experiments require at least 2 variants for comparison"
            )
        
        # Create experiment ID
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # Create variants for the campaign structure
        # We'll expand prompts during execution
        campaign_variants = []
        for i, scrambled_id in enumerate(variants):
            variant = {
                "id": f"variant_{i+1}_{scrambled_id}",
                "prompt": "",  # Will be filled during execution
                "constraints": {
                    "scrambled_id": scrambled_id,
                    "prompts": prompts,
                    "iterations_per_prompt": iterations_per_prompt
                }
            }
            campaign_variants.append(variant)
        
        # Create campaign with level=3
        campaign = Campaign(
            id=experiment_id,
            name=name,
            description=description,
            variants=campaign_variants,
            created_at=time.time(),
            status="created",
            level=3,  # Mark as Level 3 experiment
            metadata={
                "hypothesis": hypothesis,
                "prompts": prompts,
                "iterations_per_prompt": iterations_per_prompt,
                "total_iterations": len(prompts) * iterations_per_prompt * len(variants),
                "success_criteria": success_criteria or {
                    "significance_level": 0.05,
                    "effect_size_threshold": 0.2,
                    "minimum_iterations": 30  # Per variant
                }
            }
        )
        
        # Save to vault (will go to /krill/experiments/ due to level=3)
        vault = CampaignVault()
        await vault.store_campaign(campaign)
        
        return response_builder.success(
            data={
                "experiment_id": experiment_id,
                "name": name,
                "hypothesis": hypothesis,
                "variants_count": len(variants),
                "prompts_count": len(prompts),
                "iterations_per_prompt": iterations_per_prompt,
                "total_iterations": campaign.metadata["total_iterations"],
                "status": "created",
                "storage": "Isolated in /krill/experiments/ (inaccessible to LLMs)"
            },
            message=f"Created Level 3 experiment '{name}' with {len(variants)} blinded variants",
            suggestions=[
                response_builder.suggest_next(
                    "akab_execute_campaign",
                    "Execute the experiment",
                    {"campaign_id": experiment_id}
                ),
                response_builder.suggest_next(
                    "akab_reveal_experiment",
                    "Attempt to reveal results (requires statistical significance)",
                    {"experiment_id": experiment_id}
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}", exc_info=True)
        return response_builder.error(f"Failed to create experiment: {str(e)}")
