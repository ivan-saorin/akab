"""Handler for listing scrambled models"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def list_scrambled_models_handler(response_builder) -> Dict[str, Any]:
    """List available scrambled model IDs for Level 3 experiments"""
    try:
        # Import here to avoid circular imports
        from ....server import SCRAMBLED_MODELS
        
        # Get list of scrambled IDs (without revealing mappings)
        scrambled_ids = sorted(list(SCRAMBLED_MODELS.keys()))
        
        return response_builder.success(
            data={
                "scrambled_models": scrambled_ids,
                "count": len(scrambled_ids),
                "description": "Use these scrambled IDs when creating Level 3 experiments",
                "warning": "Model identities are hidden until experiment reaches statistical significance"
            },
            message=f"Found {len(scrambled_ids)} scrambled models available for experiments",
            suggestions=[
                response_builder.suggest_next(
                    "akab_create_experiment",
                    "Create a Level 3 experiment with these models"
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Error listing scrambled models: {e}", exc_info=True)
        return response_builder.error(f"Failed to list scrambled models: {str(e)}")
