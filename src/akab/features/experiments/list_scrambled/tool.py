"""Tool registration for list_scrambled_models"""
from typing import Dict, Any
from .handler import list_scrambled_models_handler


def register_list_scrambled_models_tool(server, response_builder, reference_manager):
    """Register the list_scrambled_models tool"""
    
    @server.tool()
    async def akab_list_scrambled_models() -> Dict[str, Any]:
        """List available scrambled model IDs for Level 3 experiments
        
        Returns scrambled model identifiers without revealing the actual models.
        These IDs are used in Level 3 experiments to ensure complete blinding.
        
        Example:
            >>> await akab_list_scrambled_models()
            {
                "scrambled_models": [
                    "model_7a9f2e3c",
                    "model_3b8d1c5a",
                    "model_9e4f7b2d",
                    ...
                ],
                "count": 24
            }
        """
        return await list_scrambled_models_handler(response_builder)
    
    return akab_list_scrambled_models
