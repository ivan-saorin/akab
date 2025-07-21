"""List Campaigns Tool Registration"""
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from .handler import ListCampaignsHandler


def register_list_campaigns_tools(
    server: FastMCP,
    response_builder,
    vault
) -> List[dict]:
    """Register list campaigns tools"""
    
    # Create handler
    handler = ListCampaignsHandler(response_builder, vault)
    
    @server.tool()
    async def akab_list_campaigns(
        status: Optional[str] = None,
        level: Optional[int] = None
    ) -> Dict[str, Any]:
        """List all A/B testing campaigns
        
        Args:
            status: Filter by status (created, running, completed, archived)
            level: Filter by level (1, 2, or 3)
            
        Returns:
            List of campaigns with summary information
        """
        return await handler.list_campaigns(status=status, level=level)
    
    return [{
        "name": "akab_list_campaigns",
        "description": "List all A/B testing campaigns"
    }]
