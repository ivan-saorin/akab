"""Campaign Execution Tool Registration"""
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from .handler import CampaignExecuteHandler


def register_execute_campaign_tools(
    server: FastMCP,
    response_builder,
    vault
) -> List[dict]:
    """Register campaign execution tools"""
    
    # Create handler
    handler = CampaignExecuteHandler(response_builder, vault)
    
    @server.tool()
    async def akab_execute_campaign(
        campaign_id: str,
        iterations: int = 1,
        multi_turn: Optional[bool] = None,
        max_turns: int = 10,
        target_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute A/B testing campaign with optional multi-turn support
        
        Level 2 testing - Blinded execution with progress tracking.
        
        Args:
            campaign_id: Campaign to execute
            iterations: Number of test iterations (default: 1)
            multi_turn: Enable multi-turn execution (auto-detect if None)
            max_turns: Maximum turns per test for multi-turn (default: 10)
            target_tokens: Target token count for multi-turn
            
        Multi-turn is automatically enabled when:
        - Any variant has multi_turn=true
        - Prompt contains "MINIMUM" or "[CONTINUING...]"
        - target_tokens is specified
        
        Returns:
            Execution summary with results saved to campaign
        """
        return await handler.execute(
            campaign_id=campaign_id,
            iterations=iterations,
            multi_turn=multi_turn,
            max_turns=max_turns,
            target_tokens=target_tokens
        )
    
    return [{
        "name": "akab_execute_campaign",
        "description": "Execute A/B testing campaign with progress tracking"
    }]
