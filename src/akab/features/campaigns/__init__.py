"""Campaigns feature - Level 2 A/B testing"""


def register_campaign_tools(server, response_builder, reference_manager):
    """Register all campaign-related tools"""
    tools = []
    
    # Get the vault instance from server
    from ...core.vault import CampaignVault
    vault = CampaignVault()
    
    # Import and register each sub-feature
    from .create import register_create_campaign_tools
    tools.extend(register_create_campaign_tools(server, response_builder, reference_manager))
    
    from .execute import register_execute_campaign_tools
    tools.extend(register_execute_campaign_tools(server, response_builder, vault))
    
    # TODO: Add other sub-features when implemented
    # from .analyze import register_analyze_campaign_tools
    # tools.extend(register_analyze_campaign_tools(server, response_builder, vault))
    
    # from .unlock import register_unlock_campaign_tools
    # tools.extend(register_unlock_campaign_tools(server, response_builder, vault))
    
    return tools
