"""Reporting features - Cost tracking and campaign listing"""


def register_reporting_tools(server, response_builder, reference_manager):
    """Register all reporting tools"""
    tools = []
    
    # Get the vault instance
    from ...core.vault import CampaignVault
    vault = CampaignVault()
    
    # Import and register list campaigns
    from .list import register_list_campaigns_tools
    tools.extend(register_list_campaigns_tools(server, response_builder, vault))
    
    # TODO: Implement other reporting features
    # - Cost reports
    # - Export results
    
    return tools
