"""Experiments feature - Level 3 scientific testing with triple blinding"""
from .list_scrambled import register_list_scrambled_models_tool
from .create import register_create_experiment_tool
from .reveal import register_reveal_experiment_tool
from .diagnose import register_diagnose_experiment_tool


def register_experiment_tools(server, response_builder, reference_manager):
    """Register all experiment-related tools"""
    tools = []
    
    # Register list scrambled models tool
    tools.append(register_list_scrambled_models_tool(server, response_builder, reference_manager))
    
    # Register create experiment tool
    tools.append(register_create_experiment_tool(server, response_builder, reference_manager))
    
    # Register reveal experiment tool
    tools.append(register_reveal_experiment_tool(server, response_builder, reference_manager))
    
    # Register diagnose experiment tool
    tools.append(register_diagnose_experiment_tool(server, response_builder, reference_manager))
    
    return tools
