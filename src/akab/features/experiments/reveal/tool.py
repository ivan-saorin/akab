"""Tool registration for reveal_experiment"""
from typing import Dict, Any
from .handler import reveal_experiment_handler


def register_reveal_experiment_tool(server, response_builder, reference_manager):
    """Register the reveal_experiment tool"""
    
    @server.tool()
    async def akab_reveal_experiment(
        experiment_id: str
    ) -> Dict[str, Any]:
        """Reveal Level 3 experiment results after statistical significance
        
        This tool will only reveal the scrambled model mappings if:
        1. Minimum iterations have been completed (typically 30+ per variant)
        2. Statistical significance has been reached (p < 0.05)
        3. Effect size is meaningful (Cohen's d > 0.2)
        
        If these conditions aren't met, the tool will explain what's needed.
        
        Args:
            experiment_id: The experiment ID from akab_create_experiment
        
        Example:
            >>> await akab_reveal_experiment("exp_7a9f2e3c")
            # If not ready:
            {
                "status": "insufficient_data",
                "iterations_completed": 45,
                "iterations_required": 90,
                "message": "Need 45 more iterations for statistical power"
            }
            
            # If ready:
            {
                "status": "revealed",
                "winner": "model_3b8d1c5a",
                "mappings": {
                    "model_7a9f2e3c": "anthropic_xs",
                    "model_3b8d1c5a": "anthropic_xl",
                    "model_9e4f7b2d": "anthropic_m"
                },
                "statistics": {
                    "p_value": 0.023,
                    "effect_size": 0.45,
                    "confidence": "95%"
                }
            }
        
        Returns:
            Experiment results with model mappings if criteria are met
        """
        return await reveal_experiment_handler(
            experiment_id=experiment_id,
            response_builder=response_builder,
            reference_manager=reference_manager
        )
    
    return akab_reveal_experiment
