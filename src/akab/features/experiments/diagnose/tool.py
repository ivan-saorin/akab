"""Tool registration for diagnose_experiment"""
from typing import Dict, Any
from .handler import diagnose_experiment_handler


def register_diagnose_experiment_tool(server, response_builder, reference_manager):
    """Register the diagnose_experiment tool"""
    
    @server.tool()
    async def akab_diagnose_experiment(
        experiment_id: str,
        force_reveal: bool = False
    ) -> Dict[str, Any]:
        """Diagnose why a Level 3 experiment hasn't reached significance
        
        This tool analyzes experiment data to understand:
        - Variance patterns across models
        - Effect sizes between variants
        - Prompt-specific performance
        - Convergence trends
        - Recommendations for achieving significance
        
        Args:
            experiment_id: The experiment ID to diagnose
            force_reveal: Break protocol and reveal mappings (not recommended)
        
        Example:
            >>> await akab_diagnose_experiment("exp_7a9f2e3c")
            {
                "diagnosis": {
                    "variance_analysis": "High variance in variant_2",
                    "effect_size": "Small (d=0.12)",
                    "convergence": "Not converging",
                    "problematic_prompts": ["prompt_3"]
                },
                "recommendations": [
                    "Run 50 more iterations",
                    "Consider removing high-variance prompt_3",
                    "Effect size suggests models may be too similar"
                ]
            }
        
        Returns:
            Diagnostic analysis and recommendations
        """
        return await diagnose_experiment_handler(
            experiment_id=experiment_id,
            force_reveal=force_reveal,
            response_builder=response_builder,
            reference_manager=reference_manager
        )
    
    return akab_diagnose_experiment
