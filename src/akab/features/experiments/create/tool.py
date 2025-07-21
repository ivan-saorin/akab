"""Tool registration for create_experiment"""
from typing import Dict, Any, List, Optional
from .handler import create_experiment_handler


def register_create_experiment_tool(server, response_builder, reference_manager):
    """Register the create_experiment tool"""
    
    @server.tool()
    async def akab_create_experiment(
        name: str,
        description: str,
        hypothesis: str,
        variants: List[str],
        prompts: List[str],
        iterations_per_prompt: int = 10,
        success_criteria: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Create Level 3 scientific experiment with complete blinding
        
        Level 3 experiments provide the highest rigor with triple blinding:
        - Fire-and-forget scrambling at startup
        - Complete isolation in /krill/experiments/
        - Statistical significance required before reveal
        
        Args:
            name: Experiment name (for reference)
            description: Detailed description of what you're testing
            hypothesis: Your scientific hypothesis
            variants: List of scrambled model IDs from akab_list_scrambled_models
            prompts: List of prompts to test (each will be run iterations_per_prompt times)
            iterations_per_prompt: Number of times to run each prompt (default: 10)
            success_criteria: Optional criteria for experiment success
        
        Example:
            >>> scrambled = await akab_list_scrambled_models()
            >>> await akab_create_experiment(
                    name="Context Window Utilization Study",
                    description="Testing how different models handle long context",
                    hypothesis="Larger models will maintain coherence better in long contexts",
                    variants=["model_7a9f2e3c", "model_3b8d1c5a", "model_9e4f7b2d"],
                    prompts=[
                        "Summarize this 10k token document...",
                        "Answer questions about this long context...",
                        "Extract key points from this lengthy text..."
                    ],
                    iterations_per_prompt=20
                )
        
        Returns:
            Experiment details with ID for execution
        """
        return await create_experiment_handler(
            name=name,
            description=description,
            hypothesis=hypothesis,
            variants=variants,
            prompts=prompts,
            iterations_per_prompt=iterations_per_prompt,
            success_criteria=success_criteria,
            response_builder=response_builder,
            reference_manager=reference_manager
        )
    
    return akab_create_experiment
