"""Quick Compare Tool Registration"""
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from .handler import QuickCompareHandler


def register_quick_compare_tools(
    server: FastMCP,
    response_builder,
    reference_manager
) -> List[dict]:
    """Register quick compare tools on the server"""
    
    # Create handler with substrate components
    handler = QuickCompareHandler(response_builder, reference_manager)
    
    @server.tool()
    async def akab_quick_compare(
        prompt: str,
        providers: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Quick comparison of prompt across multiple providers
        
        Level 1 testing - No blinding, immediate results.
        Perfect for rapid iteration and debugging.
        
        Args:
            prompt: The prompt to test
            providers: List of providers to test (e.g., ["anthropic", "openai"])
                      If not specified, tests all available providers
            constraints: Optional constraints:
                        - max_tokens: Maximum tokens to generate (default: 1000)
                        - temperature: Sampling temperature (default: 0.7)
                        - models_per_provider: How many models per provider (default: 1)
        
        Returns:
            Comparison results with performance metrics and suggestions
        
        Example:
            akab_quick_compare(
                prompt="Explain quantum computing in simple terms",
                providers=["anthropic", "openai"],
                constraints={"max_tokens": 500}
            )
        """
        return await handler.compare(prompt, providers, constraints)
    
    # Return tool metadata for logging
    return [{
        "name": "akab_quick_compare",
        "description": "Quick unblinded comparison across providers"
    }]
