"""Multi-turn execution support for AKAB campaigns"""
import logging
from typing import Dict, Any, List, Optional
from substrate.shared.api import ClearHermes
from ..hermes import BlindedHermes

logger = logging.getLogger(__name__)


class MultiTurnExecutor:
    """Handles multi-turn execution for long-form content generation
    
    This is an AKAB-specific feature that enables campaigns to:
    - Continue generation across multiple API calls
    - Track progress towards token targets
    - Handle natural completion detection
    - Maintain context across turns
    """
    
    def __init__(self, hermes: BlindedHermes = None):
        """Initialize with Hermes instance"""
        self.hermes = hermes or BlindedHermes()
        self.clear_hermes = ClearHermes()  # For unblinded execution
    
    async def execute_campaign_with_continuation(
        self,
        campaign: Any,  # Campaign object
        iterations: int = 1,
        max_turns_per_test: int = 10,
        target_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute campaign with multi-turn support
        
        Args:
            campaign: Campaign object with variants
            iterations: Number of test iterations
            max_turns_per_test: Maximum turns allowed per test
            target_tokens: Target token count to achieve
            
        Returns:
            List of execution results with multi-turn metrics
        """
        results = []
        
        # TODO: Implement full multi-turn logic
        # This is a placeholder showing the pattern
        
        for iteration in range(iterations):
            for variant in campaign.variants:
                # Check if variant needs multi-turn
                needs_multi_turn = (
                    variant.get("multi_turn", False) or
                    "MINIMUM" in variant.get("prompt", "") or
                    target_tokens is not None
                )
                
                if needs_multi_turn:
                    # Execute with continuation
                    result = await self._execute_multi_turn_test(
                        variant=variant,
                        campaign_id=campaign.id,
                        iteration=iteration,
                        max_turns=max_turns_per_test,
                        target_tokens=target_tokens or variant.get("target_tokens", 5000)
                    )
                else:
                    # Single turn execution
                    result = await self._execute_single_turn_test(
                        variant=variant,
                        campaign_id=campaign.id,
                        iteration=iteration
                    )
                
                results.append(result)
        
        return results
    
    async def _execute_multi_turn_test(
        self,
        variant: Dict[str, Any],
        campaign_id: str,
        iteration: int,
        max_turns: int,
        target_tokens: int
    ) -> Dict[str, Any]:
        """Execute a single multi-turn test"""
        
        # Placeholder implementation
        return {
            "variant": variant["id"],
            "iteration": iteration,
            "success": True,
            "turns_used": 1,
            "total_tokens": 1000,
            "completed_naturally": True,
            "content": "Multi-turn execution not fully implemented yet",
            "execution_time": 1.0,
            "cost": 0.01
        }
    
    async def _execute_single_turn_test(
        self,
        variant: Dict[str, Any],
        campaign_id: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Execute a single turn test"""
        
        # Placeholder implementation
        return {
            "variant": variant["id"],
            "iteration": iteration,
            "success": True,
            "turns_used": 1,
            "content": "Single turn execution",
            "execution_time": 0.5,
            "cost": 0.005
        }
    
    def calculate_fair_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate fair metrics for multi-turn campaigns
        
        Args:
            results: List of campaign results
            
        Returns:
            Dict mapping variant IDs to fair metrics
        """
        # Group results by variant
        variant_metrics = {}
        
        for result in results:
            variant_id = result.get("variant")
            if not variant_id:
                continue
            
            if variant_id not in variant_metrics:
                variant_metrics[variant_id] = {
                    "results": [],
                    "total_tests": 0,
                    "successful_tests": 0
                }
            
            variant_metrics[variant_id]["results"].append(result)
            variant_metrics[variant_id]["total_tests"] += 1
            
            if result.get("success", False):
                variant_metrics[variant_id]["successful_tests"] += 1
        
        # Calculate fair metrics for each variant
        fair_metrics = {}
        
        for variant_id, data in variant_metrics.items():
            successful_results = [
                r for r in data["results"] 
                if r.get("success", False)
            ]
            
            if successful_results:
                fair_metrics[variant_id] = {
                    "avg_turns_used": sum(r.get("turns_used", 1) for r in successful_results) / len(successful_results),
                    "avg_total_tokens": sum(r.get("total_tokens", 0) for r in successful_results) / len(successful_results),
                    "avg_response_length": sum(len(r.get("content", "")) for r in successful_results) / len(successful_results),
                    "avg_cost_per_test": sum(r.get("cost", 0) for r in successful_results) / len(successful_results),
                    "success_rate": data["successful_tests"] / data["total_tests"]
                }
            else:
                fair_metrics[variant_id] = {
                    "avg_turns_used": 0,
                    "avg_total_tokens": 0,
                    "avg_response_length": 0,
                    "avg_cost_per_test": 0,
                    "success_rate": 0
                }
        
        return fair_metrics
