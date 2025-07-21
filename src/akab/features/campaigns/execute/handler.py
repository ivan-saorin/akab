"""Campaign Execution Handler"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from substrate.shared.api import ClearHermes
from ....core.hermes import BlindedHermes
from ....core.laboratory import Laboratory, LABORATORY
from ....core.laboratory.multi_turn import MultiTurnExecutor

logger = logging.getLogger(__name__)


class CampaignExecuteHandler:
    """Handler for executing Level 2 campaigns"""
    
    def __init__(self, response_builder, vault):
        self.response_builder = response_builder
        self.vault = vault
        self.laboratory = LABORATORY
        self.blinded_hermes = BlindedHermes()
        self.multi_turn_executor = MultiTurnExecutor(self.blinded_hermes)
    
    async def execute(
        self,
        campaign_id: str,
        iterations: int = 1,
        multi_turn: Optional[bool] = None,
        max_turns: int = 10,
        target_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute campaign with blinded execution"""
        
        # Load campaign
        campaign = await self.vault.get_campaign(campaign_id)
        if not campaign:
            return self.response_builder.error(f"Campaign not found: {campaign_id}")
        
        if campaign.status != "created" and campaign.status != "running":
            return self.response_builder.error(
                f"Campaign is {campaign.status}, cannot execute"
            )
        
        # Update status to running
        campaign.status = "running"
        await self.vault.update_campaign(campaign)
        
        # Auto-detect multi-turn need
        if multi_turn is None:
            multi_turn = any(
                v.get("multi_turn", False) or 
                "MINIMUM" in v.get("prompt", "") or
                "[CONTINUING...]" in v.get("prompt", "")
                for v in campaign.variants
            )
        
        logger.info(f"Executing campaign {campaign_id}: iterations={iterations}, multi_turn={multi_turn}")
        
        try:
            # Set up blinding
            self.blinded_hermes.set_mapping(campaign.variant_mapping)
            
            # Execute based on mode
            if multi_turn:
                # Use multi-turn executor
                results = await self.multi_turn_executor.execute_campaign_with_continuation(
                    campaign=campaign,
                    iterations=iterations,
                    max_turns_per_test=max_turns,
                    target_tokens=target_tokens
                )
            else:
                # Standard execution
                results = await self._execute_standard(
                    campaign, iterations
                )
            
            # Add results to campaign
            for result in results:
                campaign.add_result(result)
            
            # Update campaign
            campaign.metadata["last_execution"] = time.time()
            campaign.metadata["total_executions"] = len(campaign.results)
            campaign.metadata["multi_turn_enabled"] = multi_turn
            
            # Update status
            if len(campaign.results) >= len(campaign.variants) * 10:
                campaign.status = "completed"
            else:
                campaign.status = "created"  # Ready for more executions
            
            await self.vault.update_campaign(campaign)
            
            # Calculate summary
            summary = self._calculate_execution_summary(results, multi_turn)
            
            return self.response_builder.success(
                data={
                    "campaign_id": campaign_id,
                    "execution_summary": summary,
                    "campaign_status": campaign.status
                },
                message=f"Executed {len(results)} tests successfully",
                suggestions=[
                    self.response_builder.suggest_next(
                        "akab_analyze_results",
                        "Analyze campaign results",
                        campaign_id=campaign_id
                    ),
                    self.response_builder.suggest_next(
                        "akab_execute_campaign",
                        "Run more iterations",
                        campaign_id=campaign_id,
                        iterations=5
                    ) if campaign.status != "completed" else None
                ]
            )
            
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            # Reset status
            campaign.status = "created"
            await self.vault.update_campaign(campaign)
            return self.response_builder.error(str(e))
    
    async def _execute_standard(
        self, 
        campaign: Any,
        iterations: int
    ) -> List[Dict[str, Any]]:
        """Standard (single-turn) execution"""
        results = []
        
        # Create tasks for all variant-iteration combinations
        tasks = []
        for iteration in range(iterations):
            for variant in campaign.variants:
                task = self._execute_single_test(
                    variant=variant,
                    campaign_id=campaign.id,
                    iteration=iteration,
                    variant_mapping=campaign.variant_mapping
                )
                tasks.append(task)
        
        # Execute with concurrency limit
        all_results = await self._gather_with_limit(tasks, limit=5)
        
        # Process results
        for idx, result in enumerate(all_results):
            if isinstance(result, Exception):
                # Create error result
                iteration = idx // len(campaign.variants)
                variant_idx = idx % len(campaign.variants)
                variant = campaign.variants[variant_idx]
                
                results.append({
                    "variant": campaign.variant_mapping[variant["id"]],
                    "iteration": iteration,
                    "success": False,
                    "error": str(result),
                    "timestamp": time.time()
                })
            else:
                results.append(result)
        
        return results
    
    async def _execute_single_test(
        self,
        variant: Dict[str, Any],
        campaign_id: str,
        iteration: int,
        variant_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Execute a single test with blinding"""
        
        start_time = time.time()
        blinded_id = variant_mapping[variant["id"]]
        
        try:
            # Use blinded hermes for execution
            result = await self.blinded_hermes.complete(
                variant_id=variant["id"],  # BlindedHermes will map this
                prompt=variant["prompt"],
                max_tokens=variant.get("max_tokens", 1000),
                temperature=variant.get("temperature", 0.7)
            )
            
            return {
                "variant": blinded_id,  # Store blinded ID
                "iteration": iteration,
                "success": True,
                "response": result["content"],
                "response_length": len(result["content"]),
                "execution_time": time.time() - start_time,
                "tokens_used": result["tokens"],
                "cost": result["tokens"] * 0.00001,  # Simplified cost calc
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "variant": blinded_id,
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }
    
    async def _gather_with_limit(self, tasks: List, limit: int = 5):
        """Execute tasks with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def bounded_task(task):
            async with semaphore:
                try:
                    return await task
                except Exception as e:
                    return e
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        return await asyncio.gather(*bounded_tasks)
    
    def _calculate_execution_summary(
        self, 
        results: List[Dict[str, Any]], 
        multi_turn: bool
    ) -> Dict[str, Any]:
        """Calculate execution summary"""
        
        summary = {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.get("success", False)),
            "failed_tests": sum(1 for r in results if not r.get("success", False)),
            "multi_turn": multi_turn
        }
        
        if multi_turn:
            # Add multi-turn metrics
            successful_results = [r for r in results if r.get("success", False)]
            
            if successful_results:
                summary["multi_turn_metrics"] = {
                    "avg_turns": sum(r.get("turns_used", 1) for r in successful_results) / len(successful_results),
                    "max_turns": max(r.get("turns_used", 1) for r in successful_results),
                    "min_turns": min(r.get("turns_used", 1) for r in successful_results),
                    "total_tokens_generated": sum(r.get("total_tokens", 0) for r in successful_results)
                }
        
        # Calculate costs
        total_cost = sum(r.get("cost", 0) for r in results)
        summary["total_cost"] = round(total_cost, 4)
        
        # Execution time
        exec_times = [r.get("execution_time", 0) for r in results if "execution_time" in r]
        if exec_times:
            summary["avg_execution_time"] = round(sum(exec_times) / len(exec_times), 2)
            summary["total_execution_time"] = round(sum(exec_times), 2)
        
        return summary
