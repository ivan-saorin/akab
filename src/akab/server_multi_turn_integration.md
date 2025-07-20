# AKAB Server Multi-Turn Integration

Add these modifications to the existing `akab/server.py` file:

## 1. Import the multi-turn module at the top

```python
from .multi_turn import MultiTurnExecutor, EnhancedCampaignExecutor
```

## 2. Update the __init__ method

```python
def __init__(self):
    super().__init__(...)
    
    # Existing initialization...
    
    # Initialize multi-turn executor
    self.multi_turn_executor = MultiTurnExecutor(
        hermes=self.hermes,
        sampling_manager=self.sampling_manager
    )
    
    # Initialize enhanced campaign executor
    self.enhanced_executor = EnhancedCampaignExecutor(
        multi_turn_executor=self.multi_turn_executor,
        pattern_enhancer=None  # Will be set if Synapse is available
    )
    
    # Try to connect to Synapse for pattern enhancement
    self._initialize_synapse_connection()
```

## 3. Add Synapse connection initialization

```python
def _initialize_synapse_connection(self):
    """Try to connect to Synapse for pattern enhancement"""
    try:
        # This would be the actual connection logic
        # For now, we'll just set a flag
        self.synapse_available = False
        
        # In production, you'd check if Synapse MCP is available
        # and create a client connection
        logger.info("Synapse pattern enhancement not connected")
        
    except Exception as e:
        logger.warning(f"Could not connect to Synapse: {e}")
        self.synapse_available = False
```

## 4. Update the execute_campaign method

```python
async def execute_campaign(self, campaign_id: str, iterations: int = 1,
                          multi_turn: bool = None, max_turns: int = 10,
                          target_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Execute A/B testing campaign with optional multi-turn support
    
    Args:
        campaign_id: Campaign to execute
        iterations: Number of test iterations
        multi_turn: Enable multi-turn execution (auto-detect if None)
        max_turns: Maximum turns per test (for multi-turn)
        target_tokens: Target token count (for multi-turn)
    
    Returns:
        Execution summary with results saved to campaign
    """
    try:
        # Load campaign
        campaign = await self.campaign_manager.get_campaign(campaign_id)
        if not campaign:
            raise ValidationError(f"Campaign {campaign_id} not found")
        
        if campaign.status != "active":
            raise ValidationError(
                f"Campaign is {campaign.status}, not active",
                suggestions=["Create a new campaign or reactivate this one"]
            )
        
        # Auto-detect multi-turn need
        if multi_turn is None:
            # Check if any variant has multi_turn flag or uses stable genius
            multi_turn = any(
                v.get("multi_turn", False) or 
                "MINIMUM" in v.get("prompt", "") or
                "[CONTINUING...]" in v.get("prompt", "")
                for v in campaign.variants
            )
        
        logger.info(f"Starting execution of campaign {campaign_id}")
        logger.info(f"Iterations: {iterations}, Multi-turn: {multi_turn}")
        
        # Track progress
        total_tests = len(campaign.variants) * iterations
        completed_tests = 0
        
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
            # Use standard execution (existing code)
            results = []
            for variant in campaign.variants:
                for i in range(iterations):
                    # ... existing execution code ...
                    pass
        
        # Save results to campaign
        for result in results:
            campaign.results.append(result)
        
        # Update campaign
        campaign.last_run = time.time()
        campaign.metadata["total_tests"] = len(campaign.results)
        campaign.metadata["multi_turn_enabled"] = multi_turn
        
        await self.campaign_manager._save_campaign(campaign)
        
        # Calculate summary
        summary = self._calculate_execution_summary(results, multi_turn)
        
        return self.create_response(
            data=summary,
            message=f"Campaign executed successfully with {len(results)} tests"
        )
        
    except ValidationError:
        raise
    except Exception as e:
        return self.create_error_response(str(e))
```

## 6. Update analyze_results to handle multi-turn

```python
async def analyze_results(self, campaign_id: str) -> Dict[str, Any]:
    """Analyze campaign results with multi-turn awareness"""
    try:
        campaign = await self.campaign_manager.get_campaign(campaign_id)
        if not campaign:
            raise ValidationError(f"Campaign {campaign_id} not found")
        
        # Check if this was a multi-turn campaign
        is_multi_turn = campaign.metadata.get("multi_turn_enabled", False)
        
        if is_multi_turn:
            # Use fair metrics calculation
            fair_metrics = self.enhanced_executor.calculate_fair_metrics(campaign.results)
            
            # Include turn analysis
            analysis_result = self.laboratory.analyze_results(
                campaign, 
                additional_metrics=fair_metrics
            )
            
            # Add multi-turn specific insights
            analysis_result["multi_turn_analysis"] = {
                "avg_turns_by_variant": {
                    k: v["avg_turns_used"] 
                    for k, v in fair_metrics.items()
                },
                "token_efficiency": {
                    k: v["avg_total_tokens"] / v["avg_turns_used"]
                    for k, v in fair_metrics.items()
                },
                "cost_per_1k_tokens": {
                    k: (v["avg_cost_per_test"] / v["avg_total_tokens"]) * 1000
                    for k, v in fair_metrics.items()
                }
            }
        else:
            # Standard analysis
            analysis_result = self.laboratory.analyze_results(campaign)
        
        # Save analysis to results
        await self._save_analysis_results(campaign_id, analysis_result, is_multi_turn)
        
        return self.create_response(
            data=analysis_result,
            message="Analysis complete" + (" with multi-turn metrics" if is_multi_turn else "")
        )
        
    except Exception as e:
        return self.create_error_response(str(e))
```

## 7. Add helper method for execution summary

```python
def _calculate_execution_summary(self, results: List[Dict[str, Any]], 
                               multi_turn: bool) -> Dict[str, Any]:
    """Calculate execution summary with multi-turn awareness"""
    
    summary = {
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r.get("success", False)),
        "failed_tests": sum(1 for r in results if not r.get("success", False)),
        "multi_turn": multi_turn
    }
    
    if multi_turn:
        # Add multi-turn specific metrics
        successful_results = [r for r in results if r.get("success", False)]
        
        if successful_results:
            summary["multi_turn_metrics"] = {
                "avg_turns": sum(r.get("turns_used", 1) for r in successful_results) / len(successful_results),
                "max_turns": max(r.get("turns_used", 1) for r in successful_results),
                "min_turns": min(r.get("turns_used", 1) for r in successful_results),
                "total_tokens_generated": sum(r.get("total_tokens", 0) for r in successful_results),
                "natural_completions": sum(1 for r in successful_results if r.get("completed_naturally", False))
            }
    
    # Calculate costs
    total_cost = sum(r.get("total_cost", r.get("cost", 0)) for r in results)
    summary["total_cost"] = round(total_cost, 4)
    
    # Execution time
    exec_times = []
    for r in results:
        if "execution_times" in r:
            exec_times.extend(r["execution_times"])
        elif "execution_time" in r:
            exec_times.append(r["execution_time"])
    
    if exec_times:
        summary["avg_execution_time"] = sum(exec_times) / len(exec_times)
        summary["total_execution_time"] = sum(exec_times)
    
    return summary
```

## 8. Add quality scoring for multi-turn

```python
async def score_multi_turn_quality(self, result: Dict[str, Any]) -> float:
    """Score quality of multi-turn generation"""
    
    content = result.get("content", "")
    turns_used = result.get("turns_used", 1)
    completed_naturally = result.get("completed_naturally", False)
    
    # Base quality score (could use LLM judge here)
    base_score = len(content) / 1000  # Simple length-based score
    
    # Adjust for efficiency
    efficiency_bonus = 0
    if completed_naturally:
        # Bonus for natural completion
        efficiency_bonus += 0.1
    
    # Penalty for too many turns (inefficient)
    if turns_used > 5:
        efficiency_bonus -= 0.05 * (turns_used - 5)
    
    # Token efficiency
    tokens_per_turn = result.get("total_tokens", 0) / turns_used
    if tokens_per_turn > 2000:
        efficiency_bonus += 0.05
    
    return min(1.0, base_score + efficiency_bonus)
```
