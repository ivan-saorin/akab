"""Campaign management for AKAB."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Import substrate components
try:
    from substrate import Campaign, generate_id, estimate_progress
except ImportError:
    try:
        from .substrate import Campaign, generate_id, estimate_progress
    except ImportError:
        from .substrate_stub import Campaign, generate_id, estimate_progress

from .comparison import ComparisonEngine
from .storage import LocalStorage


class CampaignManager:
    """Manages A/B testing campaigns."""
    
    def __init__(self, storage: LocalStorage, comparison_engine: ComparisonEngine):
        """Initialize campaign manager.
        
        Args:
            storage: Local storage instance
            comparison_engine: Comparison engine instance
        """
        self.storage = storage
        self.comparison_engine = comparison_engine
        
    async def create_campaign(
        self,
        name: str,
        description: Optional[str],
        prompts: List[Dict[str, Any]],
        providers: List[str],
        iterations: int,
        constraints: Dict[str, Any]
    ) -> Campaign:
        """Create a new campaign.
        
        Args:
            name: Campaign name
            description: Optional description
            prompts: List of prompts to test
            providers: List of providers
            iterations: Number of iterations
            constraints: Campaign constraints
            
        Returns:
            Created campaign
        """
        campaign_id = generate_id("campaign")
        
        # Store prompts with IDs
        prompt_ids = []
        for i, prompt_data in enumerate(prompts):
            prompt_id = prompt_data.get("id", f"{campaign_id}_prompt_{i}")
            prompt_ids.append(prompt_id)
            await self.storage.save_prompt(prompt_id, prompt_data)
            
        campaign = Campaign(
            id=campaign_id,
            name=name,
            description=description,
            prompts=prompt_ids,
            providers=providers,
            iterations=iterations,
            constraints=constraints,
            status="draft"
        )
        
        await self.storage.save_campaign(campaign)
        return campaign
        
    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID."""
        return await self.storage.get_campaign(campaign_id)
        
    async def list_campaigns(
        self,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Campaign]:
        """List campaigns with optional filtering."""
        # Get campaign IDs from storage
        campaign_ids = await self.storage.list_campaigns(status)
        
        # Load campaign objects
        campaigns = []
        for campaign_id in campaign_ids[offset:offset + limit]:
            campaign = await self.storage.get_campaign(campaign_id)
            if campaign:
                campaigns.append(campaign)
                
        return campaigns
        
    async def execute_campaign(
        self,
        campaign_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute a campaign.
        
        Args:
            campaign_id: Campaign ID to execute
            progress_callback: Optional progress callback
            
        Returns:
            Execution results
        """
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
            
        # Update status
        campaign.status = "running"
        await self.storage.save_campaign(campaign)
        
        results = {
            "campaign_id": campaign_id,
            "start_time": time.time(),
            "iterations": {},
            "errors": []
        }
        
        try:
            total_work = len(campaign.prompts) * len(campaign.providers) * campaign.iterations
            current_work = 0
            
            # Run iterations
            for iteration in range(campaign.iterations):
                iteration_results = {}
                
                # Load and test each prompt
                for prompt_id in campaign.prompts:
                    prompt_data = await self.storage.get_prompt(prompt_id)
                    if not prompt_data:
                        results["errors"].append(f"Prompt {prompt_id} not found")
                        continue
                    
                    # Extract prompt content - handle both old and new formats
                    if isinstance(prompt_data, dict):
                        if "content" in prompt_data:
                            prompt_text = prompt_data["content"]
                        elif "prompt" in prompt_data:
                            prompt_text = prompt_data["prompt"]
                        else:
                            # Assume the whole dict is the prompt data if it has a string representation
                            prompt_text = str(prompt_data)
                    else:
                        prompt_text = str(prompt_data)
                        
                    # Run comparison
                    comparison_results = await self.comparison_engine.compare(
                        prompt=prompt_text,
                        providers=campaign.providers,
                        parameters=prompt_data.get("parameters", {}) if isinstance(prompt_data, dict) else {},
                        constraints=campaign.constraints
                    )
                    
                    # Convert ComparisonResult objects to dicts
                    serialized_results = [
                        result.dict() if hasattr(result, 'dict') else result
                        for result in comparison_results
                    ]
                    
                    iteration_results[prompt_id] = serialized_results
                    
                    # Update progress
                    current_work += len(campaign.providers)
                    if progress_callback:
                        progress = estimate_progress(current_work, total_work)
                        await progress_callback(
                            progress,
                            f"Iteration {iteration + 1}/{campaign.iterations}"
                        )
                        
                results["iterations"][f"iteration_{iteration}"] = iteration_results
                
            # Update status
            campaign.status = "completed"
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]
            
        except Exception as e:
            campaign.status = "failed"
            results["error"] = str(e)
            raise
        finally:
            await self.storage.save_campaign(campaign)
            await self.storage.save_results(campaign_id, results)
            
        return results
        
    async def analyze_campaign(
        self,
        campaign_id: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze campaign results."""
        # Get campaign and results
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
            
        results = await self.storage.get_results(campaign_id)
        if not results:
            raise ValueError(f"No results found for campaign {campaign_id}")
            
        # Aggregate all comparison results across iterations
        all_results = []
        provider_stats = {}
        prompt_stats = {}
        
        for iteration_key, iteration_data in results.get("iterations", {}).items():
            for prompt_id, prompt_results in iteration_data.items():
                for result in prompt_results:
                    all_results.append(result)
                    
                    # Track by provider
                    provider = result["provider"]
                    if provider not in provider_stats:
                        provider_stats[provider] = {
                            "total_cost": 0.0,
                            "total_latency": 0.0,
                            "total_tokens": 0,
                            "success_count": 0,
                            "error_count": 0,
                            "responses": []
                        }
                    
                    stats = provider_stats[provider]
                    if result.get("error"):
                        stats["error_count"] += 1
                    else:
                        stats["success_count"] += 1
                        stats["total_cost"] += result.get("cost_estimate", 0)
                        stats["total_latency"] += result.get("latency_ms", 0)
                        stats["total_tokens"] += result.get("tokens_used", 0)
                        stats["responses"].append(len(result.get("response", "")))
                    
                    # Track by prompt
                    if prompt_id not in prompt_stats:
                        prompt_stats[prompt_id] = {
                            "providers": {},
                            "best_response_length": 0,
                            "fastest_provider": None,
                            "cheapest_provider": None
                        }
                        
        # Calculate averages and find winners
        for provider, stats in provider_stats.items():
            if stats["success_count"] > 0:
                stats["avg_cost"] = stats["total_cost"] / stats["success_count"]
                stats["avg_latency"] = stats["total_latency"] / stats["success_count"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["success_count"]
                stats["avg_response_length"] = (
                    sum(stats["responses"]) / len(stats["responses"])
                    if stats["responses"] else 0
                )
                stats["success_rate"] = (
                    stats["success_count"] / 
                    (stats["success_count"] + stats["error_count"])
                )
                
        # Find overall winners
        successful_providers = [
            (p, s) for p, s in provider_stats.items() 
            if s["success_count"] > 0
        ]
        
        if successful_providers:
            fastest = min(successful_providers, key=lambda x: x[1]["avg_latency"])[0]
            cheapest = min(successful_providers, key=lambda x: x[1]["avg_cost"])[0]
            most_reliable = max(successful_providers, key=lambda x: x[1]["success_rate"])[0]
            
            # Overall winner (weighted score)
            scores = {}
            for provider, stats in successful_providers:
                speed_score = 1.0 - (stats["avg_latency"] / 
                    max(s["avg_latency"] for _, s in successful_providers))
                cost_score = 1.0 - (stats["avg_cost"] / 
                    max(s["avg_cost"] for _, s in successful_providers))
                reliability_score = stats["success_rate"]
                
                # Weighted score
                scores[provider] = (
                    speed_score * 0.3 +
                    cost_score * 0.3 +
                    reliability_score * 0.4
                )
                
            overall_winner = max(scores.items(), key=lambda x: x[1])[0]
        else:
            fastest = cheapest = most_reliable = overall_winner = None
            scores = {}
            
        # Build analysis
        analysis = {
            "campaign_id": campaign_id,
            "total_completions": len(all_results),
            "total_cost": sum(r.get("cost_estimate", 0) for r in all_results if not r.get("error")),
            "total_duration": results.get("duration", 0),
            "provider_performance": provider_stats,
            "winners": {
                "overall": overall_winner,
                "fastest": fastest,
                "cheapest": cheapest,
                "most_reliable": most_reliable
            },
            "scores": scores,
            "insights": self._generate_insights(provider_stats, campaign),
            "requested_metrics": metrics or ["speed", "cost", "reliability"]
        }
        
        return analysis
        
    async def estimate_cost(self, campaign_id: str) -> Dict[str, Any]:
        """Estimate campaign execution cost."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
            
        # Simple estimation based on provider costs
        # In reality, would calculate based on prompt lengths
        total_cost = 0.0
        provider_costs = {}
        
        # Placeholder calculation
        for provider in campaign.providers:
            # Assume 500 tokens per completion
            cost_per_run = 0.01  # Placeholder
            provider_cost = cost_per_run * len(campaign.prompts) * campaign.iterations
            provider_costs[provider] = provider_cost
            total_cost += provider_cost
            
        return {
            "total_cost": total_cost,
            "provider_breakdown": provider_costs,
            "prompt_count": len(campaign.prompts),
            "iterations": campaign.iterations,
            "total_completions": len(campaign.prompts) * len(campaign.providers) * campaign.iterations
        }
        
    async def get_cost_report(
        self,
        time_period: str = "week",
        group_by: str = "provider"
    ) -> Dict[str, Any]:
        """Get cost report for campaigns."""
        # Get all campaigns
        all_campaigns = await self.storage.list_campaigns()
        
        # Calculate time boundaries
        now = time.time()
        if time_period == "day":
            cutoff = now - 86400
        elif time_period == "week":
            cutoff = now - 604800
        elif time_period == "month":
            cutoff = now - 2592000
        else:  # "all"
            cutoff = 0
            
        # Aggregate costs
        total_cost = 0.0
        breakdown = {}
        campaign_costs = {}
        
        for campaign_id in all_campaigns:
            campaign = await self.get_campaign(campaign_id)
            if not campaign or campaign.created_at < cutoff:
                continue
                
            # Get results for this campaign
            results = await self.storage.get_results(campaign_id)
            if not results:
                continue
                
            campaign_cost = 0.0
            
            # Process all results
            for iteration_key, iteration_data in results.get("iterations", {}).items():
                for prompt_id, prompt_results in iteration_data.items():
                    for result in prompt_results:
                        if not result.get("error"):
                            cost = result.get("cost_estimate", 0)
                            campaign_cost += cost
                            total_cost += cost
                            
                            # Group by requested dimension
                            if group_by == "provider":
                                provider = result["provider"]
                                breakdown[provider] = breakdown.get(provider, 0) + cost
                            elif group_by == "campaign":
                                breakdown[campaign.name] = breakdown.get(campaign.name, 0) + cost
                            elif group_by == "prompt":
                                breakdown[prompt_id] = breakdown.get(prompt_id, 0) + cost
                                
            campaign_costs[campaign_id] = campaign_cost
            
        # Add summary statistics
        report = {
            "time_period": time_period,
            "group_by": group_by,
            "total_cost": total_cost,
            "breakdown": breakdown,
            "campaign_count": len(campaign_costs),
            "average_campaign_cost": (
                total_cost / len(campaign_costs) if campaign_costs else 0
            ),
            "campaigns_analyzed": list(campaign_costs.keys()),
            "date_range": {
                "start": cutoff if cutoff > 0 else "all time",
                "end": now
            }
        }
        
        return report
        
    def _generate_insights(self, provider_stats: Dict[str, Any], campaign: Campaign) -> List[str]:
        """Generate insights from provider statistics."""
        insights = []
        
        # Find notable patterns
        if provider_stats:
            # Speed insight
            latencies = [
                (p, s["avg_latency"]) 
                for p, s in provider_stats.items() 
                if s.get("avg_latency")
            ]
            if latencies:
                fastest = min(latencies, key=lambda x: x[1])
                slowest = max(latencies, key=lambda x: x[1])
                speed_diff = (slowest[1] - fastest[1]) / fastest[1] * 100
                if speed_diff > 50:
                    insights.append(
                        f"{fastest[0]} was {speed_diff:.0f}% faster than {slowest[0]}"
                    )
                    
            # Cost insight
            costs = [
                (p, s["avg_cost"]) 
                for p, s in provider_stats.items() 
                if s.get("avg_cost")
            ]
            if costs:
                cheapest = min(costs, key=lambda x: x[1])
                most_expensive = max(costs, key=lambda x: x[1])
                if most_expensive[1] > cheapest[1] * 1.5:
                    insights.append(
                        f"{most_expensive[0]} costs {most_expensive[1]/cheapest[1]:.1f}x more than {cheapest[0]}"
                    )
                    
            # Reliability insight
            reliabilities = [
                (p, s["success_rate"]) 
                for p, s in provider_stats.items() 
                if "success_rate" in s
            ]
            if reliabilities:
                for provider, rate in reliabilities:
                    if rate < 0.95:
                        insights.append(
                            f"{provider} had {(1-rate)*100:.0f}% error rate"
                        )
                        
        return insights
