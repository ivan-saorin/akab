"""
AKAB Tools Implementation
All MCP tool implementations for AKAB
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AKABTools:
    """Implementation of all AKAB MCP tools"""
    
    def __init__(self, fs_manager, provider_manager, eval_engine):
        self.fs = fs_manager
        self.providers = provider_manager
        self.evaluator = eval_engine
        
        # Track remote execution state
        self.remote_execution_state = {}
    
    async def get_meta_prompt(self, prompt_type: str = "execution") -> Dict[str, Any]:
        """Get meta prompt for AKAB operations"""
        try:
            prompt_content = await self.fs.load_meta_prompt(prompt_type)
            
            return {
                "status": "success",
                "prompt_type": prompt_type,
                "content": prompt_content,
                "message": f"Meta prompt loaded for {prompt_type}"
            }
        except Exception as e:
            logger.error(f"Error loading meta prompt: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_next_experiment(self) -> Dict[str, Any]:
        """Get the next experiment in the current campaign"""
        try:
            campaign_id = self.fs.get_current_campaign_id()
            campaign = await self.fs.load_campaign(campaign_id)
            
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found",
                    "available_campaigns": await self._get_campaign_list()
                }
            
            # Check if all experiments are completed
            total = campaign.get("total_experiments", 0)
            completed = len(campaign.get("completed_experiments", []))
            
            if completed >= total:
                return {
                    "status": "complete",
                    "message": f"All {total} experiments completed in campaign '{campaign_id}'",
                    "campaign_id": campaign_id
                }
            
            # Get next experiment
            next_exp_num = completed + 1
            experiment_id = f"exp_{next_exp_num:03d}"
            
            # Determine provider for this experiment
            providers = campaign.get("providers", ["anthropic-local"])
            provider_index = (next_exp_num - 1) % len(providers)
            provider = providers[provider_index]
            
            return {
                "status": "ready",
                "experiment_id": experiment_id,
                "experiment_number": next_exp_num,
                "total_experiments": total,
                "campaign_id": campaign_id,
                "campaign_name": campaign.get("name"),
                "provider": provider,
                "progress": f"{next_exp_num}/{total}"
            }
            
        except Exception as e:
            logger.error(f"Error getting next experiment: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_exp_prompt(self, experiment_id: str) -> Dict[str, Any]:
        """Get the full prompt for execution"""
        try:
            campaign_id = self.fs.get_current_campaign_id()
            campaign = await self.fs.load_campaign(campaign_id)
            
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found"
                }
            
            # Check if experiment already exists
            existing = await self.fs.load_experiment(campaign_id, experiment_id)
            if existing and "prompt" in existing:
                return {
                    "status": "success",
                    "experiment_id": experiment_id,
                    "prompt": existing["prompt"],
                    "cached": True
                }
            
            # Generate prompt based on campaign config
            prompt = await self._generate_experiment_prompt(campaign, experiment_id)
            
            # Save experiment config
            exp_num = int(experiment_id.split("_")[1])
            providers = campaign.get("providers", ["anthropic-local"])
            provider = providers[(exp_num - 1) % len(providers)]
            
            config = {
                "experiment_id": experiment_id,
                "campaign_id": campaign_id,
                "provider": provider,
                "created_at": datetime.now().isoformat()
            }
            
            await self.fs.save_experiment(
                campaign_id,
                experiment_id,
                config,
                prompt
            )
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "prompt": prompt,
                "provider": provider,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment prompt: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def save_exp_result(
        self,
        experiment_id: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save experiment results and auto-analyze"""
        try:
            campaign_id = self.fs.get_current_campaign_id()
            campaign = await self.fs.load_campaign(campaign_id)
            
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found"
                }
            
            # Load experiment config
            experiment = await self.fs.load_experiment(campaign_id, experiment_id)
            if not experiment:
                return {
                    "status": "error",
                    "message": f"Experiment '{experiment_id}' not found"
                }
            
            # Evaluate response
            provider = experiment.get("config", {}).get("provider", "unknown")
            evaluation = await self.evaluator.evaluate_response(
                response,
                experiment.get("config", {}),
                provider
            )
            
            # Merge with user metadata
            if metadata:
                evaluation["user_metadata"] = metadata
            
            # Save result
            result = {
                "response": response,
                "evaluation": evaluation,
                "saved_at": datetime.now().isoformat()
            }
            
            await self.fs.save_experiment(
                campaign_id,
                experiment_id,
                experiment.get("config", {}),
                experiment.get("prompt", ""),
                result
            )
            
            # Update campaign progress
            if experiment_id not in campaign.get("completed_experiments", []):
                campaign.setdefault("completed_experiments", []).append(experiment_id)
                await self.fs.save_campaign(campaign)
            
            # Return analysis
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "evaluation": evaluation,
                "progress": f"{len(campaign['completed_experiments'])}/{campaign.get('total_experiments', 0)}",
                "message": f"Results saved for experiment '{experiment_id}'"
            }
            
        except Exception as e:
            logger.error(f"Error saving experiment result: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_campaign_status(self) -> Dict[str, Any]:
        """Get current campaign status"""
        try:
            campaign_id = self.fs.get_current_campaign_id()
            campaign = await self.fs.load_campaign(campaign_id)
            
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found",
                    "available_campaigns": await self._get_campaign_list()
                }
            
            total = campaign.get("total_experiments", 0)
            completed = len(campaign.get("completed_experiments", []))
            
            # Calculate provider breakdown
            provider_stats = {}
            for exp_id in campaign.get("completed_experiments", []):
                exp = await self.fs.load_experiment(campaign_id, exp_id)
                if exp and "config" in exp:
                    provider = exp["config"].get("provider", "unknown")
                    provider_stats[provider] = provider_stats.get(provider, 0) + 1
            
            return {
                "status": "active",
                "campaign_id": campaign_id,
                "campaign_name": campaign.get("name"),
                "total_experiments": total,
                "completed_experiments": completed,
                "remaining_experiments": total - completed,
                "progress_percentage": round((completed / total * 100) if total > 0 else 0, 1),
                "providers": campaign.get("providers", []),
                "provider_stats": provider_stats,
                "created_at": campaign.get("created_at"),
                "estimated_cost": campaign.get("estimated_cost", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting campaign status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def create_campaign(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new experiment campaign"""
        try:
            # Validate required fields
            required = ["id", "name", "total_experiments"]
            missing = [f for f in required if f not in config]
            if missing:
                return {
                    "status": "error",
                    "message": f"Missing required fields: {', '.join(missing)}"
                }
            
            # Check if campaign already exists
            existing = await self.fs.load_campaign(config["id"])
            if existing:
                return {
                    "status": "error",
                    "message": f"Campaign '{config['id']}' already exists"
                }
            
            # Set defaults
            campaign = {
                "id": config["id"],
                "name": config["name"],
                "description": config.get("description", ""),
                "providers": config.get("providers", ["anthropic-local"]),
                "total_experiments": config["total_experiments"],
                "completed_experiments": [],
                "created_at": datetime.now().isoformat(),
                "config": config,
                "current_experiment": 1
            }
            
            # Estimate cost if remote providers
            remote_providers = [
                p for p in campaign["providers"] 
                if p != "anthropic-local"
            ]
            
            if remote_providers:
                experiments_per_provider = campaign["total_experiments"] / len(campaign["providers"])
                estimated_cost = 0
                
                for provider_name in remote_providers:
                    provider = self.providers.get_provider(provider_name)
                    if provider:
                        # Estimate 500 tokens per experiment
                        cost = provider.estimate_cost(500, 1000) * experiments_per_provider
                        estimated_cost += cost
                
                campaign["estimated_cost"] = round(estimated_cost, 2)
            
            # Save campaign
            success = await self.fs.save_campaign(campaign)
            
            if success:
                # Set as current campaign
                self.fs.set_current_campaign_id(campaign["id"])
                
                return {
                    "status": "success",
                    "campaign_id": campaign["id"],
                    "message": f"Campaign '{campaign['name']}' created successfully",
                    "total_experiments": campaign["total_experiments"],
                    "providers": campaign["providers"],
                    "estimated_cost": campaign.get("estimated_cost", 0)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save campaign"
                }
                
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def analyze_results(self, campaign_id: str) -> Dict[str, Any]:
        """Analyze aggregated results from a campaign"""
        try:
            campaign = await self.fs.load_campaign(campaign_id)
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found"
                }
            
            # Load all experiments
            experiments = []
            for exp_id in campaign.get("completed_experiments", []):
                exp = await self.fs.load_experiment(campaign_id, exp_id)
                if exp:
                    experiments.append(exp)
            
            if not experiments:
                return {
                    "status": "error",
                    "message": "No completed experiments to analyze"
                }
            
            # Run analysis
            analysis = await self.evaluator.analyze_campaign_results(experiments)
            
            # Save analysis
            await self.fs.save_results(campaign_id, analysis)
            
            # Format response
            return {
                "status": "success",
                "campaign_id": campaign_id,
                "campaign_name": campaign.get("name"),
                "analysis": analysis,
                "report_saved": f"/data/akab/results/{campaign_id}/report.md"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def list_campaigns(self) -> Dict[str, Any]:
        """List all available campaigns"""
        try:
            campaigns = await self.fs.list_campaigns()
            current_id = self.fs.get_current_campaign_id()
            
            return {
                "status": "success",
                "campaigns": campaigns,
                "current_campaign_id": current_id,
                "total_campaigns": len(campaigns)
            }
            
        except Exception as e:
            logger.error(f"Error listing campaigns: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def switch_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Switch the active campaign"""
        try:
            campaign = await self.fs.load_campaign(campaign_id)
            if not campaign:
                campaigns = await self._get_campaign_list()
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found",
                    "available_campaigns": campaigns
                }
            
            # Set as current campaign
            self.fs.set_current_campaign_id(campaign_id)
            
            # Get status
            total = campaign.get("total_experiments", 0)
            completed = len(campaign.get("completed_experiments", []))
            
            return {
                "status": "success",
                "campaign_id": campaign_id,
                "campaign_name": campaign.get("name"),
                "message": f"Switched to campaign '{campaign['name']}'",
                "progress": f"{completed}/{total}",
                "next_experiment": completed + 1 if completed < total else None
            }
            
        except Exception as e:
            logger.error(f"Error switching campaign: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_current_campaign(self) -> Dict[str, Any]:
        """Get information about the currently active campaign"""
        try:
            campaign_id = self.fs.get_current_campaign_id()
            campaign = await self.fs.load_campaign(campaign_id)
            
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Current campaign '{campaign_id}' not found",
                    "campaign_id": campaign_id
                }
            
            return {
                "status": "success",
                "campaign_id": campaign_id,
                "campaign_name": campaign.get("name"),
                "description": campaign.get("description", ""),
                "providers": campaign.get("providers", []),
                "progress": f"{len(campaign.get('completed_experiments', []))}/{campaign.get('total_experiments', 0)}"
            }
            
        except Exception as e:
            logger.error(f"Error getting current campaign: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_campaign_results(
        self,
        campaign_id: str,
        format: str = "structured"
    ) -> Dict[str, Any]:
        """Get all experiment results from a campaign"""
        try:
            campaign = await self.fs.load_campaign(campaign_id)
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found"
                }
            
            results = []
            
            for exp_id in campaign.get("completed_experiments", []):
                exp = await self.fs.load_experiment(campaign_id, exp_id)
                if exp and "result" in exp:
                    if format == "structured":
                        results.append({
                            "experiment_id": exp_id,
                            "provider": exp.get("config", {}).get("provider"),
                            "evaluation": exp["result"].get("evaluation", {}),
                            "response_preview": exp["result"].get("response", "")[:200] + "..."
                        })
                    elif format == "raw":
                        results.append(exp)
                    elif format == "csv":
                        # Flatten for CSV export
                        flat = {
                            "experiment_id": exp_id,
                            "provider": exp.get("config", {}).get("provider"),
                            "innovation_score": exp["result"].get("evaluation", {}).get("metrics", {}).get("innovation_score"),
                            "coherence_score": exp["result"].get("evaluation", {}).get("metrics", {}).get("coherence_score"),
                            "practicality_score": exp["result"].get("evaluation", {}).get("metrics", {}).get("practicality_score"),
                            "bs_count": exp["result"].get("evaluation", {}).get("metrics", {}).get("bs_count"),
                            "composite_score": exp["result"].get("evaluation", {}).get("composite_score")
                        }
                        results.append(flat)
            
            return {
                "status": "success",
                "campaign_id": campaign_id,
                "campaign_name": campaign.get("name"),
                "format": format,
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error getting campaign results: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_campaign_experiments(
        self,
        campaign_id: str,
        providers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get list of experiments to run for specified providers"""
        campaign = await self.fs.load_campaign(campaign_id)
        if not campaign:
            return []
        
        experiments = []
        total = campaign.get("total_experiments", 0)
        completed = campaign.get("completed_experiments", [])
        campaign_providers = campaign.get("providers", ["anthropic-local"])
        
        # Filter providers if specified
        if providers:
            campaign_providers = [p for p in campaign_providers if p in providers]
        
        # Generate experiment list
        for i in range(1, total + 1):
            exp_id = f"exp_{i:03d}"
            if exp_id not in completed:
                provider = campaign_providers[(i - 1) % len(campaign_providers)]
                
                # Skip if not in requested providers
                if providers and provider not in providers:
                    continue
                
                experiments.append({
                    "experiment_id": exp_id,
                    "campaign_id": campaign_id,
                    "provider": provider,
                    "experiment_number": i,
                    "estimated_tokens": 500  # Default estimate
                })
        
        return experiments
    
    async def execute_campaign_remote(
        self,
        campaign_id: str,
        experiments: List[Dict[str, Any]]
    ):
        """Execute campaign experiments remotely (background task)"""
        try:
            # Update state
            self.remote_execution_state[campaign_id] = {
                "status": "running",
                "total": len(experiments),
                "completed": 0,
                "started_at": datetime.now().isoformat(),
                "current_cost": 0.0
            }
            
            for exp in experiments:
                exp_id = exp["experiment_id"]
                provider_name = exp["provider"]
                
                # Update progress
                self.remote_execution_state[campaign_id]["current_experiment"] = exp_id
                
                try:
                    # Get prompt
                    prompt_result = await self.get_exp_prompt(exp_id)
                    if prompt_result["status"] != "success":
                        logger.error(f"Failed to get prompt for {exp_id}")
                        continue
                    
                    prompt = prompt_result["prompt"]
                    
                    # Execute with provider
                    provider = self.providers.get_provider(provider_name)
                    if not provider:
                        logger.error(f"Provider {provider_name} not found")
                        continue
                    
                    response, metadata = await provider.execute(prompt)
                    
                    # Save result
                    await self.save_exp_result(exp_id, response, metadata)
                    
                    # Update cost
                    if "cost" in metadata:
                        self.remote_execution_state[campaign_id]["current_cost"] += metadata["cost"]
                    
                except Exception as e:
                    logger.error(f"Error executing {exp_id}: {e}")
                
                # Update completed count
                self.remote_execution_state[campaign_id]["completed"] += 1
                
                # Small delay between requests
                await asyncio.sleep(1)
            
            # Mark as complete
            self.remote_execution_state[campaign_id]["status"] = "completed"
            self.remote_execution_state[campaign_id]["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Remote execution error: {e}")
            self.remote_execution_state[campaign_id]["status"] = "error"
            self.remote_execution_state[campaign_id]["error"] = str(e)
    
    async def get_remote_execution_progress(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        """Get progress of remote execution"""
        state = self.remote_execution_state.get(campaign_id, {})
        
        if not state:
            return {
                "status": "not_found",
                "message": f"No execution state for campaign '{campaign_id}'"
            }
        
        progress = {
            "completed": state.get("completed", 0),
            "total": state.get("total", 0),
            "percentage": round((state["completed"] / state["total"] * 100) if state["total"] > 0 else 0, 1),
            "current_experiment": state.get("current_experiment"),
            "current_cost": state.get("current_cost", 0),
            "started_at": state.get("started_at")
        }
        
        # Estimate remaining time
        if progress["completed"] > 0 and state.get("started_at"):
            started = datetime.fromisoformat(state["started_at"])
            elapsed = (datetime.now() - started).total_seconds()
            per_experiment = elapsed / progress["completed"]
            remaining = (progress["total"] - progress["completed"]) * per_experiment
            progress["estimated_remaining_seconds"] = int(remaining)
            progress["eta"] = f"{int(remaining / 60)} minutes"
        
        return progress
    
    async def _get_campaign_list(self) -> List[str]:
        """Get simple list of campaign IDs"""
        campaigns = await self.fs.list_campaigns()
        return [c["id"] for c in campaigns]
    
    async def _generate_experiment_prompt(
        self,
        campaign: Dict[str, Any],
        experiment_id: str
    ) -> str:
        """Generate prompt for an experiment"""
        # Load template if specified
        template_name = campaign.get("prompt_template")
        if template_name:
            template = await self.fs.load_template(template_name)
            if template:
                return template
        
        # Load knowledge base if specified
        kb_content = ""
        kb_name = campaign.get("knowledge_base")
        if kb_name:
            kb = await self.fs.load_knowledge_base(kb_name)
            if kb:
                kb_content = f"\n\n## Knowledge Base\n\n{kb}\n\n"
        
        # Generate default prompt
        exp_num = int(experiment_id.split("_")[1])
        
        return f"""# Experiment {exp_num} - {campaign.get('name', 'Unknown Campaign')}

{campaign.get('description', '')}

{kb_content}

## Task

Please provide a comprehensive response to the following challenge:

Design an innovative solution for improving hybrid work collaboration in a 500-employee technology company. Consider:

1. Technical infrastructure needs
2. Cultural and social aspects
3. Productivity measurement
4. Employee well-being
5. Cost-effectiveness

Be creative and think outside conventional approaches. Provide specific, actionable recommendations.

## Response Guidelines

- Be innovative but practical
- Include specific implementation steps
- Consider potential challenges
- Provide measurable success criteria
- Think holistically about the problem

Your response should demonstrate both creativity and feasibility."""
