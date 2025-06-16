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
    
    # ============= Phase 2 Enhancements =============
    
    async def save_knowledge_base(
        self,
        name: str,
        content: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Save a knowledge base document"""
        try:
            # Validate name
            if not name or not name.strip():
                return {
                    "status": "error",
                    "message": "Knowledge base name cannot be empty"
                }
            
            # Ensure .md extension
            if not name.endswith('.md'):
                name = f"{name}.md"
            
            # Check if KB already exists
            existing = await self.fs.load_knowledge_base(name)
            if existing:
                return {
                    "status": "error",
                    "message": f"Knowledge base '{name}' already exists. Use a different name or delete the existing KB first."
                }
            
            # Save KB
            success = await self.fs.save_knowledge_base(name, content, description)
            
            if success:
                return {
                    "status": "success",
                    "kb_name": name,
                    "message": f"Knowledge base '{name}' saved successfully",
                    "description": description
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save knowledge base"
                }
                
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def list_knowledge_bases(self) -> Dict[str, Any]:
        """List available knowledge bases"""
        try:
            kbs = await self.fs.list_knowledge_bases()
            
            return {
                "status": "success",
                "knowledge_bases": kbs,
                "total_kbs": len(kbs)
            }
            
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def export_campaign(
        self,
        campaign_id: str,
        include_results: bool = True
    ) -> Dict[str, Any]:
        """Export campaign configuration and optionally results"""
        try:
            # Load campaign
            campaign = await self.fs.load_campaign(campaign_id)
            if not campaign:
                return {
                    "status": "error",
                    "message": f"Campaign '{campaign_id}' not found"
                }
            
            export_data = {
                "export_version": "1.0",
                "export_date": datetime.now().isoformat(),
                "campaign": campaign
            }
            
            # Include results if requested
            if include_results:
                experiments = []
                for exp_id in campaign.get("completed_experiments", []):
                    exp = await self.fs.load_experiment(campaign_id, exp_id)
                    if exp:
                        experiments.append(exp)
                
                export_data["experiments"] = experiments
                
                # Include analysis if available
                analysis = await self.fs.load_analysis(campaign_id)
                if analysis:
                    export_data["analysis"] = analysis
            
            # Save export file
            export_path = await self.fs.save_export(campaign_id, export_data)
            
            return {
                "status": "success",
                "campaign_id": campaign_id,
                "export_path": export_path,
                "included_results": include_results,
                "message": f"Campaign '{campaign_id}' exported successfully"
            }
            
        except Exception as e:
            logger.error(f"Error exporting campaign: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def import_campaign(
        self,
        export_data: Dict[str, Any],
        new_campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Import campaign from exported configuration"""
        try:
            # Validate export data
            if "campaign" not in export_data:
                return {
                    "status": "error",
                    "message": "Invalid export data: missing 'campaign' field"
                }
            
            campaign = export_data["campaign"].copy()
            original_id = campaign["id"]
            
            # Use new ID if provided
            if new_campaign_id:
                campaign["id"] = new_campaign_id
            else:
                # Generate unique ID
                campaign["id"] = f"{original_id}-import-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Reset campaign state
            campaign["completed_experiments"] = []
            campaign["imported_from"] = original_id
            campaign["imported_at"] = datetime.now().isoformat()
            
            # Create the campaign
            result = await self.create_campaign(campaign)
            
            if result["status"] == "success":
                result["imported_from"] = original_id
                result["message"] = f"Campaign imported successfully as '{campaign['id']}'"
            
            return result
            
        except Exception as e:
            logger.error(f"Error importing campaign: {e}")
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
            
            # CRITICAL: Require either prompt_template or prompt_content
            if not config.get("prompt_template") and not config.get("prompt_content"):
                return {
                    "status": "error",
                    "message": "Campaign MUST include either 'prompt_template' (template name) or 'prompt_content' (direct prompt text)"
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
                "current_experiment": 1,
                "prompt_template": config.get("prompt_template"),
                "prompt_content": config.get("prompt_content"),
                "template_variables": config.get("template_variables", {}),
                "knowledge_base": config.get("knowledge_base")
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
        # Get prompt content from campaign
        prompt_base = ""
        
        # Option 1: Load from template
        template_name = campaign.get("prompt_template")
        if template_name:
            template = await self.fs.load_template(template_name)
            if template:
                prompt_base = template
            else:
                raise ValueError(f"Template '{template_name}' not found!")
        
        # Option 2: Use direct prompt content
        elif campaign.get("prompt_content"):
            prompt_base = campaign["prompt_content"]
        
        # This should never happen due to validation
        else:
            raise ValueError("No prompt template or content found in campaign!")
        
        # Apply template variables if provided
        template_vars = campaign.get("template_variables", {})
        if template_vars:
            prompt_base = self._apply_template_variables(prompt_base, template_vars)
        
        # Add knowledge base if specified
        kb_name = campaign.get("knowledge_base")
        if kb_name:
            kb = await self.fs.load_knowledge_base(kb_name)
            if kb:
                prompt_base = f"{prompt_base}\n\n## Knowledge Base\n\n{kb}"
        
        # Add experiment header
        exp_num = int(experiment_id.split("_")[1])
        return f"""# Experiment {exp_num} - {campaign.get('name', 'Unknown Campaign')}

{campaign.get('description', '')}

{prompt_base}"""
    
    def _apply_template_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Apply variable substitution to template"""
        import re
        
        # Find all {{variable}} patterns
        pattern = r'\{\{(\w+)\}\}'
        
        def replace_var(match):
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            else:
                # Keep original if variable not provided
                return match.group(0)
        
        return re.sub(pattern, replace_var, template)
    
    # ============= Phase 1 Enhancements =============
    
    async def save_template(
        self,
        name: str,
        content: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Save a new prompt template"""
        try:
            # Validate name
            if not name or not name.strip():
                return {
                    "status": "error",
                    "message": "Template name cannot be empty"
                }
            
            # Ensure .md extension
            if not name.endswith('.md'):
                name = f"{name}.md"
            
            # Check if template already exists
            existing = await self.fs.load_template(name)
            if existing:
                return {
                    "status": "error",
                    "message": f"Template '{name}' already exists. Use a different name or delete the existing template first."
                }
            
            # Save template
            success = await self.fs.save_template(name, content, description)
            
            if success:
                return {
                    "status": "success",
                    "template_name": name,
                    "message": f"Template '{name}' saved successfully",
                    "description": description
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save template"
                }
                
        except Exception as e:
            logger.error(f"Error saving template: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def list_templates(self) -> Dict[str, Any]:
        """List available templates with descriptions"""
        try:
            templates = await self.fs.list_templates()
            
            return {
                "status": "success",
                "templates": templates,
                "total_templates": len(templates)
            }
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def preview_template(self, name: str) -> Dict[str, Any]:
        """Preview template content before using"""
        try:
            # Ensure .md extension
            if not name.endswith('.md'):
                name = f"{name}.md"
            
            content = await self.fs.load_template(name)
            
            if content is None:
                return {
                    "status": "error",
                    "message": f"Template '{name}' not found"
                }
            
            # Get metadata if available
            metadata = await self.fs.get_template_metadata(name)
            
            return {
                "status": "success",
                "template_name": name,
                "content": content,
                "description": metadata.get("description", "") if metadata else "",
                "created_at": metadata.get("created_at") if metadata else None,
                "word_count": len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error previewing template: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def clone_campaign(
        self,
        source_campaign_id: str,
        new_campaign_id: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Clone an existing campaign with optional modifications"""
        try:
            # Load source campaign
            source = await self.fs.load_campaign(source_campaign_id)
            if not source:
                return {
                    "status": "error",
                    "message": f"Source campaign '{source_campaign_id}' not found"
                }
            
            # Check if new campaign already exists
            existing = await self.fs.load_campaign(new_campaign_id)
            if existing:
                return {
                    "status": "error",
                    "message": f"Campaign '{new_campaign_id}' already exists"
                }
            
            # Create new campaign config
            new_config = {
                "id": new_campaign_id,
                "name": source.get("name", "") + " (Clone)",
                "description": source.get("description", ""),
                "providers": source.get("providers", ["anthropic-local"]),
                "total_experiments": source.get("total_experiments", 1),
                "prompt_template": source.get("prompt_template"),
                "prompt_content": source.get("prompt_content"),
                "template_variables": source.get("template_variables", {}),
                "knowledge_base": source.get("knowledge_base"),
                "evaluation_metrics": source.get("evaluation_metrics", []),
                "cloned_from": source_campaign_id,
                "cloned_at": datetime.now().isoformat()
            }
            
            # Apply modifications if provided
            if modifications:
                # Update name if modified
                if "name" in modifications:
                    new_config["name"] = modifications["name"]
                else:
                    # Only append (Clone) if name wasn't explicitly set
                    new_config["name"] = new_config["name"]
                
                # Apply other modifications
                for key, value in modifications.items():
                    if key != "id":  # Don't allow ID override
                        new_config[key] = value
            
            # Create the new campaign
            result = await self.create_campaign(new_config)
            
            if result["status"] == "success":
                result["cloned_from"] = source_campaign_id
                result["message"] = f"Campaign '{new_campaign_id}' cloned successfully from '{source_campaign_id}'"
            
            return result
            
        except Exception as e:
            logger.error(f"Error cloning campaign: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
