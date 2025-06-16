"""
AKAB FileSystem Manager
Handles all file operations for campaigns, experiments, and results
"""
import os
import json
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiofiles
import logging

logger = logging.getLogger(__name__)


class FileSystemManager:
    """Manages all filesystem operations for AKAB"""
    
    def __init__(self, base_path: str = "/data/akab"):
        self.base_path = Path(base_path)
        self.campaigns_dir = self.base_path / "campaigns"
        self.experiments_dir = self.base_path / "experiments"
        self.kb_dir = self.base_path / "knowledge_bases"
        self.templates_dir = self.base_path / "templates"
        self.results_dir = self.base_path / "results"
        self.meta_prompt_path = self.base_path / "meta_prompt.md"
        
        # Current active campaign
        self.current_campaign_id = "default-campaign"
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for dir_path in [
            self.campaigns_dir,
            self.experiments_dir,
            self.kb_dir,
            self.templates_dir,
            self.results_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def load_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Load campaign data from JSON file"""
        campaign_path = self.campaigns_dir / f"{campaign_id}.json"
        if not campaign_path.exists():
            logger.warning(f"Campaign not found: {campaign_id}")
            return None
        
        try:
            async with aiofiles.open(campaign_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading campaign {campaign_id}: {e}")
            return None
    
    async def save_campaign(self, campaign: Dict[str, Any]) -> bool:
        """Save campaign data to JSON file"""
        campaign_id = campaign.get("id")
        if not campaign_id:
            logger.error("Campaign missing ID")
            return False
        
        campaign_path = self.campaigns_dir / f"{campaign_id}.json"
        
        # Create campaign experiment directory
        campaign_exp_dir = self.experiments_dir / campaign_id
        campaign_exp_dir.mkdir(exist_ok=True)
        
        # Create campaign results directory
        campaign_results_dir = self.results_dir / campaign_id
        campaign_results_dir.mkdir(exist_ok=True)
        
        try:
            async with aiofiles.open(campaign_path, 'w') as f:
                await f.write(json.dumps(campaign, indent=2))
            logger.info(f"Campaign saved: {campaign_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving campaign {campaign_id}: {e}")
            return False
    
    async def list_campaigns(self) -> List[Dict[str, Any]]:
        """List all available campaigns"""
        campaigns = []
        
        for campaign_file in self.campaigns_dir.glob("*.json"):
            campaign = await self.load_campaign(campaign_file.stem)
            if campaign:
                campaigns.append({
                    "id": campaign.get("id"),
                    "name": campaign.get("name"),
                    "created": campaign.get("created_at"),
                    "total_experiments": campaign.get("total_experiments", 0),
                    "completed": len(campaign.get("completed_experiments", [])),
                    "providers": campaign.get("providers", [])
                })
        
        return campaigns
    
    def get_experiment_dir(self, campaign_id: str, experiment_id: str) -> Path:
        """Get the directory path for an experiment"""
        return self.experiments_dir / campaign_id / experiment_id
    
    async def save_experiment(
        self,
        campaign_id: str,
        experiment_id: str,
        config: Dict[str, Any],
        prompt: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save experiment data"""
        exp_dir = self.get_experiment_dir(campaign_id, experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save config
            async with aiofiles.open(exp_dir / "config.json", 'w') as f:
                await f.write(json.dumps(config, indent=2))
            
            # Save prompt
            async with aiofiles.open(exp_dir / "prompt.md", 'w') as f:
                await f.write(prompt)
            
            # Save result if provided
            if result:
                async with aiofiles.open(exp_dir / "result.json", 'w') as f:
                    await f.write(json.dumps(result, indent=2))
            
            logger.info(f"Experiment saved: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving experiment {experiment_id}: {e}")
            return False
    
    async def load_experiment(
        self,
        campaign_id: str,
        experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load experiment data"""
        exp_dir = self.get_experiment_dir(campaign_id, experiment_id)
        
        if not exp_dir.exists():
            return None
        
        try:
            experiment = {"id": experiment_id}
            
            # Load config
            config_path = exp_dir / "config.json"
            if config_path.exists():
                async with aiofiles.open(config_path, 'r') as f:
                    experiment["config"] = json.loads(await f.read())
            
            # Load prompt
            prompt_path = exp_dir / "prompt.md"
            if prompt_path.exists():
                async with aiofiles.open(prompt_path, 'r') as f:
                    experiment["prompt"] = await f.read()
            
            # Load result
            result_path = exp_dir / "result.json"
            if result_path.exists():
                async with aiofiles.open(result_path, 'r') as f:
                    experiment["result"] = json.loads(await f.read())
            
            return experiment
            
        except Exception as e:
            logger.error(f"Error loading experiment {experiment_id}: {e}")
            return None
    
    async def load_knowledge_base(self, kb_name: str) -> Optional[str]:
        """Load knowledge base content"""
        kb_path = self.kb_dir / kb_name
        
        if not kb_path.exists():
            # Try with .md extension
            kb_path = self.kb_dir / f"{kb_name}.md"
            if not kb_path.exists():
                return None
        
        try:
            async with aiofiles.open(kb_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Error loading knowledge base {kb_name}: {e}")
            return None
    
    async def load_template(self, template_name: str) -> Optional[str]:
        """Load prompt template"""
        template_path = self.templates_dir / template_name
        
        if not template_path.exists():
            # Try with .md extension
            template_path = self.templates_dir / f"{template_name}.md"
            if not template_path.exists():
                return None
        
        try:
            async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            return None
    
    async def load_meta_prompt(self, prompt_type: str = "execution") -> str:
        """Load meta prompt for AKAB operations"""
        # For campaign_template, load from specific file
        if prompt_type == "campaign_template":
            template_path = self.base_path / "campaign_template.md"
            if template_path.exists():
                try:
                    async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
                        return await f.read()
                except Exception as e:
                    logger.error(f"Error loading campaign template: {e}")
        
        # Try to load from meta_prompt.md for execution type
        if prompt_type == "execution" and self.meta_prompt_path.exists():
            try:
                async with aiofiles.open(self.meta_prompt_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"Error loading meta prompt: {e}")
        
        # Return default meta prompt
        return self._get_default_meta_prompt(prompt_type)
    
    def _get_default_meta_prompt(self, prompt_type: str) -> str:
        """Get default meta prompt"""
        if prompt_type == "execution":
            return """# AKAB Experiment Execution Protocol

You are connected to AKAB (Adaptive Knowledge Acquisition Benchmark), an AI research platform for systematic AB testing. When a user asks you to run experiments, follow this workflow:

## Standard Experiment Workflow

1. **Campaign Safety Check**
   Before EVERY experiment:
   - Use `akab_get_current_campaign()` to check active campaign
   - Confirm with user: "Ready to run experiment in campaign 'X'. Proceed?"
   - If user wants different campaign: use `akab_switch_campaign(campaign_id)`

2. **Get Next Experiment**
   ```
   Use: akab_get_next_experiment()
   Returns: experiment_id, provider info, progress
   ```

3. **Retrieve Experiment Prompt**
   ```
   Use: akab_get_exp_prompt(experiment_id)
   Returns: Full prompt ready for execution
   ```

4. **Execute the Prompt**
   - For local experiments: Execute and generate response
   - For remote experiments: Automatically handled by system
   - Pay attention to any special instructions

5. **Save Results**
   ```
   Use: akab_save_exp_result(experiment_id, response, metadata)
   Where metadata includes:
   - response_length: word count
   - execution_time: estimate
   - innovation_level: 1-10 assessment
   - key_concepts: list of notable ideas
   ```

## Multi-Provider Experiments

AKAB supports testing across multiple providers:
- **Local**: Claude (via MCP) - execute directly
- **Remote**: OpenAI, Anthropic API, Google, etc. - use batch execution

For remote execution:
1. Use `akab_batch_execute_remote(campaign_id)`
2. Monitor with `akab_get_execution_status()`
3. System handles API calls automatically

## Cost Awareness

- Local experiments: Free (via Claude Desktop)
- Remote experiments: Show cost estimates
- Warning at 20+ experiments
- Real-time cost tracking during execution

## Error Handling

- If experiment fails: Note error in metadata
- If campaign not found: List available campaigns
- If switching campaigns: Always confirm with user

## Communication Style

- Report progress: "Experiment 3/10 complete"
- Show provider: "Testing with GPT-4"
- Mention costs: "Remote execution estimated at $X.XX"
- Be clear about local vs remote execution

Remember: ALWAYS confirm campaign before experiments!"""
        
        elif prompt_type == "campaign_creation":
            return """# AKAB Campaign Creation Guide

Help users create well-structured campaigns for systematic AI testing.

## Campaign Structure
```json
{
  "id": "descriptive-id",
  "name": "Human readable name",
  "description": "What we're testing",
  "providers": ["provider/model", ...],
  "total_experiments": number,
  "prompt_template": "template_name",
  "knowledge_base": "kb_name.md",
  "evaluation_metrics": ["metric1", "metric2"],
  "execution_mode": "sequential|parallel"
}
```

## Available Providers
- anthropic-local (Claude via MCP)
- openai/gpt-4-turbo
- openai/gpt-3.5-turbo
- anthropic-api/claude-3-opus
- anthropic-api/claude-3-sonnet
- google/gemini-pro
- mistral/mixtral-8x7b

## Best Practices
1. Start with 5-10 experiments for testing
2. Use meaningful campaign IDs
3. Mix providers for comparison
4. Include evaluation metrics
5. Consider costs for remote providers"""
        
        else:
            return f"# AKAB {prompt_type.title()} Guide\n\nGuide for {prompt_type} operations."
    
    async def save_results(
        self,
        campaign_id: str,
        analysis: Dict[str, Any]
    ) -> bool:
        """Save campaign analysis results"""
        results_path = self.results_dir / campaign_id / "analysis.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(results_path, 'w') as f:
                await f.write(json.dumps(analysis, indent=2))
            
            # Also save as markdown report
            report = self._generate_markdown_report(analysis)
            report_path = self.results_dir / campaign_id / "report.md"
            
            async with aiofiles.open(report_path, 'w') as f:
                await f.write(report)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def _generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown report from analysis"""
        report = f"""# Campaign Analysis Report

**Campaign**: {analysis.get('campaign_name', 'Unknown')}  
**Date**: {analysis.get('analysis_date', datetime.now().isoformat())}  
**Total Experiments**: {analysis.get('total_experiments', 0)}

## Summary

{analysis.get('summary', 'No summary available')}

## Provider Performance

"""
        
        # Add provider metrics
        for provider, metrics in analysis.get('provider_metrics', {}).items():
            report += f"\n### {provider}\n"
            for metric, value in metrics.items():
                report += f"- **{metric}**: {value}\n"
        
        # Add recommendations
        if 'recommendations' in analysis:
            report += "\n## Recommendations\n\n"
            for rec in analysis['recommendations']:
                report += f"- {rec}\n"
        
        return report
    
    def get_current_campaign_id(self) -> str:
        """Get current active campaign ID"""
        return self.current_campaign_id
    
    def set_current_campaign_id(self, campaign_id: str):
        """Set current active campaign ID"""
        self.current_campaign_id = campaign_id
        logger.info(f"Switched to campaign: {campaign_id}")
    
    # ============= Phase 1 Enhancement Methods =============
    
    async def save_template(
        self,
        name: str,
        content: str,
        description: str = ""
    ) -> bool:
        """Save a prompt template"""
        template_path = self.templates_dir / name
        
        try:
            # Save template content
            async with aiofiles.open(template_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            # Save metadata if description provided
            if description:
                metadata = {
                    "name": name,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "word_count": len(content.split())
                }
                metadata_path = self.templates_dir / f"{name}.meta.json"
                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(json.dumps(metadata, indent=2))
            
            logger.info(f"Template saved: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template {name}: {e}")
            return False
    
    async def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        templates = []
        
        try:
            for template_file in self.templates_dir.glob("*.md"):
                template_info = {
                    "name": template_file.name,
                    "path": str(template_file.relative_to(self.base_path)),
                    "size": template_file.stat().st_size,
                    "modified": datetime.fromtimestamp(template_file.stat().st_mtime).isoformat()
                }
                
                # Load metadata if exists
                metadata_path = self.templates_dir / f"{template_file.name}.meta.json"
                if metadata_path.exists():
                    try:
                        async with aiofiles.open(metadata_path, 'r') as f:
                            metadata = json.loads(await f.read())
                            template_info["description"] = metadata.get("description", "")
                            template_info["created_at"] = metadata.get("created_at")
                    except:
                        pass
                
                templates.append(template_info)
            
            return templates
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
    
    async def get_template_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get template metadata"""
        metadata_path = self.templates_dir / f"{name}.meta.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_path, 'r') as f:
                return json.loads(await f.read())
        except Exception as e:
            logger.error(f"Error loading template metadata: {e}")
            return None
    
    # ============= Phase 2 Enhancement Methods =============
    
    async def save_knowledge_base(
        self,
        name: str,
        content: str,
        description: str = ""
    ) -> bool:
        """Save a knowledge base document"""
        kb_path = self.kb_dir / name
        
        try:
            # Save KB content
            async with aiofiles.open(kb_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            # Save metadata if description provided
            if description:
                metadata = {
                    "name": name,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "size": len(content)
                }
                metadata_path = self.kb_dir / f"{name}.meta.json"
                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(json.dumps(metadata, indent=2))
            
            logger.info(f"Knowledge base saved: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base {name}: {e}")
            return False
    
    async def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """List all available knowledge bases"""
        kbs = []
        
        try:
            for kb_file in self.kb_dir.glob("*.md"):
                kb_info = {
                    "name": kb_file.name,
                    "path": str(kb_file.relative_to(self.base_path)),
                    "size": kb_file.stat().st_size,
                    "modified": datetime.fromtimestamp(kb_file.stat().st_mtime).isoformat()
                }
                
                # Load metadata if exists
                metadata_path = self.kb_dir / f"{kb_file.name}.meta.json"
                if metadata_path.exists():
                    try:
                        async with aiofiles.open(metadata_path, 'r') as f:
                            metadata = json.loads(await f.read())
                            kb_info["description"] = metadata.get("description", "")
                            kb_info["created_at"] = metadata.get("created_at")
                    except:
                        pass
                
                kbs.append(kb_info)
            
            return kbs
            
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {e}")
            return []
    
    async def save_export(
        self,
        campaign_id: str,
        export_data: Dict[str, Any]
    ) -> str:
        """Save campaign export file"""
        exports_dir = self.base_path / "exports"
        exports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{campaign_id}_export_{timestamp}.json"
        export_path = exports_dir / export_filename
        
        try:
            async with aiofiles.open(export_path, 'w') as f:
                await f.write(json.dumps(export_data, indent=2))
            
            logger.info(f"Campaign exported: {export_path}")
            return str(export_path.relative_to(self.base_path))
            
        except Exception as e:
            logger.error(f"Error saving export: {e}")
            raise
    
    async def load_analysis(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Load campaign analysis if exists"""
        analysis_path = self.results_dir / campaign_id / "analysis.json"
        
        if not analysis_path.exists():
            return None
        
        try:
            async with aiofiles.open(analysis_path, 'r') as f:
                return json.loads(await f.read())
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return None
