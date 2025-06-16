"""
AKAB MCP Server - Adaptive Knowledge Acquisition Benchmark
Main server implementation using FastMCP
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from akab.filesystem import FileSystemManager
from akab.providers import ProviderManager, ProviderType
from akab.tools.akab_tools import AKABTools
from akab.evaluation import EvaluationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = os.getenv("DATA_PATH", "/data/akab")
SERVER_NAME = os.getenv("MCP_SERVER_NAME", "AKAB AI Server")

# Initialize FastMCP server
mcp = FastMCP(SERVER_NAME)

# Global instances
fs_manager = FileSystemManager(DATA_PATH)
provider_manager = ProviderManager()
eval_engine = EvaluationEngine()
akab_tools = AKABTools(fs_manager, provider_manager, eval_engine)

# Track active remote execution
active_remote_campaign: Optional[str] = None
remote_execution_task: Optional[asyncio.Task] = None

# Register all AKAB tools
@mcp.tool()
async def akab_get_meta_prompt(prompt_type: str = "execution") -> Dict[str, Any]:
    """Get meta prompt for AKAB operations"""
    return await akab_tools.get_meta_prompt(prompt_type)

@mcp.tool()
async def akab_get_next_experiment() -> Dict[str, Any]:
    """Get the next experiment in the current campaign"""
    return await akab_tools.get_next_experiment()

@mcp.tool()
async def akab_get_exp_prompt(experiment_id: str) -> Dict[str, Any]:
    """Get the full prompt for execution"""
    return await akab_tools.get_exp_prompt(experiment_id)

@mcp.tool()
async def akab_save_exp_result(
    experiment_id: str,
    response: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Save experiment results and auto-analyze"""
    return await akab_tools.save_exp_result(experiment_id, response, metadata)

@mcp.tool()
async def akab_get_campaign_status() -> Dict[str, Any]:
    """Get current campaign status"""
    return await akab_tools.get_campaign_status()

@mcp.tool()
async def akab_create_campaign(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new experiment campaign"""
    return await akab_tools.create_campaign(config)

@mcp.tool()
async def akab_analyze_results(campaign_id: str) -> Dict[str, Any]:
    """Analyze aggregated results from a campaign"""
    return await akab_tools.analyze_results(campaign_id)

@mcp.tool()
async def akab_list_campaigns() -> Dict[str, Any]:
    """List all available campaigns"""
    return await akab_tools.list_campaigns()

@mcp.tool()
async def akab_switch_campaign(campaign_id: str) -> Dict[str, Any]:
    """Switch the active campaign for subsequent experiment operations"""
    return await akab_tools.switch_campaign(campaign_id)

@mcp.tool()
async def akab_get_current_campaign() -> Dict[str, Any]:
    """Get information about the currently active campaign"""
    return await akab_tools.get_current_campaign()

@mcp.tool()
async def akab_get_campaign_results(
    campaign_id: str,
    format: str = "structured"
) -> Dict[str, Any]:
    """Get all experiment results from a campaign for analysis"""
    return await akab_tools.get_campaign_results(campaign_id, format)

@mcp.tool()
async def akab_batch_execute_remote(
    campaign_id: str,
    providers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Launch batch execution for a campaign using remote providers"""
    global active_remote_campaign, remote_execution_task
    
    # Check if already running
    if active_remote_campaign and remote_execution_task and not remote_execution_task.done():
        return {
            "status": "error",
            "message": f"Campaign '{active_remote_campaign}' is already running. Stop it first or wait for completion.",
            "active_campaign": active_remote_campaign
        }
    
    # Validate campaign exists
    campaign = await fs_manager.load_campaign(campaign_id)
    if not campaign:
        return {
            "status": "error",
            "message": f"Campaign '{campaign_id}' not found"
        }
    
    # Get experiments to run
    experiments = await akab_tools.get_campaign_experiments(campaign_id, providers)
    if not experiments:
        return {
            "status": "error",
            "message": "No experiments to run for specified providers"
        }
    
    # Cost estimation
    total_experiments = len(experiments)
    cost_estimate = await provider_manager.estimate_campaign_cost(experiments)
    
    # Warning for large campaigns
    warning = None
    if total_experiments > 20:
        warning = f"⚠️ About to execute {total_experiments} experiments. Estimated cost: ${cost_estimate:.2f}"
    
    # Start background execution
    active_remote_campaign = campaign_id
    remote_execution_task = asyncio.create_task(
        akab_tools.execute_campaign_remote(campaign_id, experiments)
    )
    
    return {
        "status": "started",
        "campaign_id": campaign_id,
        "total_experiments": total_experiments,
        "estimated_cost": cost_estimate,
        "warning": warning,
        "message": f"Batch execution started for campaign '{campaign_id}'"
    }

@mcp.tool()
async def akab_get_execution_status() -> Dict[str, Any]:
    """Get status of current remote execution"""
    global active_remote_campaign, remote_execution_task
    
    if not active_remote_campaign:
        return {
            "status": "idle",
            "message": "No active remote execution"
        }
    
    if remote_execution_task and not remote_execution_task.done():
        # Get progress from campaign status
        progress = await akab_tools.get_remote_execution_progress(active_remote_campaign)
        return {
            "status": "running",
            "campaign_id": active_remote_campaign,
            **progress
        }
    
    # Execution completed
    result = {
        "status": "completed",
        "campaign_id": active_remote_campaign,
        "message": "Remote execution completed"
    }
    
    # Clean up
    active_remote_campaign = None
    remote_execution_task = None
    
    return result

# Health check endpoint for Docker
@mcp.resource("health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": SERVER_NAME,
        "version": "1.0.0",
        "providers": provider_manager.list_providers()
    }

# For Docker compatibility using uvicorn
app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting {SERVER_NAME} on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
