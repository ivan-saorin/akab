"""AKAB FastMCP Server - Built on Substrate Foundation"""
import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path
from fastmcp import FastMCP

# Import everything from substrate foundation
from substrate.shared.models import get_model_registry
from substrate.shared.response import ResponseBuilder
from substrate.shared.storage import ReferenceManager
from substrate.shared.prompts import PromptLoader
from substrate.shared.api import ClearHermes

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("akab")

# Create shared instances using substrate components
response_builder = ResponseBuilder("akab")
reference_manager = ReferenceManager()
prompt_loader = PromptLoader()

# Initialize core AKAB components
from .core.laboratory import LABORATORY
from .core.vault import CampaignVault

# Create vault instance
vault = CampaignVault()

# Global registry for active executors (for callbacks)
ACTIVE_EXECUTORS = {}

# Global registry for scrambled models (Level 3 experiments)
SCRAMBLED_MODELS = {}

# Initialize scrambled models at module load time
try:
    registry = get_model_registry()
    SCRAMBLED_MODELS = registry.get_scrambled_models()
    logger.info(f"Initialized {len(SCRAMBLED_MODELS)} scrambled model mappings")
except Exception as e:
    logger.warning(f"Could not initialize scrambled models: {e}")
    SCRAMBLED_MODELS = {}

# Import AKAB-specific features
from .features.quick_compare import register_quick_compare_tools
from .features.campaigns import register_campaign_tools
from .features.experiments import register_experiment_tools
from .features.reporting import register_reporting_tools

# Register all features
def register_all_features():
    """Register all AKAB features"""
    logger.info("Registering AKAB features...")
    
    features = [
        register_quick_compare_tools,
        register_campaign_tools,
        register_experiment_tools,
        register_reporting_tools
    ]
    
    for register_func in features:
        try:
            tools = register_func(mcp, response_builder, reference_manager)
            logger.info(f"Registered {len(tools)} tools from {register_func.__module__}")
        except Exception as e:
            logger.error(f"Failed to register {register_func.__module__}: {e}")

# Base tool - self-documentation
@mcp.tool()
async def akab() -> Dict[str, Any]:
    """Get AKAB server capabilities and documentation"""
    return response_builder.success(
        data={
            "name": "akab",
            "version": "2.0.0",
            "description": "A/B Testing Framework with scientific rigor",
            "documentation": {
                "summary": "Scientific A/B testing with three levels of rigor",
                "usage": """AKAB provides three levels of testing:
                
Level 1 - Quick Compare:
- No blinding, immediate results
- Perfect for rapid iteration
- Use: akab_quick_compare

Level 2 - Campaigns:
- Execution blinding with unlock capability
- Automated winner selection
- Use: akab_create_campaign

Level 3 - Experiments:
- Complete triple blinding
- Statistical significance required
- Use: akab_create_experiment
""",
                "levels": {
                    "1": "Quick Compare - No blinding, rapid iteration",
                    "2": "Campaigns - Execution blinding, automated analysis",
                    "3": "Experiments - Triple blinding, statistical significance"
                }
            },
            "model_registry": get_model_registry().get_summary() if hasattr(get_model_registry(), 'get_summary') else {"providers": [], "models": 0}
        },
        message="AKAB ready. Choose your testing rigor level.",
        suggestions=[
            response_builder.suggest_next(
                "akab_quick_compare",
                "Quick comparison across providers",
                prompt="Your prompt here"
            ),
            response_builder.suggest_next(
                "akab_create_campaign",
                "Create blinded A/B test campaign"
            ),
            response_builder.suggest_next(
                "atlas_documentation",
                "Read AKAB methodology",
                doc_type="akab-methodology"
            )
        ]
    )

# Sampling callback - delegates to laboratory
@mcp.tool()
async def akab_sampling_callback(request_id: str, response: str) -> Dict[str, Any]:
    """Handle sampling callback responses for multi-turn execution"""
    # Check if we have an active executor for this request
    if request_id in ACTIVE_EXECUTORS:
        executor = ACTIVE_EXECUTORS[request_id]
        try:
            result = await executor.handle_callback(request_id, response)
            return response_builder.success(
                data=result,
                message="Callback processed successfully"
            )
        except Exception as e:
            logger.error(f"Error handling callback: {e}")
            return response_builder.error(str(e))
    else:
        return response_builder.error(f"No active executor for request {request_id}")

if __name__ == "__main__":
    # Register all features
    register_all_features()
    
    # Run the server
    logger.info("Starting AKAB FastMCP server...")
    mcp.run(transport='stdio')
