"""AKAB MCP Server - Open-source A/B testing for AI outputs."""

import asyncio
import ast
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context
from pydantic import Field

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import substrate - handle Docker case where it might be a stub
try:
    from substrate import (
        SubstrateMCP,
        ValidationError,
        NotFoundError,
        ProgressContext,
    )
except ImportError:
    # In Docker or local development, try different import patterns
    try:
        # Try relative import (when run as module)
        from .substrate import (
            SubstrateMCP,
            ValidationError,
            NotFoundError,
            ProgressContext,
        )
    except ImportError:
        try:
            # Try relative import of stub
            from .substrate_stub import (
                SubstrateMCP,
                ValidationError,
                NotFoundError,
                ProgressContext,
            )
        except ImportError:
            # Try absolute import of stub (for direct script execution)
            import substrate_stub
            SubstrateMCP = substrate_stub.SubstrateMCP
            ValidationError = substrate_stub.ValidationError
            NotFoundError = substrate_stub.NotFoundError
            ProgressContext = substrate_stub.ProgressContext

from .comparison import ComparisonEngine
from .campaigns import CampaignManager
from .providers import ProviderManager
from .storage import LocalStorage
from .constraints import ConstraintSuggester

logger = logging.getLogger(__name__)


class AKABServer(SubstrateMCP):
    """AKAB MCP Server for A/B testing AI outputs."""
    
    def __init__(self):
        """Initialize AKAB server."""
        super().__init__(
            name="akab",
            version="2.0.0",
            description="Open-source A/B testing tool for comparing AI outputs",
            instructions="""
            AKAB helps you scientifically compare AI outputs across different providers.
            
            Key features:
            - Quick one-shot comparisons
            - Campaign management for thorough testing
            - Cost tracking and optimization
            - Constraint suggestions via sampling
            - Support for all major LLM providers
            
            Start with akab_quick_compare() for simple comparisons or
            create campaigns for comprehensive testing.
            """
        )
        
        # Initialize components
        self.storage = LocalStorage()  # Uses AKAB_DATA_PATH env var
        self.provider_manager = ProviderManager()
        self.comparison_engine = ComparisonEngine(self.provider_manager)
        self.campaign_manager = CampaignManager(self.storage, self.comparison_engine)
        self.constraint_suggester = ConstraintSuggester(self.sampling_manager)
        
        # Register AKAB-specific tools
        self._register_tools()
        
    async def get_capabilities(self) -> Dict[str, Any]:
        """Return AKAB capabilities for self-documentation."""
        return {
            "features": [
                "quick_compare - One-shot comparison across providers",
                "campaign management - Create and run test campaigns",
                "cost tracking - Monitor and optimize costs",
                "constraint suggestions - Get help with test constraints",
                "provider support - anthropic, openai, google, more",
            ],
            "providers": self.provider_manager.list_providers(),
            "version": self.version,
        }
        
    def _register_tools(self) -> None:
        """Register AKAB-specific tools."""
        
        @self.tool(description="Quick comparison of a prompt across providers")
        async def akab_quick_compare(
            ctx: Context,
            prompt: str = Field(..., description="The prompt to test"),
            providers: List[str] = Field(
                ..., 
                description="List of providers (e.g., ['anthropic_m', 'openai_m'])"
            ),
            parameters: Optional[str] = Field(
                default=None,
                description="Optional prompt parameters as JSON string (e.g., '{\"key\": \"value\"}')"
            ),
            constraints: Optional[str] = Field(
                default=None,
                description="Optional constraints as JSON string (e.g., '{\"max_tokens\": 50, \"temperature\": 0.3}')"
            ),
        ) -> Dict[str, Any]:
            """Run a quick comparison across providers."""
            
            # Parse constraints if they come as JSON string
            constraints_parsed = self._parse_dict_param(constraints)
            parameters_parsed = self._parse_dict_param(parameters)
            
            # Validate providers
            for provider in providers:
                if not self.provider_manager.is_valid_provider(provider):
                    raise ValidationError(
                        f"Unknown provider: {provider}",
                        field="providers",
                        suggestions=self.provider_manager.list_providers()
                    )
                    
            # If no constraints, maybe suggest some
            if not constraints_parsed and self.sampling_manager.should_request_sampling("constraints"):
                response = self.create_response(
                    success=True,
                    data={
                        "status": "need_constraints",
                        "providers": providers,
                        "prompt_preview": self.truncate_prompt(prompt),
                    },
                    message="Would you like me to suggest constraints for this comparison?"
                )
                
                response["_sampling_request"] = self.sampling_manager.create_request(
                    f"The user wants to compare this prompt across {providers}. "
                    "What constraints (max_tokens, temperature, etc.) would you suggest?"
                )
                
                return response
                
            # Run comparison with progress tracking
            async with self.progress_context("quick_compare") as progress:
                await progress(0.1, "Starting comparison...")
                
                results = await self.comparison_engine.compare(
                    prompt=prompt,
                    providers=providers,
                    parameters=parameters_parsed or {},
                    constraints=constraints_parsed or {},
                    progress_callback=progress
                )
                
                await progress(0.9, "Analyzing results...")
                
                analysis = self.comparison_engine.analyze_results(results)
                
                await progress(1.0, "Complete!")
                
            return self.response_builder.comparison_result(
                results=results,
                winner=analysis.get("winner"),
                metrics=analysis.get("metrics"),
                message=f"Comparison complete! {analysis.get('summary', '')}"
            )
            
        @self.tool(description="Create a new A/B testing campaign")
        async def akab_create_campaign(
            ctx: Context,
            name: str = Field(..., description="Campaign name"),
            prompts: str = Field(
                ..., 
                description="List of prompts to test as JSON string (e.g., '[{\"name\": \"prompt1\", \"content\": \"...\"}]')"
            ),
            providers: List[str] = Field(
                ...,
                description="List of providers to test against"
            ),
            description: Optional[str] = Field(None, description="Campaign description"),
            iterations: int = Field(
                default=1,
                description="Number of iterations per test"
            ),
            constraints: Optional[str] = Field(
                default=None,
                description="Campaign-wide constraints as JSON string (e.g., '{\"max_tokens\": 100}')"
            ),
        ) -> Dict[str, Any]:
            """Create a new testing campaign."""
            
            # Parse constraints if they come as JSON string
            constraints_parsed = self._parse_dict_param(constraints)
            
            # Parse prompts if they come as JSON string
            prompts_parsed = self._parse_list_param(prompts)
            
            # Validate inputs
            if not prompts_parsed:
                raise ValidationError("At least one prompt is required", field="prompts")
                
            for provider in providers:
                if not self.provider_manager.is_valid_provider(provider):
                    raise ValidationError(
                        f"Unknown provider: {provider}",
                        field="providers"
                    )
                    
            # Create campaign
            campaign = await self.campaign_manager.create_campaign(
                name=name,
                description=description,
                prompts=prompts_parsed,
                providers=providers,
                iterations=iterations,
                constraints=constraints_parsed or {}
            )
            
            return self.create_response(
                success=True,
                data={"campaign": campaign.dict()},
                message=f"Campaign '{name}' created successfully!"
            )
            
        @self.tool(description="Execute a campaign")
        async def akab_execute_campaign(
            ctx: Context,
            campaign_id: str = Field(..., description="Campaign ID to execute"),
            dry_run: bool = Field(
                default=False,
                description="If true, estimate cost without running"
            ),
        ) -> Dict[str, Any]:
            """Execute a testing campaign."""
            
            campaign = await self.campaign_manager.get_campaign(campaign_id)
            if not campaign:
                raise NotFoundError("Campaign", campaign_id)
                
            if dry_run:
                estimate = await self.campaign_manager.estimate_cost(campaign_id)
                return self.create_response(
                    success=True,
                    data={"estimate": estimate},
                    message=f"Estimated cost: ${estimate['total_cost']:.2f}"
                )
                
            # Execute with progress
            async with self.progress_context(f"campaign_{campaign_id}") as progress:
                results = await self.campaign_manager.execute_campaign(
                    campaign_id,
                    progress_callback=progress
                )
                
            return self.create_response(
                success=True,
                data={"results": results},
                message=f"Campaign '{campaign.name}' completed!"
            )
            
        @self.tool(description="Analyze campaign results")
        async def akab_analyze_results(
            ctx: Context,
            campaign_id: str = Field(..., description="Campaign ID to analyze"),
            metrics: Optional[List[str]] = Field(
                default=None,
                description="Specific metrics to focus on"
            ),
        ) -> Dict[str, Any]:
            """Analyze results from a completed campaign."""
            
            campaign = await self.campaign_manager.get_campaign(campaign_id)
            if not campaign:
                raise NotFoundError("Campaign", campaign_id)
                
            if campaign.status != "completed":
                raise ValidationError(
                    f"Campaign is {campaign.status}, not completed",
                    suggestions=["Execute the campaign first with akab_execute_campaign"]
                )
                
            analysis = await self.campaign_manager.analyze_campaign(
                campaign_id,
                metrics=metrics
            )
            
            return self.create_response(
                success=True,
                data={"analysis": analysis},
                message="Analysis complete! See insights below."
            )
            
        @self.tool(description="List all campaigns")
        async def akab_list_campaigns(
            ctx: Context,
            status: Optional[str] = Field(
                None,
                description="Filter by status (draft, running, completed, failed)"
            ),
            limit: int = Field(default=10, description="Number of campaigns to return"),
            offset: int = Field(default=0, description="Offset for pagination"),
        ) -> Dict[str, Any]:
            """List all campaigns with optional filtering."""
            
            campaigns = await self.campaign_manager.list_campaigns(
                status=status,
                limit=limit,
                offset=offset
            )
            
            # Convert Campaign objects to dicts for serialization
            campaigns_data = [campaign.dict() for campaign in campaigns]
            
            return self.response_builder.paginated(
                items=campaigns_data,
                page=(offset // limit) + 1,
                page_size=limit,
                message=f"Found {len(campaigns)} campaigns"
            )
            
        @self.tool(description="Get cost report")
        async def akab_cost_report(
            ctx: Context,
            time_period: str = Field(
                default="week",
                description="Time period: day, week, month, all"
            ),
            group_by: str = Field(
                default="provider",
                description="Group by: provider, campaign, prompt"
            ),
        ) -> Dict[str, Any]:
            """Get cost report for testing activities."""
            
            report = await self.campaign_manager.get_cost_report(
                time_period=time_period,
                group_by=group_by
            )
            
            return self.create_response(
                success=True,
                data={"report": report},
                message=f"Cost report for {time_period}"
            )
            
    def _parse_dict_param(self, param: Any) -> Dict[str, Any]:
        """Parse a parameter that should be a dict but might be a JSON string.
        
        Args:
            param: Parameter that might be dict or JSON string
            
        Returns:
            Parsed dictionary
        """
        if param is None:
            return {}
        if isinstance(param, dict):
            return param
        if isinstance(param, str):
            # First try as JSON
            try:
                import json
                return json.loads(param)
            except (json.JSONDecodeError, ValueError):
                # If JSON parse fails, try to parse as Python dict literal
                try:
                    import ast
                    # Remove any extraneous whitespace and newlines
                    cleaned = param.strip()
                    # Safely evaluate as Python literal
                    result = ast.literal_eval(cleaned)
                    if isinstance(result, dict):
                        return result
                except (ValueError, SyntaxError):
                    pass
                logger.warning(f"Failed to parse parameter as dict: {param}")
                return {}
        return {}
    
    def _parse_list_param(self, param: Any) -> List[Any]:
        """Parse a parameter that should be a list but might be a JSON string.
        
        Args:
            param: Parameter that might be list or JSON string
            
        Returns:
            Parsed list
        """
        if param is None:
            return []
        if isinstance(param, list):
            return param
        if isinstance(param, str):
            # First try as JSON
            try:
                import json
                result = json.loads(param)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                # If JSON parse fails, try to parse as Python list literal
                try:
                    import ast
                    cleaned = param.strip()
                    result = ast.literal_eval(cleaned)
                    if isinstance(result, list):
                        return result
                except (ValueError, SyntaxError):
                    pass
            logger.warning(f"Failed to parse parameter as list: {param}")
            return []
        return []
    
    def truncate_prompt(self, prompt: str, max_length: int = 50) -> str:
        """Truncate prompt for preview."""
        if len(prompt) <= max_length:
            return prompt
        return prompt[:max_length] + "..."
        

def main():
    """Run the AKAB server."""
    # Configure logging from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr  # Important for Docker
    )
    
    logger.info(f"Starting AKAB MCP Server v2.0.0")
    logger.info(f"Transport: {os.getenv('MCP_TRANSPORT', 'stdio')}")
    logger.info(f"Data path: {os.getenv('AKAB_DATA_PATH', './akab_data')}")
    
    # Check API keys
    api_keys_configured = {
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
    }
    
    for provider, configured in api_keys_configured.items():
        if configured:
            logger.info(f"{provider} API key: Configured")
        else:
            logger.warning(f"{provider} API key: Not configured")
    
    # Create and run server
    server = AKABServer()
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("AKAB server stopped")
    

if __name__ == "__main__":
    main()
