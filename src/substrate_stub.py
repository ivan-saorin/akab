"""Substrate stub for Docker builds.

This provides a minimal implementation of substrate components needed by AKAB.
In production, substrate would be properly packaged and installed from PyPI.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from contextlib import asynccontextmanager

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Base types
class Campaign(BaseModel):
    """Campaign model."""
    id: str
    name: str
    description: Optional[str] = None
    prompts: List[str] = []
    providers: List[str] = []
    iterations: int = 1
    constraints: Dict[str, Any] = {}
    created_at: float = Field(default_factory=time.time)
    status: str = "draft"


class ProviderConfig(BaseModel):
    """Provider configuration."""
    name: str
    size: str
    model: str
    api_key_env: Optional[str] = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = True


class ComparisonResult(BaseModel):
    """Comparison result."""
    provider: str
    response: str
    latency_ms: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


# Utility functions
def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID."""
    import random
    import string
    chars = string.ascii_lowercase + string.digits
    random_part = "".join(random.choices(chars, k=length))
    return f"{prefix}_{random_part}" if prefix else random_part


def estimate_progress(current: int, total: int, base: float = 0.0, range: float = 1.0) -> float:
    """Estimate progress."""
    if total <= 0:
        return base
    progress = (current + 1) / total
    return base + (progress * range)


class Timer:
    """Simple timer context manager."""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    @property
    def elapsed(self):
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
        
    @property
    def elapsed_ms(self):
        return self.elapsed * 1000


# Error classes
class SubstrateError(Exception):
    """Base substrate error."""
    pass


class ValidationError(SubstrateError):
    """Validation error."""
    def __init__(self, message: str, field: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.field = field
        self.suggestions = suggestions or []


class NotFoundError(SubstrateError):
    """Not found error."""
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        message = f"{resource_type} '{resource_id}' not found"
        super().__init__(message)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Base MCP class
class SubstrateMCP:
    """Minimal SubstrateMCP implementation for Docker."""
    
    def __init__(self, name: str, version: str, description: str, *, instructions: Optional[str] = None, **kwargs):
        self.name = name
        self.version = version
        self.description = description
        self.mcp = FastMCP(f"{name} - {description}")
        
        if instructions:
            # FastMCP doesn't have add_instructions, so we'll store it
            self._instructions = instructions
            
        # Setup tools
        self._setup_standard_tools()
        
        # Components
        self.response_builder = ResponseBuilder()
        self.sampling_manager = SamplingManager()
        self.progress_tracker = ProgressTracker()
        
    def _setup_standard_tools(self):
        """Register standard tools."""
        
        # Create self-documentation tool with server name
        @self.mcp.tool(name=self.name)
        async def self_documentation(ctx: Context):
            """Get server capabilities."""
            caps = await self.get_capabilities()
            return {
                "success": True,
                "data": {
                    "name": self.name,
                    "version": self.version,
                    "description": self.description,
                    "capabilities": caps
                }
            }
        
        # Create sampling callback tool with proper name
        @self.mcp.tool(name=f"{self.name}_sampling_callback")
        async def sampling_callback(ctx: Context, request_id: str, response: str):
            """Handle sampling callback."""
            return {"success": True, "data": {"processed": True}}
        
    async def get_capabilities(self):
        """Override in subclass."""
        return {}
        
    def tool(self, **kwargs):
        """Tool decorator."""
        return self.mcp.tool(**kwargs)
        
    def run(self):
        """Run the server."""
        # FastMCP's run() handles the event loop creation
        self.mcp.run()
        
    def progress_context(self, operation_name):
        """Create progress context."""
        return ProgressContext(operation_name)
        
    def create_response(self, success=True, data=None, message=None, error=None, **kwargs):
        """Create response."""
        if success:
            return self.response_builder.success(data, message, **kwargs)
        else:
            return self.response_builder.error(error or "Unknown error", **kwargs)


class ResponseBuilder:
    """Response builder."""
    
    def success(self, data=None, message=None, **kwargs):
        """Build success response."""
        response = {
            "success": True,
            "timestamp": time.time()
        }
        if data is not None:
            response["data"] = data
        if message:
            response["message"] = message
        response.update(kwargs)
        return response
        
    def error(self, error, **kwargs):
        """Build error response."""
        response = {
            "success": False,
            "error": error,
            "timestamp": time.time()
        }
        response.update(kwargs)
        return response
        
    def comparison_result(self, results, winner=None, metrics=None, message=None):
        """Build comparison result."""
        data = {"results": results}
        if winner:
            data["winner"] = winner
        if metrics:
            data["metrics"] = metrics
        return self.success(data, message)
        
    def paginated(self, items, page=1, page_size=10, total=None, message=None):
        """Build paginated response."""
        data = {
            "items": items,
            "page": page,
            "page_size": page_size,
            "count": len(items)
        }
        if total is not None:
            data["total"] = total
        return self.success(data, message)


class SamplingManager:
    """Sampling manager."""
    
    def __init__(self):
        self._pending = {}
        
    def should_request_sampling(self, context):
        """Check if should sample."""
        return context in ["help", "constraints", "clarification"]
        
    def create_request(self, prompt, **kwargs):
        """Create sampling request."""
        request_id = str(uuid.uuid4())
        return {
            "id": request_id,
            "prompt": prompt,
            "instruction": f"Please process and respond with request_id='{request_id}'"
        }
        
    async def close(self):
        """Cleanup."""
        pass


class ProgressTracker:
    """Progress tracker."""
    
    def create_tracker(self, name):
        """Create tracker."""
        return OperationTracker(name)


class OperationTracker:
    """Operation tracker."""
    
    def __init__(self, name):
        self.name = name
        self._progress = 0.0
        
    async def progress(self, value, status):
        """Update progress."""
        self._progress = value
        logger.debug(f"{self.name}: {value:.0%} - {status}")
        
    async def complete(self):
        """Complete operation."""
        self._progress = 1.0


class ProgressContext:
    """Progress context manager."""
    
    def __init__(self, name):
        self.name = name
        self._tracker = None
        
    async def __aenter__(self):
        self._tracker = OperationTracker(self.name)
        return self._tracker.progress
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._tracker:
            await self._tracker.complete()


# Make substrate components available
__all__ = [
    "SubstrateMCP",
    "ValidationError", 
    "NotFoundError",
    "ProgressContext",
    "Campaign",
    "ProviderConfig",
    "ComparisonResult",
    "generate_id",
    "estimate_progress",
    "Timer",
    "ResponseBuilder",
    "SamplingManager",
]
