"""
AKAB - Adaptive Knowledge Acquisition Benchmark
Built on Substrate foundation layer
"""

__version__ = "2.0.0"
__author__ = "Ivan Saorin"

# AKAB-specific components
from .server import mcp, app
from .filesystem import AKABFileSystemManager
from .tools.akab_tools import AKABTools

# Note: These components are now provided by substrate:
# - FastMCP (from mcp.server)
# - ProviderManager (from providers)
# - EvaluationEngine (from evaluation)
# - Base FileSystemManager (from filesystem)

__all__ = [
    "mcp",
    "app",
    "AKABFileSystemManager",
    "AKABTools"
]
