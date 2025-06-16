"""
AKAB - Adaptive Knowledge Acquisition Benchmark
"""

__version__ = "1.0.0"
__author__ = "Ivan Saorin"

from .server import mcp, app
from .filesystem import FileSystemManager
from .providers import ProviderManager
from .evaluation import EvaluationEngine

__all__ = [
    "mcp",
    "app",
    "FileSystemManager",
    "ProviderManager", 
    "EvaluationEngine"
]
