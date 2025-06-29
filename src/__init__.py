"""AKAB - Open-source A/B testing tool for AI outputs."""

from .server import AKABServer
from .comparison import ComparisonEngine
from .providers import ProviderManager
from .campaigns import CampaignManager

__version__ = "2.0.0"

__all__ = [
    "AKABServer",
    "ComparisonEngine", 
    "ProviderManager",
    "CampaignManager",
]
