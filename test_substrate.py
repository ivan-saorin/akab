#!/usr/bin/env python3
"""
Test script to verify substrate integration
"""
import asyncio
import sys
import os

# Test imports from substrate
try:
    from mcp.server import FastMCP
    print("✓ FastMCP imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FastMCP: {e}")
    sys.exit(1)

try:
    from providers import ProviderManager
    print("✓ ProviderManager imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ProviderManager: {e}")
    sys.exit(1)

try:
    from evaluation import EvaluationEngine
    print("✓ EvaluationEngine imported successfully")
except ImportError as e:
    print(f"✗ Failed to import EvaluationEngine: {e}")
    sys.exit(1)

try:
    from filesystem import FileSystemManager
    print("✓ FileSystemManager imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FileSystemManager: {e}")
    sys.exit(1)

# Test AKAB imports
try:
    from akab.filesystem import AKABFileSystemManager
    print("✓ AKABFileSystemManager imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AKABFileSystemManager: {e}")
    sys.exit(1)

try:
    from akab.tools.akab_tools import AKABTools
    print("✓ AKABTools imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AKABTools: {e}")
    sys.exit(1)


async def test_basic_functionality():
    """Test basic functionality"""
    print("\n--- Testing Basic Functionality ---")
    
    # Test FastMCP
    mcp = FastMCP("Test Server")
    print(f"✓ Created FastMCP server: {mcp.name}")
    
    # Test ProviderManager
    providers = ProviderManager()
    provider_list = providers.list_providers()
    print(f"✓ ProviderManager initialized with {len(provider_list)} providers")
    for p in provider_list:
        print(f"  - {p['name']}: {'available' if p['available'] else 'not available'}")
    
    # Test EvaluationEngine
    evaluator = EvaluationEngine()
    test_response = "This is an innovative breakthrough solution using cutting-edge technology."
    innovation_score = evaluator.calculate_innovation_score(test_response)
    print(f"✓ EvaluationEngine test - Innovation score: {innovation_score}")
    
    # Test FileSystemManager
    fs = AKABFileSystemManager("/tmp/akab_test")
    campaigns = await fs.list_campaigns()
    print(f"✓ AKABFileSystemManager initialized - {len(campaigns)} campaigns found")
    
    # Test AKABTools
    tools = AKABTools(fs, providers, evaluator)
    meta_prompt = await tools.get_meta_prompt()
    print(f"✓ AKABTools initialized - Meta prompt status: {meta_prompt['status']}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    print("=== Substrate Integration Test ===\n")
    asyncio.run(test_basic_functionality())
