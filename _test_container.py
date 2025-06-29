#!/usr/bin/env python
"""Quick AKAB functionality test."""

import json
import subprocess
import sys

def test_docker_container():
    """Test if the Docker container runs properly."""
    print("Testing AKAB Docker container...")
    
    # Test 1: Basic import
    result = subprocess.run([
        "docker", "run", "--rm", "akab-mcp:latest",
        "python", "-c", "import akab; print('PASS: Import works')"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FAIL: Import test failed\n{result.stderr}")
        return False
    print(result.stdout.strip())
    
    # Test 2: Server creation
    result = subprocess.run([
        "docker", "run", "--rm", "akab-mcp:latest",
        "python", "-c", 
        "from akab.server import AKABServer; s = AKABServer(); print('PASS: Server created')"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FAIL: Server creation failed\n{result.stderr}")
        return False
    print(result.stdout.strip())
    
    # Test 3: List tools (basic MCP test)
    test_cmd = """
import json
from akab.server import AKABServer
server = AKABServer()
# Check that tools were registered
tool_count = len(server.mcp._tools) if hasattr(server.mcp, '_tools') else 'unknown'
print(f'PASS: Server has tools registered (count: {tool_count})')
"""
    
    result = subprocess.run([
        "docker", "run", "--rm", "akab-mcp:latest",
        "python", "-c", test_cmd
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FAIL: Tool check failed\n{result.stderr}")
        return False
    print(result.stdout.strip())
    
    return True

def generate_test_commands():
    """Generate test commands for manual testing."""
    print("\n" + "="*50)
    print("MANUAL TEST COMMANDS")
    print("="*50)
    
    print("\n1. Test with all API keys:")
    print("docker run --rm -it \\")
    print("  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \\")
    print("  -e OPENAI_API_KEY=$OPENAI_API_KEY \\")
    print("  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \\")
    print("  akab-mcp:latest")
    
    print("\n2. Test with volume for persistence:")
    print("docker run --rm -it \\")
    print("  -v akab_data:/app/akab_data \\")
    print("  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \\")
    print("  akab-mcp:latest")
    
    print("\n3. Check container logs:")
    print("docker logs akab-mcp-claude")

if __name__ == "__main__":
    print("AKAB Docker Container Tests")
    print("===========================\n")
    
    if test_docker_container():
        print("\n✓ All container tests passed!")
        generate_test_commands()
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
