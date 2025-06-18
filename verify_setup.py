#!/usr/bin/env python3
"""
Quick verification script for substrate + AKAB setup
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def main():
    print("=== Substrate + AKAB Setup Verification ===\n")
    
    # Check Docker
    print("1. Checking Docker...")
    success, stdout, stderr = run_command("docker --version")
    if success:
        print(f"   ✓ Docker installed: {stdout.strip()}")
    else:
        print("   ✗ Docker not found!")
        sys.exit(1)
    
    # Check if we're in the right directory
    print("\n2. Checking directory structure...")
    current_dir = Path.cwd()
    substrate_dir = current_dir.parent / "substrate"
    akab_dir = current_dir if current_dir.name == "akab" else current_dir / "akab"
    
    if substrate_dir.exists() and akab_dir.exists():
        print(f"   ✓ Found substrate at: {substrate_dir}")
        print(f"   ✓ Found akab at: {akab_dir}")
    else:
        print("   ✗ Directory structure incorrect!")
        print(f"     Expected: {current_dir.parent}/substrate and {current_dir.parent}/akab")
        sys.exit(1)
    
    # Check for substrate image
    print("\n3. Checking substrate Docker image...")
    success, stdout, _ = run_command("docker images substrate:latest --format '{{.Repository}}:{{.Tag}}'")
    if success and "substrate:latest" in stdout:
        print("   ✓ Substrate image exists")
    else:
        print("   ✗ Substrate image not found")
        print("   → Building substrate image...")
        success, _, stderr = run_command("docker build -t substrate:latest .", cwd=substrate_dir)
        if success:
            print("   ✓ Substrate image built successfully")
        else:
            print(f"   ✗ Failed to build substrate: {stderr}")
            sys.exit(1)
    
    # Check AKAB Dockerfile
    print("\n4. Checking AKAB Dockerfile...")
    dockerfile_path = akab_dir / "Dockerfile"
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            if "FROM substrate:latest" in content:
                print("   ✓ AKAB Dockerfile correctly references substrate")
            else:
                print("   ✗ AKAB Dockerfile doesn't use substrate as base")
                sys.exit(1)
    else:
        print("   ✗ AKAB Dockerfile not found")
        sys.exit(1)
    
    # Test Python imports (if in Docker context this would fail, which is OK)
    print("\n5. Testing Python imports (may fail outside Docker)...")
    try:
        sys.path.insert(0, str(substrate_dir / "src"))
        sys.path.insert(0, str(akab_dir / "src"))
        
        from mcp.server import FastMCP
        print("   ✓ FastMCP import successful")
        
        from providers import ProviderManager
        print("   ✓ ProviderManager import successful")
        
        from evaluation import EvaluationEngine
        print("   ✓ EvaluationEngine import successful")
        
        from akab.filesystem import AKABFileSystemManager
        print("   ✓ AKABFileSystemManager import successful")
        
    except ImportError as e:
        print(f"   ⚠ Import failed (expected outside Docker): {e}")
    
    print("\n=== Summary ===")
    print("✅ Substrate and AKAB are properly configured!")
    print("\nNext steps:")
    print("1. Build AKAB image: docker build -t akab:latest . (in akab directory)")
    print("2. Run AKAB: docker-compose up (or docker run -p 8000:8000 akab:latest)")
    print("\nFor development, you can use the build-all script:")
    print("  Windows: .\\build-all.ps1")
    print("  Linux: ./build-all.sh")


if __name__ == "__main__":
    main()
