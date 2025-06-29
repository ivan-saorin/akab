#!/usr/bin/env python
"""Setup script for AKAB development environment."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main():
    """Setup AKAB development environment."""
    print("AKAB Development Setup")
    print("======================")
    
    project_path = Path(__file__).parent
    substrate_path = project_path.parent / "substrate"
    
    # Check if substrate exists
    if not substrate_path.exists():
        print(f"Error: Substrate not found at {substrate_path}")
        print("Please install substrate first!")
        return 1
    
    # Create virtual environment if it doesn't exist
    venv_path = project_path / "venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command([sys.executable, "-m", "venv", "venv"], cwd=project_path):
            return 1
            
    # Activate virtual environment and install
    if sys.platform == "win32":
        pip_cmd = [str(venv_path / "Scripts" / "pip.exe")]
        activate_cmd = str(venv_path / "Scripts" / "activate.bat")
    else:
        pip_cmd = [str(venv_path / "bin" / "pip")]
        activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
        
    # Upgrade pip
    print("Upgrading pip...")
    run_command(pip_cmd + ["install", "--upgrade", "pip"], cwd=project_path)
    
    # Install substrate first
    print("Installing Substrate dependency...")
    if not run_command(pip_cmd + ["install", "-e", str(substrate_path)], cwd=project_path):
        print("Failed to install Substrate!")
        return 1
    
    # Install AKAB in editable mode
    print("Installing AKAB in editable mode...")
    if not run_command(pip_cmd + ["install", "-e", "."], cwd=project_path):
        return 1
        
    # Install dev dependencies
    print("Installing development dependencies...")
    run_command(pip_cmd + ["install", "-e", ".[dev]"], cwd=project_path)
    
    # Install provider dependencies
    print("Installing provider dependencies...")
    run_command(pip_cmd + ["install", "-e", ".[providers]"], cwd=project_path)
    
    # Copy env example
    env_example = project_path / ".env.example"
    env_file = project_path / ".env"
    if env_example.exists() and not env_file.exists():
        print("Creating .env file from example...")
        import shutil
        shutil.copy(env_example, env_file)
        print("Please edit .env and add your API keys!")
    
    print("\nSetup Complete!")
    print(f"To activate the environment: {activate_cmd}")
    print("\nNext steps:")
    print("1. Edit .env and add your API keys")
    print("2. Run: python -m akab")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
