"""Example: Quick comparison across providers."""

import asyncio
import os
from dotenv import load_dotenv

# This example shows how to use AKAB programmatically
# In practice, you'd use it through Claude Desktop MCP

# Load environment variables
load_dotenv()


async def quick_compare_example():
    """Demonstrate quick comparison functionality."""
    
    # Example 1: Simple comparison
    print("Example 1: Simple comparison")
    print("-" * 50)
    
    prompt = "Explain the concept of recursion in programming"
    providers = ["anthropic_m", "openai_m"]
    
    print(f"Prompt: {prompt}")
    print(f"Providers: {providers}")
    print("\nIn Claude Desktop, you would say:")
    print('Use akab to compare "Explain the concept of recursion in programming" across anthropic_m and openai_m')
    
    # Example 2: Comparison with constraints
    print("\n\nExample 2: Comparison with constraints")
    print("-" * 50)
    
    prompt = "Write a haiku about artificial intelligence"
    providers = ["anthropic_s", "anthropic_m", "openai_s"]
    constraints = {
        "max_tokens": 50,
        "temperature": 0.9
    }
    
    print(f"Prompt: {prompt}")
    print(f"Providers: {providers}")
    print(f"Constraints: {constraints}")
    print("\nIn Claude Desktop, you would say:")
    print('Use akab to compare "Write a haiku about AI" across anthropic_s, anthropic_m, and openai_s with max_tokens=50 and temperature=0.9')
    
    # Example 3: Code generation comparison
    print("\n\nExample 3: Code generation comparison")
    print("-" * 50)
    
    prompt = """
    Write a Python function that implements binary search.
    Include docstring and type hints.
    """
    providers = ["anthropic_m", "openai_m", "google_m"]
    
    print(f"Prompt: {prompt.strip()}")
    print(f"Providers: {providers}")
    print("\nIn Claude Desktop, you would say:")
    print("Use akab to compare a binary search implementation across anthropic_m, openai_m, and google_m")
    
    # Example 4: Getting cost estimates
    print("\n\nExample 4: Cost awareness")
    print("-" * 50)
    
    print("AKAB automatically tracks costs for each comparison.")
    print("You can ask for cost reports:")
    print("\nIn Claude Desktop:")
    print("Show me the AKAB cost report for today")
    print("What's the most cost-effective provider for code generation?")


if __name__ == "__main__":
    print("AKAB Quick Compare Examples")
    print("=" * 50)
    print("Note: These examples show what you would do in Claude Desktop")
    print("AKAB is designed to be used through the MCP interface\n")
    
    asyncio.run(quick_compare_example())
