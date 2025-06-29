"""Example: Campaign management for thorough A/B testing."""

import asyncio


async def campaign_example():
    """Demonstrate campaign management functionality."""
    
    # Example 1: Creating a prompt optimization campaign
    print("Example 1: Prompt Optimization Campaign")
    print("-" * 50)
    
    campaign_config = {
        "name": "customer_support_optimization",
        "description": "Finding the best prompt for customer support responses",
        "prompts": [
            {
                "id": "formal_tone",
                "prompt": "As a professional customer service representative, please help the customer with: {{issue}}",
                "parameters": {"issue": "refund request"}
            },
            {
                "id": "friendly_tone", 
                "prompt": "Hi there! I'd be happy to help you with your {{issue}}. Let me see what I can do!",
                "parameters": {"issue": "refund request"}
            },
            {
                "id": "empathetic_tone",
                "prompt": "I understand how frustrating {{issue}} can be. Let's work together to resolve this quickly.",
                "parameters": {"issue": "refund request"}
            }
        ],
        "providers": ["anthropic_s", "anthropic_m", "openai_m"],
        "iterations": 5,
        "constraints": {
            "max_tokens": 200,
            "temperature": 0.7
        }
    }
    
    print("Campaign Configuration:")
    print(f"- Name: {campaign_config['name']}")
    print(f"- Testing {len(campaign_config['prompts'])} prompt variations")
    print(f"- Across {len(campaign_config['providers'])} providers")
    print(f"- {campaign_config['iterations']} iterations each")
    print(f"- Total tests: {len(campaign_config['prompts']) * len(campaign_config['providers']) * campaign_config['iterations']}")
    
    print("\nIn Claude Desktop, you would say:")
    print("Create an AKAB campaign to test 3 customer support prompt variations across anthropic_s, anthropic_m, and openai_m with 5 iterations")
    
    # Example 2: Running a campaign with cost estimation
    print("\n\nExample 2: Campaign Execution with Cost Estimation")
    print("-" * 50)
    
    print("Before running a campaign, you can estimate costs:")
    print("\nIn Claude Desktop:")
    print("Estimate the cost for running campaign customer_support_optimization")
    print("\nThen execute:")
    print("Run the customer_support_optimization campaign")
    
    # Example 3: Analyzing results
    print("\n\nExample 3: Analyzing Campaign Results")
    print("-" * 50)
    
    print("After the campaign completes, analyze the results:")
    print("\nIn Claude Desktop:")
    print("Analyze the results of the customer_support_optimization campaign")
    print("Focus on response quality and cost efficiency")
    
    # Example 4: Complex campaign for model selection
    print("\n\nExample 4: Model Selection Campaign")
    print("-" * 50)
    
    model_selection_config = {
        "name": "model_selection_q4_2024",
        "description": "Comprehensive evaluation for Q4 model selection",
        "prompts": [
            # Reasoning tasks
            {"prompt": "Solve this logic puzzle: {{puzzle}}", "type": "reasoning"},
            # Coding tasks  
            {"prompt": "Implement {{algorithm}} in Python", "type": "coding"},
            # Creative tasks
            {"prompt": "Write a short story about {{theme}}", "type": "creative"},
            # Analysis tasks
            {"prompt": "Analyze the following data: {{data}}", "type": "analysis"},
        ],
        "providers": [
            "anthropic_xs", "anthropic_s", "anthropic_m", "anthropic_l",
            "openai_s", "openai_m", "openai_l",
            "google_s", "google_m"
        ],
        "iterations": 10
    }
    
    print("Large-scale model selection campaign:")
    print(f"- Testing {len(model_selection_config['prompts'])} task types")
    print(f"- Across {len(model_selection_config['providers'])} provider/size combinations")
    print(f"- {model_selection_config['iterations']} iterations")
    print(f"- Total tests: {len(model_selection_config['prompts']) * len(model_selection_config['providers']) * model_selection_config['iterations']}")
    
    print("\nThis helps you:")
    print("- Find the best model for each task type")
    print("- Understand cost/performance tradeoffs")
    print("- Make data-driven model selection decisions")


if __name__ == "__main__":
    print("AKAB Campaign Management Examples")
    print("=" * 50)
    print("Campaigns allow you to run comprehensive A/B tests")
    print("with multiple prompts, providers, and iterations.\n")
    
    asyncio.run(campaign_example())
