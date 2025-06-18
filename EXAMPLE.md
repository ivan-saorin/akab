# Example: Using AKAB with Substrate

This example demonstrates how AKAB leverages the Substrate foundation.

## 1. Understanding the Architecture

```python
# These imports come from Substrate (base image)
from mcp.server import FastMCP
from providers import ProviderManager
from evaluation import EvaluationEngine

# This import is AKAB-specific
from akab.filesystem import AKABFileSystemManager
```

## 2. Creating a Simple Campaign

```python
# Create a campaign to test different models
campaign = {
    "id": "model-comparison-2025",
    "name": "Compare Latest Models",
    "total_experiments": 6,
    "providers": [
        "anthropic-local",      # Free - Claude via MCP
        "openai/gpt-4-turbo",   # ~$0.01 per experiment
        "anthropic-api/claude-3-opus"  # ~$0.015 per experiment
    ],
    "prompt_content": """
Analyze the following business scenario and provide recommendations:

A mid-size retail company wants to expand their e-commerce presence 
while maintaining their physical stores. They have a budget of $500K 
and 6 months to implement. What should be their strategy?

Provide:
1. Top 3 priorities
2. Timeline
3. Budget allocation
4. Success metrics
"""
}
```

## 3. How Substrate Components Work

### Providers (from Substrate)
```python
# The ProviderManager handles all LLM interactions
providers = ProviderManager()
providers.list_providers()
# Returns:
# [
#   {"name": "anthropic-local", "type": "local", "available": true},
#   {"name": "openai/gpt-4-turbo", "type": "remote", "available": true},
#   {"name": "anthropic-api/claude-3-opus", "type": "remote", "available": true}
# ]
```

### Evaluation Engine (from Substrate)
```python
# Automatically scores each response
evaluator = EvaluationEngine()
# Metrics include:
# - innovation_score (0-10)
# - coherence_score (0-10)
# - practicality_score (0-10)
# - bs_count (buzzword detection)
# - key_concepts (extracted automatically)
```

### MCP Server (from Substrate)
```python
# FastMCP provides the tool interface
mcp = FastMCP("AKAB Server")

@mcp.tool()
async def akab_get_next_experiment():
    # AKAB-specific logic here
    pass
```

## 4. Complete Workflow Example

```python
# 1. Create campaign
await akab_create_campaign(campaign)

# 2. Run experiments
for i in range(6):
    # Get next experiment
    exp = await akab_get_next_experiment()
    # Returns: {"experiment_id": "exp_001", "provider": "anthropic-local", ...}
    
    # Get prompt
    prompt = await akab_get_exp_prompt(exp["experiment_id"])
    
    # Execute (handled by substrate's provider system)
    # For local: you execute and provide response
    # For remote: system calls API automatically
    
    # Save results
    await akab_save_exp_result(
        exp["experiment_id"],
        response,
        {"execution_time": 2.5}
    )

# 3. Analyze results (using substrate's evaluation engine)
analysis = await akab_analyze_results("model-comparison-2025")
```

## 5. What Each Layer Provides

### Substrate Provides:
- **FastMCP**: Tool registration and HTTP server
- **Providers**: Unified interface to all LLMs
- **Evaluation**: Consistent scoring across all responses
- **FileSystem**: Async file operations

### AKAB Adds:
- **Campaign Management**: Experiment organization
- **Prompt Templates**: Reusable prompts with variables
- **Knowledge Bases**: Attach domain knowledge
- **Experiment Tracking**: Progress and results management
- **Analysis Tools**: Aggregate insights across experiments

## 6. Cost Example

For the campaign above:
- 2 experiments with anthropic-local: FREE
- 2 experiments with GPT-4-turbo: ~$0.02
- 2 experiments with Claude Opus: ~$0.03
- **Total cost: ~$0.05** for comprehensive comparison

## 7. Results You Get

```json
{
  "campaign_name": "Compare Latest Models",
  "total_experiments": 6,
  "provider_metrics": {
    "anthropic-local": {
      "innovation_score_avg": 7.5,
      "practicality_score_avg": 8.0,
      "composite_score_avg": 7.8
    },
    "openai/gpt-4-turbo": {
      "innovation_score_avg": 7.0,
      "practicality_score_avg": 8.5,
      "composite_score_avg": 7.6
    }
  },
  "key_findings": [
    "GPT-4 more practical, Claude more innovative",
    "All models suggested similar priorities",
    "Cost difference minimal for quality gained"
  ],
  "best_provider": {
    "name": "anthropic-local",
    "composite_score": 7.8
  }
}
```

## Summary

AKAB + Substrate gives you:
1. **Scientific method** for prompt engineering
2. **Multi-provider** testing out of the box
3. **Automatic evaluation** of responses
4. **Cost tracking** and optimization
5. **Reusable components** for other AI projects

The architecture ensures that improvements to Substrate (like new providers or better evaluation) automatically benefit AKAB and all other projects built on top.
