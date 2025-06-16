# AKAB Campaign Template

Use this template as a starting point for creating new campaigns. Copy and modify the JSON structure below:

```json
{
  "id": "your-campaign-id",
  "name": "Your Campaign Name",
  "description": "A clear description of what this campaign tests",
  "providers": [
    "anthropic-local",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4-turbo"
  ],
  "total_experiments": 10,
  "prompt_template": "optional_template_name",
  "knowledge_base": "optional_kb_name.md",
  "evaluation_metrics": [
    "innovation_score",
    "practicality_score", 
    "coherence_score",
    "bs_count"
  ],
  "config": {
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "tags": ["tag1", "tag2"]
}
```

## Field Descriptions

- **id**: Unique identifier (lowercase, hyphens for spaces)
- **name**: Human-readable name
- **description**: What you're testing and why
- **providers**: List of AI providers to test (see available providers below)
- **total_experiments**: Total number of experiments to run
- **prompt_template**: (Optional) Name of template in /templates folder
- **knowledge_base**: (Optional) KB file to include
- **evaluation_metrics**: Metrics to track
- **config**: Model parameters
- **tags**: Categories for organization

## Available Providers

- `anthropic-local` - Claude via MCP (free)
- `openai/gpt-3.5-turbo` - Fast and cheap ($0.0005/1K tokens)
- `openai/gpt-4-turbo` - Most capable GPT ($0.01/1K tokens)
- `anthropic-api/claude-3-opus` - Most capable Claude ($0.015/1K tokens)
- `anthropic-api/claude-3-sonnet` - Balanced Claude ($0.003/1K tokens)
- `google/gemini-pro` - Google's model ($0.0005/1K tokens)

## Example: Wikipedia Emulation Test

```json
{
  "id": "wikipedia-test",
  "name": "Wikipedia Page Emulation Test",
  "description": "Compare how different models create Wikipedia-style articles",
  "providers": [
    "anthropic-local",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4-turbo"
  ],
  "total_experiments": 3,
  "config": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "custom_prompt": "Create a comprehensive Wikipedia-style article about 'The Phenomenon of Synchronicity in Urban Bird Flocking Patterns'. Include sections: overview, history, scientific explanations, notable examples, cultural significance, and references."
  }
}
```