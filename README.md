# AKAB - Scientific A/B Testing Framework

A production-grade A/B testing framework with three levels of scientific rigor for comparing LLM outputs.

## Overview

AKAB provides a unified testing framework with three distinct levels of rigor, from quick explorations to fully blinded scientific experiments. It's the ONLY component in the Atlas system that should handle model comparisons and A/B testing.

## Three-Level Testing Architecture

### Level 1: Quick Compare (No Blinding)

- **Purpose**: Debugging, exploration, rapid iteration
- **Features**: Direct provider visibility, immediate results
- **Use Case**: Testing prompts, exploring model behaviors
- **No Winner Selection**: Human judgment required

### Level 2: Campaign (Execution Blinding)

- **Purpose**: Standard A/B testing with debugging capability
- **Features**: Blinded execution, unlockable results, automated winner selection
- **Use Case**: Production A/B tests, performance comparisons
- **Dynamic Success Criteria**: Configurable metrics and constraints

### Level 3: Experiment (Complete Blinding)

- **Purpose**: Unbiased scientific evaluation
- **Features**: Fire-and-forget scrambling, statistical significance required
- **Use Case**: Academic research, unbiased model evaluation
- **Hypothesis Testing**: Formal experimental design

## Key Features

### Production-Grade Implementation

- **Real API Calls**: Actually executes against Anthropic, OpenAI, and Google (Gemini)
- **Real Results**: Returns actual LLM responses, not mocks
- **Working Features**: Every advertised feature is fully implemented
- **No Silent Failures**: Errors fail loudly with clear messages

### Scientific Rigor

- **Statistical Analysis**: Trimmed means (10% trim), confidence intervals, effect sizes
- **Blinding Options**: Three levels from transparent to fully scrambled
- **Reproducibility**: Complete result archival in `/krill/` directory
- **Hypothesis Testing**: Formal experiment design for Level 3

### Dynamic Success Criteria

```python
criteria = {
    "primary": {
        "metric": "quality_score",    # LLM-judged quality
        "weight": 0.7,
        "aggregation": "mean"
    },
    "secondary": {
        "metric": "speed",            # Response time
        "weight": 0.3,
        "aggregation": "p50"
    },
    "constraints": {
        "must_include": ["key phrase"],
        "max_tokens": 1000,
        "min_quality": 7.0
    }
}
```

### Intelligent Assistance

- **Constraint Suggestions**: Claude helps design effective tests
- **Error Recovery**: Intelligent guidance when things go wrong
- **Progress Tracking**: Real-time updates during execution
- **Context-Aware**: Only requests help when truly beneficial

## Supported Providers

### Currently Active

- **Anthropic**: Claude models (Haiku, Sonnet, Opus)
- **OpenAI**: GPT models (3.5-turbo, GPT-4 variants)

### Experimental Support

- **Google**: Gemini models (requires `google-generativeai` package)
  - Note: Google/Gemini support is implemented but not fully activated in the current release
  - To enable: Install `pip install google-generativeai` and set `GOOGLE_API_KEY`

## Setup

### Environment Variables

Create `.env` file with your API keys:

```bash
# Required for core functionality
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Optional for Google/Gemini support
GOOGLE_API_KEY=your_google_key  # Experimental

# Optional model overrides (defaults shown)
ANTHROPIC_XS_MODEL=claude-3-haiku-20240307
ANTHROPIC_S_MODEL=claude-3-5-haiku-20241022
ANTHROPIC_M_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_L_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_XL_MODEL=claude-3-opus-20240229
ANTHROPIC_XXL_MODEL=claude-3-opus-20240229

OPENAI_XS_MODEL=gpt-3.5-turbo
OPENAI_S_MODEL=gpt-4o-mini
OPENAI_M_MODEL=gpt-4
OPENAI_L_MODEL=gpt-4-turbo
OPENAI_XL_MODEL=gpt-4-turbo-preview
OPENAI_XXL_MODEL=gpt-4-turbo-preview

# Google models (when enabled)
GOOGLE_XS_MODEL=gemini-1.5-flash
GOOGLE_S_MODEL=gemini-1.5-flash
GOOGLE_M_MODEL=gemini-1.5-pro
GOOGLE_L_MODEL=gemini-1.5-pro
GOOGLE_XL_MODEL=gemini-1.5-pro
GOOGLE_XXL_MODEL=gemini-1.5-pro
```

### Docker Deployment

```bash
# Build from atlas root (REQUIRED)
cd C:/projects/atlas
build.bat --akab

# Run with proper volume mounts
docker run -it --rm \
  -v ./krill:/krill \
  --env-file ./akab/.env \
  akab-mcp:latest
```

### Local Development

```bash
cd C:/projects/atlas/akab
pip install -e ../substrate  # Install substrate first
pip install -e .

# For Google/Gemini support
pip install google-generativeai

python -m akab              # Run MCP server
```

## Usage Examples

### Level 1: Quick Compare

```python
# Basic comparison without constraints
result = await akab_quick_compare(
    ctx,
    prompt="Explain quantum computing to a child",
    providers=["anthropic_m", "openai_l"]
)

# With specific constraints
result = await akab_quick_compare(
    ctx,
    prompt="Write a haiku about programming",
    providers=["anthropic_s", "openai_s"],
    constraints={
        "max_tokens": 50,
        "temperature": 0.7,
        "must_include": ["code", "debug"]
    }
)
```

### Level 2: Campaign

```python
# Create campaign with success criteria
campaign = await akab_create_campaign(
    ctx,
    name="Creative Writing Test",
    description="Compare creative capabilities",
    variants=[
        {
            "provider": "anthropic",
            "size": "xl",
            "temperature": 0.9,
            "prompt": "Write a story about time travel"
        },
        {
            "provider": "openai",
            "size": "xl", 
            "temperature": 0.9,
            "prompt": "Write a story about time travel"
        }
    ],
    success_criteria={
        "primary": {
            "metric": "quality_score",
            "weight": 0.8
        },
        "constraints": {
            "min_length": 200,
            "max_length": 500
        }
    }
)

# Execute with multiple iterations
await akab_execute_campaign(ctx, campaign.id, iterations=10)

# Analyze results
analysis = await akab_analyze_results(ctx, campaign.id)

# Unlock to see provider mappings
unlocked = await akab_unlock(ctx, campaign.id)
```

### Level 3: Scientific Experiment

```python
# List available scrambled models
models = await akab_list_scrambled_models(ctx)
# Returns: ["model_7a9f2e", "model_3b8d1c", ...]

# Create formal experiment
experiment = await akab_create_experiment(
    ctx,
    name="Reasoning Capability Study",
    description="Evaluate logical reasoning across models",
    hypothesis="Larger models show better multi-step reasoning",
    variants=["model_7a9f2e", "model_3b8d1c", "model_9e5a1f"],
    prompts=[
        "Solve: If all roses are flowers and some flowers fade...",
        "Explain the logical flaw in this argument...",
        # More prompts for statistical power
    ],
    iterations_per_prompt=20,
    success_criteria={
        "primary": {"metric": "reasoning_score"}
    }
)

# Results available only after statistical significance
result = await akab_reveal_experiment(ctx, experiment.id)

# If not significant, diagnose why
diagnosis = await akab_diagnose_experiment(ctx, experiment.id)

# Archive after completion
archived = await akab_unlock(ctx, experiment.id)
```

## Tools Reference

### Core Tools

- `akab` - Get capabilities and documentation
- `akab_sampling_callback` - Handle sampling responses from Claude

### Level 1 Tools

- `akab_quick_compare` - Quick comparison with no blinding

### Level 2 Tools  

- `akab_create_campaign` - Create A/B testing campaign
- `akab_execute_campaign` - Execute campaign with iterations
- `akab_analyze_results` - Statistical analysis of results
- `akab_list_campaigns` - List campaigns by status
- `akab_cost_report` - Cost tracking and analysis

### Level 3 Tools

- `akab_list_scrambled_models` - List available scrambled model IDs
- `akab_create_experiment` - Create scientific experiment
- `akab_reveal_experiment` - Check for statistical significance
- `akab_diagnose_experiment` - Diagnose convergence issues

### Archival Tools

- `akab_unlock` - Unlock and archive completed campaigns/experiments

## Storage Architecture

All data stored in `/krill/` (outside LLM access for security):

```text
/krill/
├── scrambling/          # Fire-and-forget model mappings
│   └── session.json     # Current session scrambling
├── campaigns/           
│   ├── quick/          # Level 1 results
│   ├── standard/       # Level 2 campaigns  
│   └── experiments/    # Level 3 experiments
├── results/            # Raw execution data
│   └── <campaign_id>/  # Individual test results
└── archive/            # Unlocked campaigns
    └── <id>/
        ├── blinded/    # Original blinded state
        ├── clear/      # Revealed mappings
        └── metadata.json
```

## Advanced MCP Patterns

This implementation demonstrates advanced MCP patterns:

1. **Simulated Sampling**: Intelligent assistance via `_sampling_request`
2. **Progress Tracking**: Real-time updates via `_progress`
3. **Response Annotations**: Priority, tone, and visualization hints
4. **Structured Errors**: Actionable recovery suggestions
5. **Context-Aware Decisions**: Help only when beneficial

## Provider Support Status

### Fully Supported

- **Anthropic (Claude)**: All models, full pricing data
- **OpenAI (GPT)**: All models, full pricing data

### Experimental

- **Google (Gemini)**: Code implemented, needs activation
  - BlindedHermes includes full Google provider support
  - Server configuration needs updating to enable
  - Install `google-generativeai` package to use

### Adding Google/Gemini Support

To fully enable Google/Gemini:

1. Install the package: `pip install google-generativeai`
2. Set `GOOGLE_API_KEY` in your `.env` file
3. Update `server.py` to include "google" in `valid_providers`
4. Add Google model mappings to `model_sizes` dictionary

## Important Notes

### Production-Grade Means

- **REAL API CALLS**: No mocks, stubs, or fake responses
- **ACTUAL RESULTS**: Real LLM outputs with content
- **WORKING METRICS**: Real token counts, costs, timings
- **LOUD FAILURES**: No silent errors or empty successes

### Common Issues

- **Import Errors**: Ensure Anthropic and OpenAI packages installed
- **API Keys**: Must be valid and have sufficient credits
- **Docker Context**: Always build from atlas root directory
- **Stdout Purity**: Never print to stdout (breaks MCP protocol)
- **Google Support**: Currently experimental, requires manual activation

### Integration Pattern

The "improve" functionality uses AKAB Level 1:

1. Generate variant based on improvement direction
2. Use `akab_quick_compare` for immediate comparison
3. Return results for human evaluation
4. User decides whether to accept improvement

This maintains separation of concerns - AKAB handles testing, other MCPs handle their specific domains.
