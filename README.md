# AKAB - Scientific A/B Testing for AI

<p align="center">
  <strong>🧪 Open-source A/B testing tool for comparing AI outputs across providers</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#providers">Providers</a> •
  <a href="#api">API</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MCP-Compatible-blue" alt="MCP Compatible">
  <img src="https://img.shields.io/badge/Docker-Ready-brightgreen" alt="Docker Ready">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License MIT">
</p>

## What is AKAB?

AKAB (A/B testing Kinda Advanced, Bro) is a scientific tool for comparing AI model outputs across multiple providers. It helps you make data-driven decisions about which AI models to use based on performance, cost, and quality metrics.

### Why AKAB?

- **🔬 Scientific Comparison**: Compare outputs across providers with consistent methodology
- **💰 Cost Optimization**: Track and compare costs to optimize your AI spending
- **📊 Comprehensive Metrics**: Analyze speed, quality, token usage, and more
- **🎯 Provider Agnostic**: Works with Anthropic, OpenAI, Google, and more
- **🚀 Production Ready**: Built on [Substrate MCP Foundation](https://github.com/yourusername/substrate)

## Features

- **Quick Compare**: One-shot comparisons across multiple providers
- **Campaign Management**: Create comprehensive test campaigns with multiple prompts
- **Cost Tracking**: Monitor spending with detailed breakdowns
- **Provider Abstraction**: Size-based naming (anthropic_m, openai_l, etc.)
- **Smart Constraints**: Get constraint suggestions via sampling
- **Scientific Analysis**: Statistical comparison with winner selection
- **Progress Tracking**: Long operations with timeout prevention
- **Template Support**: Use `{{variables}}` in prompts

## Quick Start

### 1. Install with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/akab.git
cd akab

# Build Docker image
docker build -t akab-mcp:latest .

# Run with your API keys
docker run --rm -i \
  -e ANTHROPIC_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  akab-mcp:latest
```

### 2. Configure Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "akab": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-e", "ANTHROPIC_API_KEY=your_key",
        "-e", "OPENAI_API_KEY=your_key",
        "-e", "GOOGLE_API_KEY=your_key",
        "akab-mcp:latest"
      ]
    }
  }
}
```

### 3. Start Testing!

```
Use akab to compare "Explain quantum computing in simple terms" across anthropic_m and openai_m
```

## Installation

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t akab-mcp:latest .

# Or use docker-compose
docker-compose up akab
```

### Option 2: Local Installation

```bash
# Clone both repositories
git clone https://github.com/yourusername/substrate.git
git clone https://github.com/yourusername/akab.git

# Install substrate first
cd substrate
pip install -e .

# Install AKAB
cd ../akab
pip install -e .

# Set environment variables
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key

# Run the server
python -m akab
```

## Usage

### Quick Compare

Compare a single prompt across providers:

```python
# Simple comparison
"Use akab to compare 'Write a haiku about AI' across anthropic_s and openai_s"

# With constraints
"Use akab to compare with max_tokens=50 and temperature=0"

# With template parameters
"Compare 'Write a {{style}} about {{topic}}' with style='sonnet' and topic='machine learning'"
```

### Campaign Management

Run comprehensive tests with multiple prompts:

```python
# Create a campaign
"Create an AKAB campaign called 'content_quality' to test my writing prompts"

# Execute campaign
"Execute the content_quality campaign"

# Analyze results
"Analyze results from the content_quality campaign"
```

### Cost Tracking

Monitor your AI spending:

```python
# Get cost report
"Show me AKAB cost report for this week grouped by provider"

# Campaign cost estimate
"Estimate cost for campaign_xyz before running it"
```

## Providers

AKAB uses size-based provider naming for future-proof configuration:

| Size | Anthropic | OpenAI | Google |
|------|-----------|---------|---------|
| XS | `anthropic_xs` (Haiku) | - | - |
| S | `anthropic_s` (Haiku) | `openai_s` (GPT-3.5) | `google_s` (Gemini Flash) |
| M | `anthropic_m` (Sonnet) | `openai_m` (GPT-4) | `google_m` (Gemini Pro) |
| L | `anthropic_l` (Sonnet) | `openai_l` (GPT-4 Turbo) | - |
| XL | `anthropic_xl` (Opus) | - | - |

Set API keys as environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`

## API Reference

### Tools

#### `akab`
Get server information and capabilities.

#### `akab_quick_compare`
Compare a prompt across providers.

**Parameters:**
- `prompt` (str): The prompt to test
- `providers` (list): List of providers (e.g., ["anthropic_m", "openai_m"])
- `parameters` (dict, optional): Template parameters
- `constraints` (dict, optional): Constraints like max_tokens, temperature

**Example Response:**
```json
{
  "success": true,
  "data": {
    "results": [...],
    "winner": "anthropic_m",
    "metrics": {
      "fastest_provider": "openai_s",
      "cheapest_provider": "anthropic_s",
      "average_latency_ms": 1523.45
    }
  }
}
```

#### `akab_create_campaign`
Create a new testing campaign.

**Parameters:**
- `name` (str): Campaign name
- `prompts` (list): List of prompt configurations
- `providers` (list): Providers to test
- `iterations` (int): Number of test iterations
- `constraints` (dict, optional): Global constraints

#### `akab_execute_campaign`
Execute a campaign with optional dry run.

**Parameters:**
- `campaign_id` (str): Campaign ID
- `dry_run` (bool): If true, only estimate cost

#### `akab_analyze_results`
Analyze campaign results with insights.

**Parameters:**
- `campaign_id` (str): Campaign ID
- `metrics` (list, optional): Specific metrics to analyze

#### `akab_list_campaigns`
List all campaigns with filtering.

**Parameters:**
- `status` (str, optional): Filter by status
- `limit` (int): Results per page
- `offset` (int): Pagination offset

#### `akab_cost_report`
Get detailed cost analysis.

**Parameters:**
- `time_period` (str): "day", "week", "month", "all"
- `group_by` (str): "provider", "campaign", "prompt"

## Example Results

### Quick Compare Output
```json
{
  "results": [
    {
      "provider": "anthropic_m",
      "response": "Quantum bits dance,\nSuperposition's strange waltz,\nReality blurs.",
      "latency_ms": 1245.67,
      "tokens_used": 42,
      "cost_estimate": 0.00021
    },
    {
      "provider": "openai_m",
      "response": "Qubits spinning round\nBoth here and there at once, strange\nQuantum mysteries",
      "latency_ms": 987.23,
      "tokens_used": 38,
      "cost_estimate": 0.00019
    }
  ],
  "winner": "openai_m",
  "metrics": {
    "fastest_provider": "openai_m",
    "cheapest_provider": "openai_m",
    "average_latency_ms": 1116.45
  }
}
```

### Campaign Analysis
```json
{
  "campaign_id": "campaign_abc123",
  "total_completions": 24,
  "total_cost": 0.0453,
  "provider_performance": {
    "anthropic_m": {
      "avg_latency": 1523.45,
      "avg_cost": 0.00234,
      "success_rate": 1.0
    },
    "openai_m": {
      "avg_latency": 1102.33,
      "avg_cost": 0.00187,
      "success_rate": 0.98
    }
  },
  "insights": [
    "openai_m was 38% faster than anthropic_m",
    "anthropic_m responses were 25% longer on average",
    "Cost difference: openai_m saves $0.47 per 1000 requests"
  ]
}
```

## Architecture

Built on [Substrate MCP Foundation](https://github.com/yourusername/substrate):

```
AKAB Server
├── Substrate Base (SubstrateMCP)
│   ├── Standard Tools
│   ├── Progress Tracking
│   └── Error Handling
├── Provider Manager
│   ├── Provider Configuration
│   └── Size-based Mapping
├── Comparison Engine
│   ├── Parallel Execution
│   └── Result Analysis
├── Campaign Manager
│   ├── Campaign Storage
│   └── Batch Processing
└── Cost Tracker
    ├── Usage Monitoring
    └── Spend Analysis
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/akab.git
cd akab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=akab

# Run in Docker
docker-compose run --rm akab pytest
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## Docker

### Building

```bash
# Production image
docker build -t akab-mcp:latest .

# Development image with live reload
docker build -f Dockerfile.dev -t akab-mcp:dev .
```

### Docker Compose

```yaml
services:
  akab:
    image: akab-mcp:latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - akab_data:/app/akab_data
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for Contribution

- Additional provider support (Cohere, AI21, etc.)
- Advanced analysis metrics
- Web UI for results visualization
- Export formats (CSV, JSON, charts)
- Automated testing workflows

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [Substrate MCP Foundation](https://github.com/yourusername/substrate)
- Designed for the [Model Context Protocol](https://modelcontextprotocol.io)
- Inspired by the need for scientific AI comparison

## Links

- [Substrate Foundation](https://github.com/yourusername/substrate)
- [MCP Documentation](https://modelcontextprotocol.io)
- [Issue Tracker](https://github.com/yourusername/akab/issues)
- [Discussions](https://github.com/yourusername/akab/discussions)

---

<p align="center">Made with ❤️ for the AI community</p>
