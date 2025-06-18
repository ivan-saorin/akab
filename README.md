# AKAB - Adaptive Knowledge Acquisition Benchmark

<p align="center">
  <img src="docs/images/akab-logo.png" alt="AKAB Logo" width="400"/>
</p>

<p align="center">
  <strong>The Definitive Platform for Systematic AI Research and AB Testing</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#providers">Providers</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

## 🚀 Overview

AKAB (Adaptive Knowledge Acquisition Benchmark) is a revolutionary MCP (Model Context Protocol) server that enables systematic, scientific AB testing across multiple AI models and providers. Built for researchers, developers, and AI enthusiasts who need rigorous experimentation capabilities with complete control over their data.

### Why AKAB?

- **🔬 Scientific Rigor**: Structured experiments with reproducible results
- **💰 Cost Transparency**: Know exactly what you're spending before you start
- **🌐 Multi-Provider**: Compare Claude, GPT-4, Gemini, and more in one platform
- **📊 Rich Analytics**: Deep insights from systematic experimentation
- **🔒 Your Data**: Filesystem-based storage you control completely
- **🚄 Blazing Fast**: Local or remote execution with minimal overhead

## ✨ Features

### Core Capabilities

- **Multi-Model AB Testing**: Compare responses across providers scientifically
- **Campaign Management**: Organize experiments into logical campaigns
- **Meta-Prompt System**: Self-configuring workflows for automated experimentation
- **Dual-Mode Execution**: Run locally via MCP or remotely via APIs
- **Cost Tracking**: Real-time cost estimation and tracking
- **Knowledge Base Integration**: Inject context to guide AI behavior
- **Structured Outputs**: Consistent results via Instructor library
- **Docker-Native**: Easy deployment anywhere

### Supported Providers

| Provider | Models | Status | Cost/1K tokens |
|----------|--------|--------|----------------|
| Anthropic (Local) | Claude via MCP | ✅ Ready | Free* |
| OpenAI | GPT-3.5/4 | ✅ Ready | $0.002-$0.03 |
| Anthropic API | Claude 3 Opus/Sonnet | ✅ Ready | $0.003-$0.025 |
| Google | Gemini Pro | 🚧 Coming | $0.001 |
| Mistral | Open models | 🚧 Coming | $0.0002 |

*Via Claude Desktop or compatible MCP client

## 🏃 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for development)
- Claude Desktop (for local MCP execution)
- API keys for remote providers

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/akab.git
cd akab
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start AKAB Server

```bash
docker compose up -d
```

### 3. Configure Claude Desktop

Generic MCP client configuration.

```json
{
  "mcpServers": {
    "akab": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "--name", "akab-mcp", 
               "-v", "${HOME}/akab-data:/data/akab", 
               "-p", "8000:8000", "akab:latest"]
    }
  }
}
```

Add to your Claude Desktop configuration:

```json
	"akab": {
      "command": "npx",
      "args": [
        "-y",
        "supergateway",
        "--streamableHttp",
        "http://localhost:8001/mcp"
      ]
    }
  }
```

### 4. Run Your First Experiment

In Claude Desktop:

```
Use the akab_get_meta_prompt tool to load instructions, then run experiments!
```

## 🏗️ Architecture

AKAB follows a clean, modular architecture designed for extensibility and reliability:

```
┌─────────────────┐     ┌──────────────────┐
│  Claude Desktop │     │   API Providers  │
│   (MCP Client)  │     │ (OpenAI, etc.)   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────┐   ┌───────────┘
                 │   │
         ┌───────▼───▼────────┐
         │   AKAB MCP Server  │
         │  ┌──────────────┐  │
         │  │ FastMCP Core │  │
         │  └──────┬───────┘  │
         │         │          │
         │  ┌──────▼───────┐  │     ┌─────────────────┐
         │  │Tool Registry │  ├─────►│Provider Manager │
         │  └──────┬───────┘  │     └─────────────────┘
         │         │          │
         │  ┌──────▼───────┐  │     ┌─────────────────┐
         │  │  Filesystem  │  ├─────►│ Cost Tracker    │
         │  │   Storage    │  │     └─────────────────┘
         │  └──────────────┘  │
         └────────────────────┘
                 │
         ┌───────▼────────┐
         │  Docker Volume │
         │  /data/akab/   │
         └────────────────┘
```

### Key Components

- **FastMCP Core**: Official Anthropic MCP implementation
- **Tool Registry**: Workflow-oriented tools for experimentation
- **Provider Manager**: Abstraction layer for multi-model support
- **Filesystem Storage**: Transparent, version-controllable data
- **Cost Tracker**: Real-time cost monitoring and warnings

## 📁 Data Organization

AKAB uses a campaign-based folder structure for clarity:

```
/data/akab/
├── campaigns/              # Campaign definitions
├── experiments/            # Organized by campaign
│   ├── my-campaign/
│   │   ├── exp_001/       # Individual experiments
│   │   │   ├── config.json
│   │   │   ├── prompt.md
│   │   │   └── result.json
│   │   └── exp_002/
│   └── another-campaign/
├── knowledge_bases/        # Reusable context
├── templates/             # Prompt templates
└── results/               # Analysis outputs
```

## 🛠️ Available Tools

AKAB provides a comprehensive toolkit via MCP:

### Meta Configuration

- `akab_get_meta_prompt()` - Load self-configuring instructions

### Experiment Management

- `akab_get_next_experiment()` - Queue management
- `akab_get_exp_prompt(exp_id)` - Retrieve prompts
- `akab_save_exp_result(exp_id, result, metadata)` - Store results

### Campaign Operations

- `akab_create_campaign(config)` - Initialize campaigns
- `akab_list_campaigns()` - View all campaigns
- `akab_switch_campaign(campaign_id)` - Change active campaign
- `akab_get_campaign_status()` - Check progress
- `akab_analyze_results(campaign_id)` - Generate insights

### Remote Execution

- `akab_batch_execute_remote(campaign_id)` - Launch batch runs
- `akab_get_execution_status()` - Monitor progress

## 💰 Cost Management

AKAB includes sophisticated cost tracking:

- **Pre-execution estimates** with warnings for 20+ experiments
- **Real-time tracking** during batch execution
- **Provider comparison** metrics
- **Cost/quality analysis** in results

Example cost warning:

```
⚠️ About to execute 50 experiments across 3 providers
Estimated cost: $12.50
- OpenAI GPT-4: $8.00
- Claude 3 Opus: $3.50
- Gemini Pro: $1.00
```

## 📚 Documentation

- [USER-JOURNEYS.md](./USER-JOURNEYS.md) - Detailed usage scenarios
- [Architecture Guide](./docs/ARCHITECTURE.md) - Technical deep dive
- [Provider Integration](./docs/PROVIDERS.md) - Adding new providers
- [API Reference](./docs/API.md) - Complete tool documentation
- [Cost Optimization](./docs/COST-OPTIMIZATION.md) - Save money on experiments

## 🔬 Example Use Cases

### 1. Compare Model Creativity

Test how different models approach creative writing tasks:

```python
{
    "name": "creativity-benchmark",
    "providers": ["openai/gpt-4", "anthropic/claude-3-opus", "google/gemini-pro"],
    "experiments": 30,
    "prompt_template": "creative_writing",
    "analysis_focus": ["novelty", "coherence", "engagement"]
}
```

### 2. Technical Accuracy Testing

Evaluate code generation capabilities:

```python
{
    "name": "code-generation-test",
    "providers": ["openai/gpt-4", "anthropic/claude-3-sonnet"],
    "experiments": 50,
    "prompt_template": "python_algorithms",
    "validation": "automated_testing"
}
```

### 3. Cost/Performance Optimization

Find the best model for your budget:

```python
{
    "name": "budget-optimization",
    "providers": ["openai/gpt-3.5-turbo", "anthropic/claude-instant", "mistral/mixtral"],
    "experiments": 100,
    "constraint": "max_cost=$10",
    "optimize_for": "quality_per_dollar"
}
```

## 🚀 Advanced Features

### Parallel Execution

Run experiments across providers simultaneously:

```python
config["execution_mode"] = "parallel"
config["max_concurrent"] = 5
```

### Custom Evaluation Metrics

Define your own scoring functions:

```python
config["evaluators"] = ["innovation_score", "practical_value", "custom_metric"]
```

### Knowledge Base Injection

Guide model behavior with context:

```python
config["knowledge_bases"] = ["domain_expertise.md", "style_guide.md"]
```

## 🔧 Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start development server
python -m akab.server
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Full test suite with coverage
pytest --cov=akab --cov-report=html
```

## 🚢 Production Deployment

### Using CapRover

1. Install CapRover on your server
2. Create new app named `akab`
3. Deploy using Captain CLI:

```bash
captain deploy
```

### Manual Docker Deployment

```bash
docker build -t akab:latest .
docker run -d \
  --name akab-prod \
  -v /path/to/data:/data/akab \
  -p 8000:8000 \
  --env-file .env.prod \
  akab:latest
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Tool call overhead | <50ms | FastMCP optimized |
| Experiment execution (local) | <100ms | Via MCP client |
| Experiment execution (remote) | 1-5s | Depends on provider |
| Campaign analysis (100 exp) | <2s | Filesystem-based |
| Batch execution (50 exp) | 2-5min | With rate limiting |

## 🔒 Security

- API keys stored in environment variables only
- No credentials in filesystem storage
- Docker container isolation
- Optional API key rotation
- Rate limiting for all providers

## 📝 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for the MCP protocol and Claude
- OpenAI, Google, and other providers for their APIs
- The Instructor library team for structured outputs
- Ivan Saorin for the vision and architecture

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/akab&type=Date)](https://star-history.com/#yourusername/akab&Date)

---

<p align="center">
  Built with ❤️ for the AI research community
</p>

<p align="center">
  <a href="https://github.com/yourusername/akab/issues">Report Bug</a> •
  <a href="https://github.com/yourusername/akab/issues">Request Feature</a> •
  <a href="https://discord.gg/akab">Join Discord</a>
</p>
