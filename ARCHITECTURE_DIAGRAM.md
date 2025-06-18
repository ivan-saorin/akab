# AKAB v2.0 Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User/Claude Desktop                    │
│                              ↓                                │
│                         MCP Protocol                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                      AKAB Server (Port 8000)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  AKAB Application Layer              │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │    │
│  │  │  Campaign   │  │   AKAB       │  │   Meta     │ │    │
│  │  │ Management  │  │   Tools      │  │  Prompts   │ │    │
│  │  └─────────────┘  └──────────────┘  └────────────┘ │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │    │
│  │  │ Experiment  │  │  Templates   │  │ Knowledge  │ │    │
│  │  │ Tracking    │  │  & Variables │  │   Bases    │ │    │
│  │  └─────────────┘  └──────────────┘  └────────────┘ │    │
│  │  ┌────────────────────────────────────────────────┐ │    │
│  │  │        AKABFileSystemManager (extends)         │ │    │
│  │  └────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                            │
                     FROM substrate:latest
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    Substrate Foundation Layer                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Core Infrastructure                 │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │    │
│  │  │ MCP Server  │  │  Provider    │  │ Evaluation │ │    │
│  │  │ (FastMCP)   │  │  Manager     │  │   Engine   │ │    │
│  │  └─────────────┘  └──────────────┘  └────────────┘ │    │
│  │  ┌─────────────────────────────────────────────────┐ │    │
│  │  │              FileSystemManager (base)           │ │    │
│  │  └─────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    LLM Providers                     │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────┐ │    │
│  │  │   Local   │ │  OpenAI   │ │ Anthropic │ │ ... │ │    │
│  │  │ (Claude)  │ │ (GPT-4)   │ │   (API)   │ │     │ │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └─────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Component Dependencies

```
User Request
    ↓
MCP Tool Call (e.g., akab_get_next_experiment)
    ↓
AKAB Tools (AKABTools class)
    ├── Uses: AKABFileSystemManager (AKAB layer)
    ├── Uses: ProviderManager (Substrate layer)
    └── Uses: EvaluationEngine (Substrate layer)
         ↓
File Operations
    ├── Campaign/Experiment specific → AKABFileSystemManager
    └── Generic file operations → FileSystemManager (base)
         ↓
Provider Execution
    ├── Local → Returns prompt for user
    └── Remote → API call via Provider classes
         ↓
Response Evaluation
    └── EvaluationEngine scores response
         ↓
Results Storage
    └── Back through filesystem layers
```

## Docker Layer Architecture

```
┌─────────────────────────────┐
│      akab:latest            │ Layer 3: Application
│   (AKAB-specific code)      │ Size: ~50MB
├─────────────────────────────┤
│    substrate:latest         │ Layer 2: Foundation
│  (Shared infrastructure)    │ Size: ~200MB
├─────────────────────────────┤
│   python:3.11-slim          │ Layer 1: Base
│     (Python runtime)        │ Size: ~150MB
└─────────────────────────────┘
                              Total: ~400MB

Benefits:
- Substrate layer shared across projects
- Rebuilding AKAB only updates Layer 3
- Docker caches Layers 1 & 2
```

## Import Structure

```python
# ===== From Substrate (Foundation) =====
from mcp.server import FastMCP              # MCP protocol
from providers import (                     # LLM providers
    ProviderManager, 
    Provider, 
    ProviderType
)
from evaluation import EvaluationEngine     # Response scoring
from filesystem import FileSystemManager    # Base file operations

# ===== From AKAB (Application) =====
from akab.filesystem import AKABFileSystemManager  # Extended filesystem
from akab.tools.akab_tools import AKABTools        # Tool implementations
from akab.server import app, mcp                   # Server setup
```

## Data Flow Example: Running an Experiment

```
1. User: "Run next experiment"
       ↓
2. Claude Desktop → MCP → akab_get_next_experiment()
       ↓
3. AKABTools.get_next_experiment()
       ├── Load campaign (AKABFileSystemManager)
       ├── Check progress
       └── Return experiment info
       ↓
4. User: "Get prompt"
       ↓
5. akab_get_exp_prompt(exp_001)
       ├── Load template (AKABFileSystemManager)
       ├── Apply variables
       └── Return prompt
       ↓
6. Execution
       ├── Local: User runs prompt
       └── Remote: ProviderManager.execute()
       ↓
7. akab_save_exp_result(exp_001, response)
       ├── Evaluate (EvaluationEngine)
       ├── Save result (AKABFileSystemManager)
       └── Update campaign progress
```

## Future Projects on Substrate

```
substrate:latest
    │
    ├── akab:latest (Experiment Management)
    │   └── Campaign orchestration, templates, analysis
    │
    ├── thrice:latest (Phenomenon Engine)
    │   └── Guided hallucination, innovation scoring
    │
    ├── synapse:latest (AI Interaction)
    │   └── Code execution, syntax processing
    │
    └── your-ai-tool:latest
        └── Your domain-specific AI logic
```

This architecture enables rapid development of AI tools while maintaining consistency and quality across the ecosystem.
