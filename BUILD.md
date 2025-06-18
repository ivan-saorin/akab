# Building and Running AKAB with Substrate

## Prerequisites
- Docker installed
- Substrate base image built

## Quick Start

### 1. Build Substrate (one-time)
```bash
cd ../substrate
docker build -t substrate:latest .
```

### 2. Build AKAB
```bash
docker build -t akab:latest .
```

### 3. Run AKAB
```bash
docker-compose up
```

Or manually:
```bash
docker run -p 8000:8000 -v $(pwd)/data:/data/akab akab:latest
```

## Architecture

AKAB now uses Substrate as its foundation layer:

```
substrate:latest (Base Image)
├── MCP Server (FastMCP)
├── Multi-Provider Support
├── Evaluation Engine
└── Filesystem Utilities

akab:latest (FROM substrate:latest)
├── AKAB-specific filesystem
├── Campaign management
├── Experiment tools
└── MCP tool implementations
```

## Development

When developing, remember:
- Shared utilities are in `substrate/`
- AKAB-specific code is in `akab/src/akab/`
- Imports from substrate are available directly (no path needed)

## API Keys

For remote providers, set environment variables:
```bash
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GOOGLE_API_KEY=your-key
```

Or create a `.env` file (see `.env.example`).
