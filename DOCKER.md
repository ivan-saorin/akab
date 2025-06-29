# Docker Setup for AKAB

## Quick Start

### 1. Build the Docker Image

```bash
# Windows
.\build-docker.bat

# Linux/Mac
./build-docker.sh
```

### 2. Run with Docker

```bash
# Interactive mode
docker run --rm -it \
  -e ANTHROPIC_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  akab-mcp:latest

# With persistent data
docker run --rm -it \
  -v akab_data:/app/akab_data \
  -e ANTHROPIC_API_KEY=your_key \
  akab-mcp:latest
```

### 3. Run with Docker Compose

```bash
# Start the service
docker-compose up -d akab

# View logs
docker-compose logs -f akab

# Stop the service
docker-compose down
```

## Claude Desktop Configuration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "akab": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--name", "akab-mcp-claude",
        "--log-driver", "json-file",
        "--log-opt", "max-size=1m",
        "-e", "ANTHROPIC_API_KEY=your_key_here",
        "-e", "OPENAI_API_KEY=your_key_here",
        "-e", "GOOGLE_API_KEY=your_key_here",
        "-e", "MCP_TRANSPORT=stdio",
        "akab-mcp:latest"
      ]
    }
  }
}
```

## Files

- `Dockerfile` - Main Docker image
- `Dockerfile.production` - Optimized production build
- `docker-compose.yml` - Docker Compose configuration
- `build-docker.sh/bat` - Build scripts
- `test-docker.sh/bat` - Test scripts
- `.dockerignore` - Build exclusions

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for Anthropic providers
- `OPENAI_API_KEY` - Required for OpenAI providers
- `GOOGLE_API_KEY` - Required for Google providers
- `MCP_TRANSPORT` - Always "stdio" for MCP
- `LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR)
- `AKAB_DATA_PATH` - Data storage path (default: /app/akab_data)

## Data Persistence

Campaign data is stored in Docker volumes. To backup:

```bash
docker run --rm -v akab_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/akab-backup.tar.gz -C /data .
```
