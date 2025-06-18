#!/bin/bash
# AKAB Quick Setup Script

echo "🚀 AKAB - Adaptive Knowledge Acquisition Benchmark"
echo "================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  Please edit .env and add your API keys for remote providers"
    echo "   - OPENAI_API_KEY"
    echo "   - ANTHROPIC_API_KEY"
    echo "   - GOOGLE_API_KEY"
    echo ""
    read -p "Press Enter to continue after adding API keys (or skip for local-only mode)..."
    echo ""
fi

# Build and start containers
echo "🔨 Building AKAB Docker image..."
docker compose build

echo ""
echo "🚀 Starting AKAB server..."
docker compose up -d

# Wait for server to be ready
echo ""
echo "⏳ Waiting for server to be ready..."
sleep 5

# Check if server is running
if docker compose ps | grep -q "akab-mcp-server.*running"; then
    echo "✅ AKAB server is running!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Configure Claude Desktop with:"
    echo '   {
     "mcpServers": {
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
   }'
    echo ""
    echo "2. In Claude Desktop, run:"
    echo "   'Use akab_get_meta_prompt to load instructions'"
    echo ""
    echo "3. Start experimenting!"
    echo ""
    echo "📊 View logs: docker compose logs -f"
    echo "🛑 Stop server: docker compose down"
    echo ""
    echo "🎉 Happy experimenting with AKAB!"
else
    echo "❌ Failed to start AKAB server"
    echo "Check logs with: docker compose logs"
    exit 1
fi
