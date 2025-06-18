@echo off
REM AKAB Quick Setup Script for Windows

echo.
echo 🚀 AKAB - Adaptive Knowledge Acquisition Benchmark
echo ==================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo Visit: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    docker-compose --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Docker Compose is not installed. Please install Docker Desktop.
        echo Visit: https://docs.docker.com/desktop/install/windows-install/
        pause
        exit /b 1
    )
)

echo ✅ Docker and Docker Compose are installed
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ✅ .env file created
    echo.
    echo ⚠️  Please edit .env and add your API keys for remote providers
    echo    - OPENAI_API_KEY
    echo    - ANTHROPIC_API_KEY
    echo    - GOOGLE_API_KEY
    echo.
    echo Press any key to continue after adding API keys (or skip for local-only mode)...
    pause >nul
    echo.
)

REM Build and start containers
echo 🔨 Building AKAB Docker image...
docker compose build

echo.
echo 🚀 Starting AKAB server...
docker compose up -d

REM Wait for server to be ready
echo.
echo ⏳ Waiting for server to be ready...
timeout /t 5 /nobreak >nul

REM Check if server is running
docker compose ps | findstr "akab-mcp-server.*running" >nul
if %errorlevel% equ 0 (
    echo ✅ AKAB server is running!
    echo.
    echo 📋 Next steps:
    echo 1. Configure Claude Desktop with:
    echo    "mcpServers": {
    echo      "akab": {
    echo        "command": "npx",
    echo        "args": [
    echo          "-y",
    echo          "supergateway",
    echo          "--streamableHttp",
    echo          "http://localhost:8001/mcp"
    echo        ]
    echo      }
    echo    }
    echo.
    echo 2. In Claude Desktop, run:
    echo    'Use akab_get_meta_prompt to load instructions'
    echo.
    echo 3. Start experimenting!
    echo.
    echo 📊 View logs: docker compose logs -f
    echo 🛑 Stop server: docker compose down
    echo.
    echo 🎉 Happy experimenting with AKAB!
) else (
    echo ❌ Failed to start AKAB server
    echo Check logs with: docker compose logs
    exit /b 1
)

pause
