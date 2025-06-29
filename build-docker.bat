@echo off
REM Build AKAB Docker image

echo Building AKAB Docker image...
echo.

REM Build with the main Dockerfile
docker build -t akab-mcp:latest .

if errorlevel 1 (
    echo.
    echo Build failed!
    exit /b 1
)

echo.
echo Testing image...
docker run --rm akab-mcp:latest python -c "import akab; print(f'AKAB {akab.__version__} ready!')"

if errorlevel 1 (
    echo.
    echo Import test failed!
    exit /b 1
)

echo.
echo Build successful!
echo.
echo To run AKAB:
echo   docker run --rm -i ^
echo     -e ANTHROPIC_API_KEY=your_key ^
echo     -e OPENAI_API_KEY=your_key ^
echo     -e GOOGLE_API_KEY=your_key ^
echo     akab-mcp:latest
