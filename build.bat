@echo off
REM Build akab Docker image

echo Building akab MCP Docker image...

REM Save current directory
set ORIGINAL_DIR=%CD%

REM Change to atlas directory (parent of akab)
cd /d "%~dp0\.."

echo Building from directory: %CD%
echo Using Dockerfile: %~dp0Dockerfile

REM Build the image using akab's Dockerfile
docker build -t akab-mcp:latest -f "%~dp0Dockerfile" .

set BUILD_RESULT=%ERRORLEVEL%

REM Return to original directory
cd /d "%ORIGINAL_DIR%"

if %BUILD_RESULT% EQU 0 (
    echo.
    echo Build successful!
    echo Docker image: akab-mcp:latest
    echo.
) else (
    echo.
    echo Build failed!
    exit /b 1
)
