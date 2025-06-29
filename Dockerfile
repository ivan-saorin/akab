# AKAB MCP Server Docker Image - Fixed Package Structure
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 akab

WORKDIR /app

# Install dependencies first (better layer caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastmcp pydantic httpx aiofiles python-dotenv \
        anthropic openai google-generativeai

# Create proper package structure
# The key is that we need the files in a directory named 'akab'
# so that 'import akab' works
RUN mkdir -p /app/akab

# Copy all Python source files to the akab package directory
COPY src/__init__.py /app/akab/__init__.py
COPY src/__main__.py /app/akab/__main__.py
COPY src/server.py /app/akab/server.py
COPY src/providers.py /app/akab/providers.py
COPY src/comparison.py /app/akab/comparison.py
COPY src/campaigns.py /app/akab/campaigns.py
COPY src/storage.py /app/akab/storage.py
COPY src/constraints.py /app/akab/constraints.py
COPY src/substrate_stub.py /app/akab/substrate_stub.py

# Create substrate.py from substrate_stub.py
RUN cp /app/akab/substrate_stub.py /app/akab/substrate.py

# Ensure Python can find our package
ENV PYTHONPATH=/app:$PYTHONPATH

# Create data directory
RUN mkdir -p /app/akab_data && chown -R akab:akab /app

# Test that the import works before switching users
RUN python -c "import akab; from akab import AKABServer; print('Import test passed!')"

# Switch to non-root user
USER akab

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=stdio \
    AKAB_DATA_PATH=/app/akab_data \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import akab; print('OK')"

# Run AKAB server
CMD ["python", "-m", "akab"]
