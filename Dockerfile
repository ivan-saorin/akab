FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy substrate first (dependency) - INCLUDING docs
COPY substrate /substrate

# Copy akab
COPY akab /app

# Install dependencies
# First install substrate
RUN cd /substrate && pip install --no-cache-dir -e .

# Then install akab (substrate will already be available)
RUN pip install --no-cache-dir -e .

# Copy .env.example to .env if it doesn't exist (for default model definitions)
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# Create krill storage directory structure
RUN mkdir -p /krill/scrambling \
    /krill/campaigns/quick \
    /krill/campaigns/standard \
    /krill/campaigns/experiments \
    /krill/experiments \
    /krill/results

# Environment
ENV PYTHONUNBUFFERED=1
ENV AKAB_MODE=production
ENV INSTANCE_TYPE=akab

# MCP server runs on stdio
CMD ["python", "-m", "akab"]
