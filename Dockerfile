FROM substrate:latest

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy AKAB-specific code
COPY src/akab /app/akab
COPY data /app/data
COPY requirements.txt /app/requirements-akab.txt

# Install AKAB-specific dependencies (if any)
RUN pip install --no-cache-dir -r requirements-akab.txt || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "akab.server:app", "--host", "0.0.0.0", "--port", "8000"]
