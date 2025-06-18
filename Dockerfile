FROM substrate:latest

# Copy AKAB-specific code
COPY src/akab /app/src/akab

# Ensure PYTHONPATH includes both locations
ENV PYTHONPATH=/app:/app/src:${PYTHONPATH:-}

# Working directory
WORKDIR /app

RUN python -c "import sys; print('PYTHONPATH:', sys.path)"
RUN ls -la /app/
RUN ls -la /app/src/ || echo "No /app/src"

# Start command
CMD ["uvicorn", "akab.server:app", "--host", "0.0.0.0", "--port", "8000"]