# Multi-stage build for Ibn Battouta MCP Server
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy MCP server requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder to a location accessible by all users
COPY --from=builder /root/.local /usr/local

# Ensure Python scripts are on PATH
ENV PATH="/usr/local/bin:${PATH}"

# Copy application code only (no src dependency)
COPY ibn_battouta_mcp/ ./ibn_battouta_mcp/

# Set working directory to ibn_battouta_mcp
WORKDIR /app/ibn_battouta_mcp

# Create a non-root user (optional but recommended for security)
RUN useradd -m -u 1000 mcp && chown -R mcp:mcp /app
USER mcp

# Expose port if using HTTP transport (not needed for stdio)
EXPOSE 8080

# Health check (adjust based on your needs)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - runs the stdio server
CMD ["python", "server.py"]
