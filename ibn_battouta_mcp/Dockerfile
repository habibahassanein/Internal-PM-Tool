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

# Add user local bin to PATH to suppress warnings
ENV PATH=/root/.local/bin:$PATH

COPY --from=builder /root/.local /usr/local

# Install Python dependencies
RUN pip install -r requirements.txt --system

COPY . /app

# Expose port if using HTTP transport (not needed for stdio)
EXPOSE 8080

# Default command - runs the stdio server
CMD ["python", "server.py"]
