# Multi-stage build for Ibn Battouta MCP Server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev pkg-config \
    libjpeg-dev \
    gcc unzip zip \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add user local bin to PATH to suppress warnings
ENV PATH=/root/.local/bin:$PATH

COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port if using HTTP transport (not needed for stdio)
EXPOSE 8080

# Default command - runs the stdio server
CMD ["python", "main.py"]
