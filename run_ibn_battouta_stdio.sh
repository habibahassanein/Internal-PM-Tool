#!/bin/bash
# Wrapper script to run Ibn Battouta MCP server with environment variables loaded

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables from .env file (safer method)
if [ -f "$SCRIPT_DIR/ibn_battouta_mcp/.env" ]; then
    set -a  # Automatically export all variables
    source "$SCRIPT_DIR/ibn_battouta_mcp/.env"
    set +a
fi

# Add source directory to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the HTTP server (not stdio, since that's what currently works)
cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/.venv_mcp311/bin/python" -m ibn_battouta_mcp.server --port 8081
