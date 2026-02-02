#!/bin/bash
# Setup script for server preparation (run this once)

echo "🚀 Setting up server for CI/CD deployments..."

# SSH into server and create shared .env
ssh -i ~/Downloads/instance-7196.pem incorta@35.225.164.65 << 'ENDSSH'
  # Create directory if it doesn't exist
  mkdir -p /home/incorta

  # Check if .env exists
  if [ ! -f /home/incorta/.env ]; then
    echo "📝 Creating /home/incorta/.env file..."
    cat > /home/incorta/.env << 'EOF'
# Slack
SLACK_TOKEN=xoxb-your-token-here
SLACK_BOT_TOKEN=xoxb-your-bot-token-here

# Confluence
CONFLUENCE_URL=https://your-confluence.atlassian.net
CONFLUENCE_EMAIL=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token

# Qdrant
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION_NAME=incorta_docs

# Incorta
INCORTA_HOST=your-host
INCORTA_TENANT=your-tenant
INCORTA_USERNAME=your-username
INCORTA_PASSWORD=your-password
EOF

    echo "✅ Created .env file at /home/incorta/.env"
    echo "⚠️  IMPORTANT: Edit this file with your actual credentials:"
    echo "    nano /home/incorta/.env"
  else
    echo "✅ .env file already exists at /home/incorta/.env"
  fi

  # Set proper permissions
  chmod 600 /home/incorta/.env
  
  echo ""
  echo "📋 Current .env file:"
  cat /home/incorta/.env
ENDSSH

echo ""
echo "✅ Server setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit /home/incorta/.env on the server with your actual credentials"
echo "2. Push code to trigger deployment: git push origin semantic-search-public-docs"
