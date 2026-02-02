# 🚀 CI/CD Setup Guide for Internal PM Tool

## Overview

This repository uses **GitHub Actions** for automated Docker image building and deployment to your production server.

---

## 🎯 What This Does

### **Automatic on Push:**
1. ✅ Builds Docker image with **layer caching** (fast rebuilds)
2. ✅ Pushes to **GitHub Container Registry** (free, private)
3. ✅ Deploys to your server (`35.225.164.65`)
4. ✅ Zero-downtime container restart
5. ✅ Automatic cleanup of old images

### **Branch-Specific Ports:**
| Branch | Port | Container Name |
|--------|------|----------------|
| `main` | 8080 | internal-pm-main |
| `search-apis` | 9092 | internal-pm-search-apis |
| `semantic-search-public-docs` | 9093 | internal-pm-semantic-search |
| `feat/search-apis` | 9094 | internal-pm-feat-search-apis |
| `feat/public-docs` | 9091 | internal-pm-public-docs |

---

## ⚙️ Setup Instructions

### **Step 1: Add GitHub Secrets**

Go to your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**

Add these 3 secrets:

#### 1. `SSH_PRIVATE_KEY`
```bash
# On your local machine, copy the PEM file content:
cat ~/Downloads/instance-7196.pem
```
Copy the entire output and paste as secret value.

#### 2. `SERVER_HOST`
```
35.225.164.65
```

#### 3. `SERVER_USER`
```
incorta
```

---

### **Step 2: Enable GitHub Actions**

1. Go to your repo → **Settings → Actions → General**
2. Under "Workflow permissions", select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
3. Click **Save**

---

### **Step 3: Enable GitHub Container Registry**

1. Go to your repo → **Settings → Packages**
2. Make sure package visibility is set appropriately
3. GitHub Actions will automatically push images to `ghcr.io/habibahassanein/internal-pm-tool`

---

### **Step 4: Prepare Server (One-time Setup)**

SSH into your server and set up credentials:

```bash
ssh -i ~/Downloads/instance-7196.pem incorta@35.225.164.65

# Create a centralized .env file (shared by all containers)
sudo mkdir -p /home/incorta
sudo chown incorta:incorta /home/incorta

# Create/edit the .env file with your credentials
nano /home/incorta/.env
```

Add your environment variables:
```env
# Slack
SLACK_TOKEN=xoxb-your-token
SLACK_BOT_TOKEN=xoxb-your-token

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

# Add any other environment variables your app needs
```

Save and exit (Ctrl+X, Y, Enter).

---

## 🔄 How to Use

### **Option 1: Automatic Deployment (Push to Branch)**

Just push to any tracked branch:
```bash
git add .
git commit -m "Update feature X"
git push origin semantic-search-public-docs
```

GitHub Actions will automatically:
1. Build the Docker image with caching
2. Push to GitHub Container Registry
3. Deploy to your server on the appropriate port
4. Restart the container

**Check progress:** Go to your repo → **Actions** tab

---

### **Option 2: Manual Deployment**

Go to your repo → **Actions** → **Build and Deploy MCP Server** → **Run workflow**

Select the branch you want to deploy and click **Run workflow**.

---

## 📊 Monitoring

### **Check Deployment Status:**
```bash
# On your server
ssh -i ~/Downloads/instance-7196.pem incorta@35.225.164.65

# List running containers
docker ps

# Check specific container logs
docker logs internal-pm-semantic-search -f

# Check container health
curl http://localhost:9093/mcp
```

### **View GitHub Actions Logs:**
1. Go to your repo → **Actions**
2. Click on the latest workflow run
3. View detailed logs for each step

---

## 🚀 Benefits of This Setup

### **Before (Manual):**
```bash
# SSH into server
ssh -i instance-7196.pem incorta@35.225.164.65

# Pull code
git pull

# Build image (slow, no caching)
docker build -t internal-pm:semantic-search-public-docs .

# Stop old container
docker stop boring_proskuriakova

# Run new container
docker run -d -p 9093:8080 internal-pm:semantic-search-public-docs

# Total time: 5-10 minutes
```

### **After (Automated):**
```bash
# Just push
git push origin semantic-search-public-docs

# Done! GitHub Actions handles everything
# Total time: 2-3 minutes (with caching)
```

---

## ⚡ Performance Optimizations

### **Docker Layer Caching:**
- GitHub Actions caches Docker layers
- Rebuilds only changed layers
- **First build:** ~5 minutes
- **Subsequent builds:** ~1-2 minutes

### **Multi-Stage Caching:**
- System dependencies cached
- Python packages cached
- Only your code is rebuilt

---

## 🔧 Troubleshooting

### **Problem: Deployment fails with SSH error**
**Solution:** Check that `SSH_PRIVATE_KEY` secret is correctly formatted (entire PEM file content).

### **Problem: Container fails to start**
**Solution:** Check logs on server:
```bash
docker logs internal-pm-semantic-search
```
Usually it's a missing environment variable in `/home/incorta/.env`.

### **Problem: Image build is slow**
**Solution:** 
1. Make sure `.dockerignore` is in place
2. Check GitHub Actions cache is enabled
3. Order Dockerfile commands from least to most frequently changed

### **Problem: Port conflict**
**Solution:** Edit `.github/workflows/deploy.yml` and change port mapping for your branch.

---

## 🎛️ Customization

### **Add a New Branch:**

Edit `.github/workflows/deploy.yml` and add your branch to the port mapping:

```yaml
case "$BRANCH_NAME" in
  "main")
    PORT=8080
    CONTAINER_NAME="internal-pm-main"
    ;;
  "your-new-branch")  # ← Add here
    PORT=9096          # ← Choose available port
    CONTAINER_NAME="internal-pm-your-branch"
    ;;
```

### **Change Deploy Trigger:**

Edit `.github/workflows/deploy.yml`:

```yaml
on:
  push:
    branches:
      - main
      - your-branch  # Add/remove branches here
```

### **Deploy on Tag Instead:**

```yaml
on:
  push:
    tags:
      - 'v*'  # Deploy on version tags like v1.0.0
```

---

## 📝 Best Practices

1. ✅ **Use separate branches for testing** before merging to production
2. ✅ **Monitor the Actions tab** for failed deployments
3. ✅ **Keep secrets up to date** (rotate tokens regularly)
4. ✅ **Use `.dockerignore`** to exclude unnecessary files
5. ✅ **Test locally first** with `docker build`
6. ✅ **Review logs** after each deployment

---

## 🔐 Security Notes

- ✅ Secrets are encrypted by GitHub
- ✅ Secrets are never exposed in logs
- ✅ SSH key has minimal permissions (deployment only)
- ✅ Images are private in GitHub Container Registry
- ✅ Environment variables are mounted, not baked into image

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

---

**Questions?** Check the Actions tab for detailed logs or review this guide.

**Ready to deploy?** Just `git push`! 🚀
