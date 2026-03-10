# Deployment Guide

## 🚀 Deployment Options

### Option 1: Railway (Recommended)

Railway provides easy deployment with automatic builds.

#### Steps:

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   railway init
   ```

4. **Add Environment Variables**
   ```bash
   # Set your chosen provider (deepseek, anthropic, openai, or gemini)
   railway variables set LLM_PROVIDER=deepseek
   railway variables set DEEPSEEK_API_KEY=your_key_here
   # or for other providers:
   # railway variables set ANTHROPIC_API_KEY=your_key_here
   # railway variables set OPENAI_API_KEY=your_key_here
   ```

5. **Deploy**
   ```bash
   railway up
   ```

6. **Get URL**
   ```bash
   railway open
   ```

### Option 2: Docker

Deploy anywhere using Docker.

#### Build Image

```bash
docker build -t rag-system .
```

#### Run Container

```bash
docker run -p 8000:8000 \
  -e LLM_PROVIDER=deepseek \
  -e DEEPSEEK_API_KEY=your_key \
  rag-system
```

#### Docker Compose

```yaml
version: '3.8'
services:
  rag-system:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER:-deepseek}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
```

Run with:
```bash
docker-compose up
```

### Option 3: Vercel

Vercel supports Python serverless functions.

#### Steps:

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login and deploy from this project folder**
   ```bash
   cd module-4-rag-evaluation/rag-system
   vercel login
   vercel
   ```

   Recommended answers:
   - Set up and deploy? **Y**
   - Link to existing project? **N** (or **Y** if reusing one)
   - Project name: **rag-system** (or your preferred name)
   - In which directory is your code located? **./**
   - Want to modify settings? **N**

3. **Set Environment Variables**
   ```bash
   vercel env add LLM_PROVIDER        # e.g. deepseek
   vercel env add DEEPSEEK_API_KEY    # if using DeepSeek
   vercel env add ANTHROPIC_API_KEY   # if using Anthropic
   vercel env add OPENAI_API_KEY      # if using OpenAI
   ```

   **Production is currently configured with DeepSeek.**  
   Live URL: https://rag-system-vanderlan-lab4.vercel.app

4. **Deploy to production**
   ```bash
   vercel --prod
   ```

5. **Optional: Dashboard setup (Import from GitHub)**
   - Framework Preset: **Other**
   - Root Directory: **module-4-rag-evaluation/rag-system**
   - Build Command: leave empty
   - Output Directory: leave empty

**Note:** Vercel has limitations:
- 50MB deployment size limit
- 10s execution timeout for serverless functions
- Index storage is ephemeral per deployment/instance

### Option 4: AWS EC2

For full control and persistent storage.

#### Steps:

1. **Launch EC2 Instance** (Ubuntu 22.04, t3.small minimum)

2. **SSH into Instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git
   ```

4. **Clone and Setup**
   ```bash
   git clone <your-repo>
   cd rag-system
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Set Environment Variables**
   ```bash
   export LLM_PROVIDER=deepseek
   export DEEPSEEK_API_KEY=your_key
   # or: export ANTHROPIC_API_KEY / OPENAI_API_KEY for other providers
   ```

6. **Run with Systemd**

   Create `/etc/systemd/system/rag-system.service`:
   ```ini
   [Unit]
   Description=RAG System API
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/rag-system
   Environment="LLM_PROVIDER=deepseek"
   Environment="DEEPSEEK_API_KEY=your_key"
   ExecStart=/home/ubuntu/rag-system/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:
   ```bash
   sudo systemctl enable rag-system
   sudo systemctl start rag-system
   ```

7. **Setup Nginx Reverse Proxy**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 🔒 Security Considerations

1. **Environment Variables**: Never commit API keys to git
2. **CORS**: Configure appropriately for production
3. **Rate Limiting**: Add rate limiting middleware
4. **Authentication**: Add API key authentication for production
5. **HTTPS**: Use SSL certificates (Let's Encrypt)

## 📊 Monitoring

### Health Check Endpoint

```bash
curl https://your-domain.com/health
```

### Metrics to Track

- Request latency
- Embedding generation time
- Vector search time
- LLM response time
- Error rates
- Vector store size

## 🔄 CI/CD

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Railway
        run: |
          npm install -g @railway/cli
          railway up
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

## 💾 Data Persistence

Local and Docker runs store index data in `./chroma_db/`.

On Vercel, the app uses `/tmp/chroma_db` because serverless functions require a writable temp directory. This storage is ephemeral and not shared across cold starts.

For production:
- Mount persistent volume in Docker
- Use external storage (S3, EFS)
- Regular backups of vector database

## 🧪 Testing Deployment

```bash
# Health check
curl https://rag-system-vanderlan-lab4.vercel.app/health

# Index files
curl -X POST https://rag-system-vanderlan-lab4.vercel.app/index/files \
  -H "Content-Type: application/json" \
  -d '{"files": {"test.py": "def hello():\n    return \"hi\""}}'

# Index a GitHub repository
curl -X POST https://rag-system-vanderlan-lab4.vercel.app/index/github \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/owner/repo"}'

# Query
curl -X POST https://rag-system-vanderlan-lab4.vercel.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the hello function do?"}'
```
