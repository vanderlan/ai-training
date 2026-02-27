# Deployment Guide

## Local Development

### Prerequisites
- Python 3.9+
- pip
- OpenAI API key

### Setup

```bash
cd migration-workflow-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### Run Locally

```bash
# Start development server
uvicorn src.main:app --reload

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Docker

### Build and Run

```bash
# Build Docker image
docker build -t migration-workflow-system .

# Run container
docker run \
  -e OPENAI_API_KEY=your-api-key \
  -p 8000:8000 \
  migration-workflow-system
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      MODEL: gpt-4-turbo
    restart: unless-stopped
```

## Railway Deployment

### Setup Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Set environment variables
railway variables set OPENAI_API_KEY "your-api-key"
railway variables set MODEL "gpt-4-turbo"
```

### Deploy

```bash
# Deploy to Railway
railway up

# View logs
railway logs

# The service will be available at: https://<project>.railway.app
```

### Railway Configuration

The `railway.json` file configures:
- Python runtime
- Build command: `pip install -r requirements.txt`
- Start command: FastAPI with uvicorn
- Port: 8000

## Vercel Deployment

### Prerequisites
- Vercel account
- Vercel CLI installed

### Deploy with Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Add environment variables in Vercel dashboard:
# - OPENAI_API_KEY
# - MODEL=gpt-4-turbo
```

### Vercel Configuration

The `vercel.json` configures Python runtime and build settings.

## Production Checklist

- [ ] Set `DEBUG=false` in environment
- [ ] Configure CORS for your domain
- [ ] Set up API rate limiting
- [ ] Configure logging and monitoring
- [ ] Set up backup/recovery procedures
- [ ] Configure API authentication
- [ ] Enable HTTPS
- [ ] Set up health check monitoring
- [ ] Configure auto-scaling if needed
- [ ] Set up incident alerting

## Environment Variables

Key environment variables:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional
MODEL=gpt-4-turbo                  # LLM model to use
PORT=8000                          # Server port
DEBUG=false                        # Debug mode
```

## Monitoring and Logging

### Health Check

```bash
curl https://your-deployment.app/health
```

### Logs

- Local: Check uvicorn console output
- Railway: `railway logs`
- Vercel: Dashboard → Logs

## Scaling Considerations

For production:

1. **Request Queue**: Implement message queue for async migrations
2. **Caching**: Cache analysis results for repeated frameworks
3. **Rate Limiting**: Limit requests per user/IP
4. **Database**: Store migration history
5. **Worker Pool**: Use background tasks for long-running migrations

## Troubleshooting

### API Key Issues

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API connection
python -c "from src.llm_client import LLMClient; c = LLMClient(); print('✓ Connected')"
```

### Port Already in Use

```bash
# Use different port
uvicorn src.main:app --port 8001
```

### Memory Issues

- Reduce `max_iterations` in `MigrationState`
- Split large files before migration
- Implement streaming responses

## Performance Tuning

1. **Increase workers**: `uvicorn --workers 4`
2. **Connection pooling**: For database operations
3. **Cache frameworks**: Store common framework patterns
4. **Async operations**: Use async/await for I/O operations

## Security Hardening

1. Always use HTTPS in production
2. Rotate API keys regularly
3. Implement request validation
4. Log security events
5. Monitor rate limits
6. Use VPC/network isolation

## Rollback

To rollback to previous version:

### Railway
```bash
railway down
# Deploy previous version
```

### Vercel
```bash
vercel rollback
```

### Docker
```bash
docker pull migration-workflow-system:previous-tag
docker run -e OPENAI_API_KEY=... previous-tag
```

## Support

For deployment issues:
- Check logs (local/Railway/Vercel)
- Verify environment variables are set
- Test API locally first
- Contact OpenAI support for API issues
