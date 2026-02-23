# Code Analyzer Agent - Deployment Guide

**Live URL**: https://code-analyzer-agent.vercel.app

This document provides comprehensive deployment instructions for the Code Analyzer Agent on Vercel.

---

## 🚀 Quick Deploy to Vercel

### Prerequisites

- [Vercel account](https://vercel.com/signup) (free tier works)
- [Vercel CLI](https://vercel.com/cli): `npm install -g vercel`
- OpenAI, Anthropic, or Google API key

### One-Command Deploy

```bash
# From project directory
vercel --prod
```

Follow the prompts to:
1. Link to existing project or create new one
2. Confirm project settings
3. Deploy to production

After deployment, set environment variables in the [Vercel Dashboard](https://vercel.com/dashboard).

---

## 🔧 Configuration

### Environment Variables

Set these in your Vercel project settings:

**Required:**
- `LLM_PROVIDER` - Choose one: `openai`, `anthropic`, or `google`

**API Keys (at least one required):**
- `OPENAI_API_KEY` - Your OpenAI API key (starts with `sk-...`)
- `ANTHROPIC_API_KEY` - Your Anthropic API key (starts with `sk-ant-...`)
- `GOOGLE_API_KEY` - Your Google API key

### Setting Environment Variables

#### Via Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Select your project (`code-analyzer-agent`)
3. Go to **Settings** → **Environment Variables**
4. Add each variable:
   - **Name**: `LLM_PROVIDER`
   - **Value**: `openai` (or `anthropic`/`google`)
   - **Environment**: `Production`, `Preview`, `Development`
5. Click **Save**
6. Redeploy: `vercel --prod`

#### Via Vercel CLI

```bash
# Add environment variables
vercel env add LLM_PROVIDER production
# When prompted, enter: openai

vercel env add OPENAI_API_KEY production
# When prompted, enter: sk-your-actual-key-here

# Redeploy to apply changes
vercel --prod
```

---

## 📡 API Endpoints

All endpoints are deployed at: `https://code-analyzer-agent.vercel.app/api/`

### 1. General Code Analysis

```http
POST /api/analyze
Content-Type: application/json

{
  "code": "def hello():\n    print('world')",
  "language": "python"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "summary": "Brief overview of code quality",
    "issues": [
      {
        "severity": "medium",
        "line": 2,
        "category": "style",
        "description": "Missing docstring",
        "suggestion": "Add function docstring describing purpose"
      }
    ],
    "suggestions": ["Use type hints", "Add error handling"],
    "metrics": {
      "complexity": "low",
      "readability": "good",
      "test_coverage_estimate": "none"
    }
  }
}
```

### 2. Security-Focused Analysis

```http
POST /api/analyze/security
Content-Type: application/json

{
  "code": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
  "language": "python"
}
```

**Focus Areas:**
- SQL injection vulnerabilities
- Command injection risks
- Path traversal attacks
- Hardcoded credentials
- Input validation issues
- Authentication/authorization flaws

### 3. Performance-Focused Analysis

```http
POST /api/analyze/performance
Content-Type: application/json

{
  "code": "for i in range(len(items)):\n    process(items[i])",
  "language": "python"
}
```

**Focus Areas:**
- Algorithm complexity (Big O)
- Inefficient loops
- Memory allocation
- Database query optimization
- Caching opportunities
- Unnecessary computations

---

## 🧪 Testing the Deployment

### Using curl

**General Analysis:**
```bash
curl -X POST "https://code-analyzer-agent.vercel.app/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def test():\n    pass",
    "language": "python"
  }'
```

**Security Analysis:**
```bash
curl -X POST "https://code-analyzer-agent.vercel.app/api/analyze/security" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import os\nos.system(user_input)",
    "language": "python"
  }'
```

### Using PowerShell

```powershell
$body = @{
    code = "def unsafe_query(user_id):`n    query = f`"SELECT * FROM users WHERE id = {user_id}`"`n    return db.execute(query)"
    language = "python"
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "https://code-analyzer-agent.vercel.app/api/analyze/security" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response | ConvertTo-Json -Depth 10
```

### Using JavaScript/TypeScript

```typescript
const response = await fetch('https://code-analyzer-agent.vercel.app/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: 'function hello() {\n  console.log("world");\n}',
    language: 'javascript'
  })
});

const data = await response.json();
console.log(data.analysis);
```

### Using Python

```python
import requests

response = requests.post(
    'https://code-analyzer-agent.vercel.app/api/analyze',
    json={
        'code': 'def hello():\n    print("world")',
        'language': 'python'
    }
)

analysis = response.json()['analysis']
print(f"Summary: {analysis['summary']}")
print(f"Issues found: {len(analysis['issues'])}")
```

---

## 🌐 Web Interface

The deployment includes a beautiful web UI at the root URL:

**URL**: https://code-analyzer-agent.vercel.app

**Features:**
- ✨ Modern gradient design
- 📝 Code input with syntax highlighting
- 🔀 Multi-language support (Python, JavaScript, TypeScript, Java, C#, Go, Rust)
- 🔍 Three analysis modes (General, Security, Performance)
- 📊 Color-coded severity levels
- 📱 Responsive mobile design

---

## 🗂️ Project Structure for Deployment

```
code-analyzer-agent/
├── api/
│   └── index.ts           # Vercel serverless function
├── public/
│   └── index.html         # Web UI (served at root)
├── src/
│   ├── analyzer.ts        # Analysis logic
│   ├── llm-client.ts      # LLM provider abstraction
│   ├── prompts.ts         # System prompts
│   └── types.ts           # Zod schemas
├── vercel.json            # Deployment config
└── .vercelignore          # Exclude from deployment
```

### vercel.json Configuration

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.ts",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/analyze/security",
      "dest": "api/index.ts"
    },
    {
      "src": "/api/analyze/performance",
      "dest": "api/index.ts"
    },
    {
      "src": "/api/analyze",
      "dest": "api/index.ts"
    },
    {
      "src": "/",
      "dest": "public/index.html"
    }
  ]
}
```

---

## 🐛 Troubleshooting

### Issue: "FUNCTION_INVOCATION_FAILED" or 500 errors

**Cause**: Missing or incorrect environment variables

**Solutions:**
1. Verify environment variables are set in Vercel dashboard
2. Check variable names match exactly: `LLM_PROVIDER`, `OPENAI_API_KEY`, etc.
3. Ensure API key is valid and has sufficient credits
4. Check Vercel function logs: `vercel logs [deployment-url]`

### Issue: "Unknown provider" error

**Cause**: `LLM_PROVIDER` environment variable has wrong value

**Solution**: Set `LLM_PROVIDER` to exactly `openai`, `anthropic`, or `google` (lowercase)

### Issue: API key not working

**Cause**: API key format or permissions issue

**Solutions:**
1. Verify API key starts with correct prefix:
   - OpenAI: `sk-proj-...` or `sk-...`
   - Anthropic: `sk-ant-...`
   - Google: varies
2. Check API key has not expired
3. Verify billing is set up for the LLM provider
4. Test API key locally first to confirm it works

### Issue: Deployment builds but API doesn't respond

**Cause**: TypeScript compilation or import path issues

**Solutions:**
1. Check build logs: `vercel logs`
2. Verify all imports use `.js` extension: `import { x } from './file.js'`
3. Ensure `@vercel/node` is in `package.json` dependencies
4. Check `tsconfig.json` has correct module settings

### Issue: CORS errors in browser

**Cause**: Missing CORS headers

**Solution**: The API includes proper CORS headers. If still seeing errors:
- Clear browser cache
- Try different origin
- Check browser console for specific CORS error details

---

## 📊 Monitoring & Logs

### View Deployment Logs

```bash
# View latest deployment logs
vercel logs

# View logs for specific deployment
vercel logs [deployment-url]

# Stream logs in real-time
vercel logs --follow
```

### Check Deployment Status

```bash
# List all deployments
vercel ls

# Get deployment details
vercel inspect [deployment-url]
```

### Performance Metrics

Access analytics in Vercel Dashboard:
1. Go to your project
2. Click **Analytics** tab
3. View:
   - Request count
   - Response times
   - Error rates
   - Bandwidth usage

---

## 💰 Cost Considerations

### Vercel Costs (Free Tier)

- ✅ 100 GB bandwidth/month
- ✅ 100 serverless function hours/month
- ✅ Unlimited deployments
- ⚠️ No cold starts for first ~10 min of activity

### LLM API Costs (Approximate)

**OpenAI GPT-4:**
- Input: $10 per 1M tokens
- Output: $30 per 1M tokens
- Typical analysis: ~1,000 input + 500 output tokens = $0.025/analysis

**Anthropic Claude 3.5 Sonnet:**
- Input: $3 per 1M tokens
- Output: $15 per 1M tokens  
- Typical analysis: ~1,000 input + 500 output tokens = $0.011/analysis

**Google Gemini 1.5 Flash:**
- Input: FREE up to 2M tokens/day
- Output: FREE up to 2M tokens/day
- Best for high-volume testing

---

## 🔐 Security Best Practices

1. **Never commit `.env` files** - Use `.gitignore`
2. **Rotate API keys regularly** - Every 90 days minimum
3. **Use Vercel environment variables** - Never hardcode secrets
4. **Implement rate limiting** - Prevent abuse (future enhancement)
5. **Monitor usage** - Set up billing alerts on LLM providers
6. **Validate all inputs** - Never trust user input
7. **Use HTTPS only** - Vercel provides this automatically

---

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel CLI Reference](https://vercel.com/docs/cli)
- [Serverless Functions](https://vercel.com/docs/functions/serverless-functions)
- [Environment Variables Guide](https://vercel.com/docs/projects/environment-variables)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com)

---

**Successfully Deployed!** 🎉

Your Code Analyzer Agent is now live at: https://code-analyzer-agent.vercel.app
