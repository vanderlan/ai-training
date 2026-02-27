# Quick Start Guide - Migration Workflow System

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd module-3-agent-architectures/migration-workflow-system
pip install -r requirements.txt
```

### 2. Set Your API Key
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 3. Start the Server
```bash
uvicorn src.main:app --reload
```

### 4. Open API Docs
Visit: **http://localhost:8000/docs**

## Testing Your Setup

### Run Unit Tests
```bash
python tests.py
```

Expected output:
```
============================================================
🧪 Running Tests
============================================================

✅ test_state_initialization
✅ test_state_to_dict
✅ test_migration_step
...
============================================================
Results: 8 passed, 0 failed
============================================================
```

### Test API Locally
```bash
curl -X POST http://localhost:8000/health
```

## First Migration

### Via Web UI
1. Visit http://localhost:8000/docs
2. Click "Try it out" on `/migrate` endpoint
3. Enter request body:
```json
{
  "source_framework": "express",
  "target_framework": "fastapi",
  "files": {
    "server.js": "const app = require('express')(); app.get('/api/users', (req, res) => res.json([])); app.listen(3000);"
  }
}
```
4. Click "Execute"

### Via Command Line
```bash
python cli.py express fastapi server.js
```

## Understanding the Output

The API returns a complete migration result:

```json
{
  "success": true,
  "source_framework": "express",
  "target_framework": "fastapi",
  "phase": "complete",
  "plan_executed": [
    {
      "id": 1,
      "description": "Setup FastAPI application skeleton",
      "status": "completed",
      "output_files": ["app.py"]
    },
    ...
  ],
  "migrated_files": {
    "app.py": "from fastapi import FastAPI\n..."
  },
  "verification": {
    "verification_status": "passed",
    "checks": { ... }
  },
  "errors": [],
  "iterations": 4
}
```

## Next Steps

### 1. Explore the Code
- **Agent Logic**: [src/agent.py](src/agent.py) - Core 4-phase workflow
- **State Management**: [src/state.py](src/state.py) - Data structures
- **LLM Integration**: [src/llm_client.py](src/llm_client.py) - OpenAI integration
- **API Endpoints**: [src/main.py](src/main.py) - FastAPI server

### 2. Customize for Your Needs
- Edit prompts in [src/prompts.py](src/prompts.py)
- Add new phases by extending `MigrationAgent`
- Add framework examples to [examples.py](examples.py)

### 3. Deploy to Production
- See [DEPLOYMENT.md](DEPLOYMENT.md) for Railway, Vercel, or Docker
- Configure environment variables
- Set up monitoring and logging

## Common Tasks

### Migrate a Directory
```bash
python cli.py express fastapi ./my-express-app --output ./migrated
```

### Get JSON Output
```bash
python cli.py express fastapi ./app --json > migration_result.json
```

### Run Examples
```bash
python examples.py
```

### Test Different Frameworks
```bash
# Flask to FastAPI
python cli.py flask fastapi app.py

# Vue to React
python cli.py vue react component.vue
```

## Architecture Overview

The system uses a 4-phase agent loop:

1. **ANALYSIS** 🔍
   - Reads source files
   - Identifies patterns
   - Lists dependencies

2. **PLANNING** 📋
   - Creates migration plan
   - Breaks down into steps
   - Orders dependencies

3. **EXECUTION** ⚙️
   - Executes each step
   - Generates transformed code
   - Tracks progress

4. **VERIFICATION** ✅
   - Validates output
   - Checks correctness
   - Reports issues

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Port 8000 Already in Use
```bash
# Use different port
uvicorn src.main:app --port 8001
```

### Dependencies Not Installing
```bash
# Upgrade pip first
pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

### LLM Connection Issues
```bash
# Check connection
python -c "import openai; openai.api_key = 'test'; print('✓ OpenAI module installed')"
```

## Documentation

- **Full README**: [README.md](README.md)
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Deliverables**: [DELIVERABLES.md](DELIVERABLES.md)
- **API Documentation**: http://localhost:8000/docs

## Learning Resources

Inside this repository:
- `Training_Content/curriculum/day3-agents.md` - Agent concepts
- `Training_Content/labs/lab03-migration-workflow/` - Lab instructions

## Getting Help

1. Check the logs - they show phase progression
2. Review API response - includes detailed error messages
3. Check [DEPLOYMENT.md](DEPLOYMENT.md) for common issues
4. Look at [examples.py](examples.py) for working examples

---

**Ready to migrate?** Start with the simple Express to FastAPI example above!
