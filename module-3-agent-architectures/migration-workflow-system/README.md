# Migration Workflow System

**Module 3: Agent Architectures**

An AI-powered multi-phase agent system for autonomous code migrations using OpenAI's GPT and LangGraph-style patterns.

## 🎯 System Architecture

The system implements the **Observe → Think → Act** agent loop across 4 phases:

```
┌──────────────────────────────────────────────────────────┐
│           Migration Workflow Agent Loop                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐                                        │
│  │  OBSERVE    │  Read source code, understand context  │
│  └──────┬──────┘                                        │
│         │                                               │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │   THINK     │  LLM analyzes and plans steps          │
│  │  (LLM)      │                                        │
│  └──────┬──────┘                                        │
│         │                                               │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │    ACT      │  Execute migration steps               │
│  │ (Transform) │                                        │
│  └──────┬──────┘                                        │
│         │                                               │
│         ▼                                               │
│  ┌─────────────┐                                        │
│  │   VERIFY    │  Validate results                      │
│  └─────────────┘                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## 🏗️ Core Components

### 1. **State Management** (`src/state.py`)
- `MigrationState`: Tracks complete migration progress
- `MigrationStep`: Individual tasks in the plan
- `Phase`: ANALYSIS → PLANNING → EXECUTION → VERIFICATION → COMPLETE

### 2. **LLM Client** (`src/llm_client.py`)
- Communicates with OpenAI API
- Parses structured outputs from LLM
- Handles JSON response parsing

### 3. **Agent Core** (`src/agent.py`)
- Implements the 4-phase workflow
- Orchestrates state transitions
- Error handling and fallback mechanisms

### 4. **System Prompts** (`src/prompts.py`)
- Phase-specific guidance for the LLM
- Structured output specifications
- JSON schema definitions

### 5. **FastAPI Server** (`src/main.py`)
- REST endpoints for migrations
- Health checks and status tracking
- Example migration suggestions

## 🚀 Quick Start

### Setup

```bash
# Clone and navigate to project
cd module-3-agent-architectures/migration-workflow-system

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Server

```bash
uvicorn src.main:app --reload
```

Then visit: http://localhost:8000/docs

### Example API Call

```bash
curl -X POST http://localhost:8000/migrate \
  -H "Content-Type: application/json" \
  -d '{
    "source_framework": "express",
    "target_framework": "fastapi",
    "files": {
      "server.js": "const app = require(\"express\")(); app.get(\"/\", (req, res) => res.json({})); app.listen(3000);"
    }
  }'
```

## 📚 Phase Details

### Phase 1: ANALYSIS 🔍
**Goal**: Understand source code structure and patterns

```python
# Actions:
# - Parse all source files
# - Identify framework patterns
# - Extract dependencies
# - Note potential challenges

# Output (MigrationState.analysis):
{
    "framework_patterns": [...],
    "dependencies": [...],
    "key_components": {...},
    "potential_issues": [...],
    "complexity_assessment": {...}
}
```

### Phase 2: PLANNING 📋
**Goal**: Create step-by-step migration plan

```python
# Actions:
# - Break work into logical steps
# - Identify dependencies between steps
# - Order steps for efficiency
# - Estimate complexity

# Output (MigrationState.plan):
[
    {
        "id": 1,
        "description": "...",
        "dependencies": [],
        "estimated_complexity": "medium",
        "input_files": [...],
        "output_files": [...]
    },
    ...
]
```

### Phase 3: EXECUTION ⚙️
**Goal**: Transform code according to plan

```python
# For each step:
# - Load input files
# - Call LLM with step description
# - Generate transformed code
# - Store results

# Output (MigrationState.migrated_files):
{
    "app.py": "# Migrated code...",
    "models.py": "# Migrated code...",
    ...
}
```

### Phase 4: VERIFICATION ✅
**Goal**: Validate migrated code quality

```python
# Actions:
# - Check syntax correctness
# - Verify API compatibility
# - Assess functional equivalence
# - Review code quality

# Output (MigrationState.verification_result):
{
    "verification_status": "passed",
    "checks": {...},
    "issues": [...],
    "recommendations": [...]
}
```

## 🧪 Testing

Run unit tests:

```bash
python tests.py
```

Run example migrations:

```bash
# Express to FastAPI
python cli.py express fastapi examples/express-server.js

# Flask to FastAPI
python cli.py flask fastapi examples/flask-app.py
```

## 💻 CLI Usage

```bash
# Single file migration
python cli.py express fastapi server.js

# Directory migration
python cli.py express fastapi ./src

# Save to specific output directory
python cli.py express fastapi ./src --output ./migrated-app

# Output as JSON
python cli.py express fastapi ./src --json
```

## 🔧 Configuration

### Environment Variables

```bash
# .env
OPENAI_API_KEY=your-api-key-here
MODEL=gpt-4-turbo
PORT=8000
```

### Customization Points

1. **Modify system prompts** in `src/prompts.py` to change agent behavior
2. **Extend state** in `src/state.py` for new migration data
3. **Add tools** by extending `LLMClient` methods
4. **Add phases** by extending `MigrationAgent._step()` method

## 📊 API Reference

### POST /migrate
Execute a complete migration workflow.

**Request:**
```json
{
    "source_framework": "express",
    "target_framework": "fastapi",
    "files": {
        "filename.js": "file content"
    }
}
```

**Response:**
```json
{
    "success": true,
    "source_framework": "express",
    "target_framework": "fastapi",
    "phase": "complete",
    "plan_executed": [...],
    "migrated_files": {...},
    "verification": {...},
    "errors": [],
    "iterations": 4
}
```

### GET /health
Check API health status.

### GET /examples
Get example migrations.

## 🎓 Learning Concepts

### Agent Loop
- **Observe**: Perception phase - gather input and context
- **Think**: Reasoning phase - LLM decides next action
- **Act**: Action phase - execute based on LLM decision
- **Iterate**: Repeat until goal achieved

### State Management
- Persistent state across agent iterations
- Phase-based progression tracking
- Error handling and recovery strategies

### Multi-Phase Architecture
- Clear responsibilities per phase
- Clean phase transitions
- Error handling at each phase

### LLM Integration
- Structured prompts for consistent output
- JSON parsing for tool compatibility
- Context windowing for large migrations

## 🚢 Deployment

- [ ] Configure agent settings
- [ ] Set up Docker container
- [ ] Deploy to Railway or Vercel
- [ ] Configure environment variables
- [ ] Test in production
- [ ] Set up memory storage
- [ ] Test with sample migrations
- [ ] Document usage and patterns

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Pattern Paper](https://arxiv.org/abs/2210.03629)
- Add more resources as needed

---

**Part of Taller AI Training Program - Module 3**
