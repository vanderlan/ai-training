# Lab 05 — Multi-Agent Orchestrator

Multi-agent research assistant using the **supervisor pattern** with DeepSeek.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Multi-Agent Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────┐                          │
│                    │ SUPERVISOR  │                          │
│                    │    AGENT    │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
│              ┌────────────┼────────────┐                    │
│              │            │            │                    │
│              ▼            ▼            ▼                    │
│       ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│       │RESEARCHER│ │  WRITER  │ │ REVIEWER │               │
│       │  AGENT   │ │  AGENT   │ │  AGENT   │               │
│       └──────────┘ └──────────┘ └──────────┘               │
│                                                             │
│   Flow:                                                     │
│   1. Supervisor receives task                               │
│   2. Delegates research to Researcher                       │
│   3. Sends research to Writer for polishing                 │
│   4. Optionally sends to Reviewer for quality check         │
│   5. Supervisor synthesizes final output                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python / FastAPI |
| LLM Provider | DeepSeek (via OpenAI-compatible API) |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |
| Deployment | Vercel (serverless) |

## Production Patterns

- **Rate limiting** — Token bucket per IP (5 req/min)
- **Response caching** — 1-hour TTL, SHA-256 keyed
- **Cost tracking** — Daily budget cap with DeepSeek pricing
- **Prompt injection defense** — 10 regex patterns + input validation
- **Output redaction** — API keys, tokens, emails stripped from LLM output
- **Resilience** — Retry with exponential backoff + circuit breaker
- **Graceful fallback** — MockClient when no API key configured

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your DeepSeek API key
export DEEPSEEK_API_KEY=your_key_here

# Run the server
uvicorn main:app --reload

# Open http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/health` | Health check with cost stats |
| POST | `/run` | Run multi-agent task |
| DELETE | `/cache` | Clear response cache |

### POST /run

```json
{
  "task": "Write a brief explanation of how RAG systems work for a technical blog post",
  "max_iterations": 5
}
```

**Response:**
```json
{
  "result": "...",
  "steps": [
    {
      "iteration": 1,
      "action": "delegate",
      "agent": "Researcher",
      "task": "Research RAG systems...",
      "result_preview": "..."
    }
  ],
  "steps_taken": 3,
  "cached": false
}
```

## Project Structure

```
multi-agent-orchestrator/
├── main.py                  # FastAPI app (routes, rate limit, cache)
├── requirements.txt
├── vercel.json
├── api/
│   └── index.py             # Vercel adapter
├── src/
│   ├── __init__.py
│   ├── agents.py            # Worker agents (Researcher, Writer, Reviewer)
│   ├── supervisor.py        # Supervisor agent (coordination logic)
│   ├── llm_client.py        # DeepSeek LLM client
│   ├── cost_tracker.py      # Cost monitoring & budget enforcement
│   ├── resilience.py        # Retry + circuit breaker
│   └── security.py          # Prompt injection + output validation
└── static/
    ├── index.html           # Dashboard UI
    ├── style.css            # Dark-themed stylesheet
    └── app.js               # Frontend logic
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `LLM_PROVIDER` | `deepseek` | LLM provider |
| `RATE_LIMIT_RPM` | `5` | Requests per minute per IP |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `DAILY_BUDGET_USD` | `5.0` | Daily LLM cost limit |
