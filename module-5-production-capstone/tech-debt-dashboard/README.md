# Tech Debt Dashboard

Analyze GitHub repositories for technical debt using **static regex analysis + LLM-enhanced review**, with a SonarCloud-style code viewer dashboard.

Built as the **Module 5 Capstone** for the Agentic AI Intensive Training Program (Option C: Tech Debt Analyzer).

## Features

- **Two-pass debt detection**: Free regex-based static analysis (pass 1) + targeted LLM review for medium/high-debt files (pass 2)
- **6 static detectors**: TODO/FIXME comments, large files, deep nesting, magic numbers, long functions, missing docstrings
- **SonarCloud-style code viewer**: Click any file to see inline issue annotations alongside the source code
- **Production patterns**: Rate limiting, caching, SSRF protection, retry with backoff, circuit breaker, cost tracking, prompt injection defense

## Production Patterns Implemented

| Pattern | Implementation |
|---|---|
| **Rate Limiting** | Token bucket (per-IP, configurable RPM) |
| **Caching** | Exact-match response cache with TTL + eviction |
| **SSRF Protection** | Regex URL validation — only `github.com` allowed |
| **Retry + Backoff** | Exponential backoff with jitter on LLM calls |
| **Circuit Breaker** | Fail-fast after N consecutive LLM failures |
| **Graceful Fallback** | MockClient when DeepSeek credentials are missing |
| **Prompt Injection Defense** | Input isolation markers + injection pattern detection |
| **Output Validation** | Sensitive data redaction in LLM responses |
| **Cost Tracking** | Per-call token estimation with daily budget cap |
| **Input Validation** | File size limits, path traversal protection, binary detection |
| **Structured Logging** | Python `logging` module throughout |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API key

# 3. Run the server
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload

# 4. Open the dashboard
# http://127.0.0.1:8001
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/health` | Health check + cost stats |
| `POST` | `/analyze/github` | Analyze a public GitHub repo (rate-limited, cached) |
| `POST` | `/analyze/files` | Analyze uploaded `{filename: content}` dict |
| `DELETE` | `/cache` | Clear the response cache |
| `GET` | `/docs` | OpenAPI / Swagger UI |

## Architecture

```
main.py                 FastAPI app + rate limiter + cache + SSRF guard
src/
├── llm_client.py       DeepSeek LLM client (OpenAI-compatible API) + MockClient fallback
├── debt_detector.py    6 regex-based static debt detectors
├── debt_scorer.py      Weighted 0-10 scoring + LLM pass orchestration
├── report_generator.py Pipeline: GitHub download → analyze → score → report
├── security.py         Prompt injection defense + input/output validation
├── resilience.py       Retry with backoff + circuit breaker
├── cost_tracker.py     LLM usage tracking + daily budget cap
└── chunker.py          Language-aware code chunker (from Module 4)
static/
├── index.html          Dashboard layout
├── app.js              Client-side logic + Chart.js + code viewer
└── style.css           Dark theme UI
```

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | `deepseek` or `mock` |
| `DEEPSEEK_API_KEY` | — | Your DeepSeek API key |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name |
| `ENABLE_LLM_PASS` | `true` | Set to `false` for regex-only analysis |
| `RATE_LIMIT_RPM` | `3` | Max GitHub analyze requests per minute per IP |
| `CACHE_TTL_SECONDS` | `3600` | Cache duration for repo results |
| `LLM_TRIGGER_THRESHOLD` | `4.0` | Minimum static score to trigger LLM pass |
| `DAILY_BUDGET_USD` | `5.0` | Daily LLM cost cap (approximate) |

## Deployment

Configured for **Vercel** (serverless):

```bash
npm i -g vercel
vercel --prod
vercel env add DEEPSEEK_API_KEY
vercel env add DEEPSEEK_MODEL
```

Also supports **Railway** via the included `railway.json`.

## Testing

```bash
# Start server first, then:
python test_integration.py
```

Tests cover: health, dashboard serving, static analysis, LLM analysis, SSRF blocking, input validation, caching, and rate limiting.

## License

Training project — not for production use.
