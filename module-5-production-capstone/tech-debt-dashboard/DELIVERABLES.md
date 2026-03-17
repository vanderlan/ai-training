# Module 5 Capstone — Deliverables

## Option C: Tech Debt Analyzer (HIGH difficulty)

### Core Functionality (40%) ✅

- [x] Codebase indexing — GitHub repo download + file extraction with directory filtering
- [x] Pattern detection — 6 regex-based static detectors (TODO/FIXME, large files, deep nesting, magic numbers, long functions, missing docstrings)
- [x] Priority scoring — Weighted 0-10 score with 5 category breakdown + LLM augmentation
- [x] Report generation — Full structured JSON report with summary stats, per-file scores, and issue details
- [x] LLM-enhanced analysis — DeepSeek-powered pass for medium+ debt files (top 3 critical issues)
- [x] Input validation — URL validation, file size limits, path traversal protection, binary detection

### Architecture (20%) ✅

- [x] Modular code structure — 7 focused modules under `src/`
- [x] Clean separation — detector → scorer → report pipeline
- [x] Documented API — FastAPI auto-generated OpenAPI at `/docs`
- [x] DeepSeek LLM client with MockClient fallback

### Production Ready (20%) ✅

- [x] **Rate Limiting** — Token bucket per-IP (configurable RPM, bounded memory)
- [x] **Caching** — Exact-match with TTL + max-size eviction
- [x] **SSRF Protection** — Regex validation, only `github.com` URLs accepted
- [x] **Retry + Exponential Backoff** — LLM calls retry with jitter (max 2 retries)
- [x] **Circuit Breaker** — Fail-fast after 5 consecutive LLM failures, auto-recovery after 120s
- [x] **Graceful Fallback** — MockClient when credentials missing; circuit breaker fallback to empty results
- [x] **Prompt Injection Defense** — Input isolation markers + 10 injection pattern detectors
- [x] **Output Validation** — Sensitive data redaction (API keys, tokens, emails) in LLM responses
- [x] **Cost Tracking** — Per-call token estimation, daily budget cap, stats in `/health`
- [x] **Input Validation** — File size limits (500KB/file, 10MB total), path traversal blocking, binary detection
- [x] **Structured Logging** — Python `logging` module (replaced all `print()` calls)
- [x] **Environment Configuration** — All secrets via `.env`, documented in `.env.example`
- [x] **Deployment Config** — Vercel (`vercel.json` + `api/index.py`) and Railway (`railway.json`)
- [x] **Health Check** — `/health` endpoint with LLM status, rate limit config, cost stats

### Documentation (10%) ✅

- [x] README with setup instructions and architecture overview
- [x] `.env.example` with all configurable values
- [x] DELIVERABLES.md (this file)
- [x] API documentation at `/docs` (auto-generated)

### Day 5 Curriculum Coverage

| Curriculum Topic | Status | Where |
|---|---|---|
| Rate Limiting (Token Bucket) | ✅ | `main.py` |
| Caching (exact match + TTL) | ✅ | `main.py` |
| Fallback & Retry (exp backoff) | ✅ | `src/resilience.py` |
| Circuit Breaker | ✅ | `src/resilience.py` |
| Graceful Degradation | ✅ | `src/llm_client.py` (MockClient fallback for DeepSeek) |
| Prompt Injection Defense | ✅ | `src/security.py` |
| Input Validation | ✅ | `src/security.py` + `main.py` |
| Output Validation | ✅ | `src/security.py` |
| Cost Management | ✅ | `src/cost_tracker.py` |
| SSRF Protection | ✅ | `main.py` |
| Environment Management | ✅ | `.env` + `.env.example` |
| Structured Logging | ✅ | All modules via `logging` |
| Deployment (Vercel) | ✅ | `vercel.json` + `api/index.py` |
| Health Checks | ✅ | `/health` endpoint |
