"""
Tech Debt Dashboard — FastAPI application
Production patterns applied:
  • Per-IP rate limiting (token bucket, 3 req/min on /analyze/github)
  • Response cache with 1-hour TTL
  • SSRF protection: only github.com URLs accepted
  • Graceful DeepSeek LLM fallback via MockClient
  • Prompt injection defense (input isolation)
  • Retry with exponential backoff + circuit breaker
  • Cost tracking with daily budget cap
  • Output validation (sensitive data redaction)
  • Structured logging
"""
import logging
import os
import re
import time
import threading
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from src.llm_client import get_llm_client
from src.report_generator import ReportGenerator, _parse_github_url
from src.security import validate_uploaded_files
from src.cost_tracker import cost_tracker

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tech_debt_dashboard")

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tech Debt Dashboard",
    description="Analyze GitHub repos for technical debt using static analysis + LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for dashboard UI
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ---------------------------------------------------------------------------
# LLM + Report generator setup
# ---------------------------------------------------------------------------
_provider = os.getenv("LLM_PROVIDER", "deepseek").strip()
_llm = get_llm_client(_provider)
_enable_llm = os.getenv("ENABLE_LLM_PASS", "true").strip().lower() == "true"
_generator = ReportGenerator(_llm, enable_llm_pass=_enable_llm)

# ---------------------------------------------------------------------------
# Rate limiter (token bucket, per IP)
# ---------------------------------------------------------------------------
_RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "3"))  # requests per minute

class _TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens/second
        self.last_refill = time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


_rate_buckets: Dict[str, _TokenBucket] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(ip: str) -> bool:
    with _rate_lock:
        # Prevent unbounded growth: evict oldest bucket if too many IPs tracked
        if ip not in _rate_buckets and len(_rate_buckets) >= 1000:
            oldest_ip = min(_rate_buckets, key=lambda k: _rate_buckets[k].last_refill)
            del _rate_buckets[oldest_ip]
        if ip not in _rate_buckets:
            _rate_buckets[ip] = _TokenBucket(
                capacity=_RATE_LIMIT_RPM,
                refill_rate=_RATE_LIMIT_RPM / 60.0,
            )
        return _rate_buckets[ip].acquire()


# ---------------------------------------------------------------------------
# Response cache (exact URL match, 1-hour TTL)
# ---------------------------------------------------------------------------
_CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
_cache: Dict[str, tuple] = {}  # url → (timestamp, report)
_cache_lock = threading.Lock()


def _cache_get(key: str) -> Optional[dict]:
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < _CACHE_TTL:
            return entry[1]
        if entry:
            del _cache[key]
        return None


def _cache_set(key: str, value: dict) -> None:
    with _cache_lock:
        # Evict oldest entries if cache grows too large
        if len(_cache) >= 100:
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest_key]
        _cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# SSRF guard — only github.com is allowed
# ---------------------------------------------------------------------------
_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+((/tree/[A-Za-z0-9_.\-/]+)?)?$"
)


def _validate_github_url(url: str) -> None:
    url = url.strip()
    if not _GITHUB_URL_RE.match(url):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid GitHub URL. Expected format: "
                "https://github.com/owner/repo or https://github.com/owner/repo/tree/branch"
            ),
        )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class GitHubAnalyzeRequest(BaseModel):
    repo_url: str

    @field_validator("repo_url")
    @classmethod
    def strip_url(cls, v: str) -> str:
        return v.strip()


class FilesAnalyzeRequest(BaseModel):
    files: Dict[str, str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_root():
    index = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"message": "Tech Debt Dashboard API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": _provider,
        "llm_pass_enabled": _enable_llm,
        "rate_limit_rpm": _RATE_LIMIT_RPM,
        "cache_ttl_seconds": _CACHE_TTL,
        "cost": cost_tracker.get_stats(),
    }


@app.post("/analyze/github")
async def analyze_github(body: GitHubAnalyzeRequest, request: Request):
    """
    Analyze a public GitHub repository for technical debt.

    - Validates URL (SSRF protection: only github.com)
    - Rate-limited per IP (default 3 req/min)
    - Results cached for 1 hour
    """
    _validate_github_url(body.repo_url)

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_RATE_LIMIT_RPM} requests/minute per IP.",
        )

    # Cache check
    cached = _cache_get(body.repo_url)
    if cached:
        return JSONResponse(content={**cached, "cached": True})

    # Run analysis
    try:
        report = _generator.analyze_github(body.repo_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    _cache_set(body.repo_url, report)
    return JSONResponse(content={**report, "cached": False})


@app.post("/analyze/files")
async def analyze_files(body: FilesAnalyzeRequest):
    """
    Analyze a dict of {filename: content} for technical debt.
    Useful for testing without a GitHub URL.
    """
    if not body.files:
        raise HTTPException(status_code=422, detail="No files provided.")
    if len(body.files) > 200:
        raise HTTPException(status_code=422, detail="Too many files (max 200).")

    # Validate file content (size limits, path traversal, binary detection)
    validation_errors = validate_uploaded_files(body.files)
    if validation_errors:
        raise HTTPException(status_code=422, detail="; ".join(validation_errors))

    try:
        report = _generator.analyze_files(body.files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return JSONResponse(content=report)


@app.delete("/cache")
async def clear_cache():
    """Clear the analysis result cache."""
    with _cache_lock:
        count = len(_cache)
        _cache.clear()
    return {"cleared": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
