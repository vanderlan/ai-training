"""
Multi-Agent Orchestrator — FastAPI application
Production patterns applied:
  • Per-IP rate limiting (token bucket, 5 req/min on /run)
  • Response cache with 1-hour TTL
  • Graceful DeepSeek LLM fallback via MockClient
  • Prompt injection defense (input validation)
  • Retry with exponential backoff + circuit breaker
  • Cost tracking with daily budget cap
  • Output validation (sensitive data redaction)
  • Structured logging
"""
import logging
import os
import hashlib
import time
import threading
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from src.llm_client import get_llm_client
from src.supervisor import SupervisorAgent
from src.security import check_prompt_injection
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
logger = logging.getLogger("multi_agent")

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multi-Agent Orchestrator",
    description="Multi-agent research assistant with supervisor pattern using DeepSeek.",
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
# LLM + Supervisor setup
# ---------------------------------------------------------------------------
_provider = os.getenv("LLM_PROVIDER", "deepseek").strip()
_llm = get_llm_client(_provider)

# ---------------------------------------------------------------------------
# Rate limiter (token bucket, per IP)
# ---------------------------------------------------------------------------
_RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "5"))


class _TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate
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
# Response cache (task hash match, 1-hour TTL)
# ---------------------------------------------------------------------------
_CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _cache_key(task: str, max_iterations: int) -> str:
    raw = f"{task.strip().lower()}::{max_iterations}"
    return hashlib.sha256(raw.encode()).hexdigest()


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
        if len(_cache) >= 100:
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest_key]
        _cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class TaskRequest(BaseModel):
    task: str
    max_iterations: int = 5

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Task cannot be empty")
        if len(v) > 5000:
            raise ValueError("Task too long (max 5000 characters)")
        return v

    @field_validator("max_iterations")
    @classmethod
    def validate_iterations(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("max_iterations must be between 1 and 10")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_root():
    index = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"message": "Multi-Agent Orchestrator API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": _provider,
        "rate_limit_rpm": _RATE_LIMIT_RPM,
        "cache_ttl_seconds": _CACHE_TTL,
        "cost": cost_tracker.get_stats(),
    }


@app.post("/run")
async def run_task(body: TaskRequest, request: Request):
    """
    Run a multi-agent research task.
    The supervisor coordinates Researcher, Writer, and Reviewer agents.
    """
    # Prompt injection check
    is_suspicious, patterns = check_prompt_injection(body.task)
    if is_suspicious:
        raise HTTPException(
            status_code=422,
            detail="Input rejected — suspicious content detected.",
        )

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_RATE_LIMIT_RPM} requests/minute per IP.",
        )

    # Cache check
    key = _cache_key(body.task, body.max_iterations)
    cached = _cache_get(key)
    if cached:
        return JSONResponse(content={**cached, "cached": True})

    # Run multi-agent workflow
    try:
        supervisor = SupervisorAgent(_llm)
        report = supervisor.run(body.task, body.max_iterations)
    except Exception as exc:
        logger.exception("Multi-agent workflow failed")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}")

    _cache_set(key, report)
    return JSONResponse(content={**report, "cached": False})


@app.post("/run/stream")
async def run_task_stream(body: TaskRequest, request: Request):
    """
    Run a multi-agent task with Server-Sent Events streaming.
    Each step is emitted in real-time so the frontend can visualize progress.
    """
    # Prompt injection check
    is_suspicious, patterns = check_prompt_injection(body.task)
    if is_suspicious:
        raise HTTPException(
            status_code=422,
            detail="Input rejected — suspicious content detected.",
        )

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_RATE_LIMIT_RPM} requests/minute per IP.",
        )

    def event_generator():
        supervisor = SupervisorAgent(_llm)
        yield from supervisor.run_streaming(body.task, body.max_iterations)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/cache")
async def clear_cache():
    """Clear the response cache."""
    with _cache_lock:
        count = len(_cache)
        _cache.clear()
    return {"cleared": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
