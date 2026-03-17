"""
Resilience module — retry with exponential backoff + circuit breaker.
Implements Day 5 curriculum resilience patterns.
"""
import time
import random
import logging
import threading
from typing import Callable, TypeVar

logger = logging.getLogger("tech_debt_dashboard.resilience")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> T:
    """
    Retry a function with exponential backoff.
    Raises the last exception if all retries fail.
    """
    last_error: Exception = RuntimeError("No attempt made")

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= 0.5 + random.random()
            logger.warning(
                "Attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)

    logger.error("All %d retries exhausted. Last error: %s", max_retries, last_error)
    raise last_error


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
        CLOSED  — normal, requests pass through
        OPEN    — failures exceeded threshold, requests fail fast
        HALF_OPEN — testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = "closed"
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = "half_open"
                    self._half_open_calls = 0
                    logger.info("Circuit breaker → half_open (testing recovery)")
            return self._state

    def execute(self, func: Callable[[], T], fallback: Callable[[], T] = None) -> T:
        """Execute func with circuit breaker protection."""
        current_state = self.state

        if current_state == "open":
            logger.warning("Circuit breaker OPEN — skipping call")
            if fallback:
                return fallback()
            raise RuntimeError("Circuit breaker is open — service unavailable")

        if current_state == "half_open":
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    if fallback:
                        return fallback()
                    raise RuntimeError("Circuit breaker half-open limit reached")
                self._half_open_calls += 1

        try:
            result = func()
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure()
            if fallback:
                logger.warning("Primary failed, using fallback: %s", exc)
                return fallback()
            raise

    def _record_success(self):
        with self._lock:
            if self._state == "half_open":
                self._state = "closed"
                logger.info("Circuit breaker → closed (service recovered)")
            self._failure_count = 0

    def _record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.error(
                    "Circuit breaker → OPEN after %d failures", self._failure_count
                )

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
        }
