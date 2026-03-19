"""
Resilience module — retry with exponential backoff + circuit breaker.
Implements Day 5 curriculum resilience patterns.
"""
import time
import random
import logging
import threading
from typing import Callable, TypeVar

logger = logging.getLogger("multi_agent.resilience")

T = TypeVar("T")


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


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
        CLOSED  — normal, requests pass through
        OPEN    — failures exceeded threshold, requests blocked
        HALF_OPEN — testing if service recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failures = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state

    def call(self, func: Callable[[], T]) -> T:
        """Execute function through the circuit breaker."""
        current_state = self.state

        if current_state == self.OPEN:
            raise RuntimeError("Circuit breaker is OPEN — call blocked")

        try:
            result = func()
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def _on_success(self):
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED

    def _on_failure(self):
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = self.OPEN
                logger.warning(
                    "Circuit breaker OPEN after %d failures", self._failures
                )
