"""Structured logging and observability for the RAG system."""
import logging
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Configure root logger to emit JSON-like structured output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("rag_system")


@dataclass
class Span:
    """Timing span for a single pipeline stage."""
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, **meta) -> float:
        """Mark span as finished and return elapsed seconds."""
        self.end_time = time.time()
        self.metadata.update(meta)
        return self.duration_ms

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return round((end - self.start_time) * 1000, 1)


@dataclass
class RequestContext:
    """Context for a single RAG request."""
    query: str
    request_id: str = field(default_factory=lambda: uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    spans: List[Span] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round((time.time() - self.start_time) * 1000, 1)


class RAGLogger:
    """Structured logger for the RAG pipeline."""

    def __init__(self):
        self._context: Optional[RequestContext] = None

    def start_request(self, query: str) -> RequestContext:
        """Begin a new request context."""
        self._context = RequestContext(query=query)
        self._emit("request_start", {"query": query})
        return self._context

    def finish_request(self, n_sources: int = 0) -> None:
        """Log request completion."""
        if not self._context:
            return
        self._emit("request_finish", {
            "total_ms": self._context.total_ms,
            "n_sources": n_sources,
            "spans": [
                {"stage": s.name, "ms": s.duration_ms, **s.metadata}
                for s in self._context.spans
            ],
        })
        self._context = None

    @contextmanager
    def span(self, name: str, **initial_meta):
        """Context manager that times a pipeline stage."""
        s = Span(name=name, metadata=initial_meta)
        if self._context:
            self._context.spans.append(s)
        try:
            yield s
        finally:
            if s.end_time is None:
                s.finish()

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        request_id = self._context.request_id if self._context else "-"
        record = {"event": event, "request_id": request_id, **data}
        logger.info(json.dumps(record))


# Module-level singleton
rag_logger = RAGLogger()
