"""
Debt scorer — converts raw DebtIssue lists into a 0-10 score per file,
then optionally runs a targeted LLM pass on high-debt files.
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.debt_detector import DebtIssue
from src.llm_client import LLMClient
from src.security import sanitize_for_llm, validate_llm_output
from src.resilience import retry_with_backoff, CircuitBreaker
from src.cost_tracker import cost_tracker

logger = logging.getLogger("tech_debt_dashboard.scorer")


# ---------------------------------------------------------------------------
# Score thresholds
# ---------------------------------------------------------------------------
LOW_THRESHOLD = 4.0
HIGH_THRESHOLD = 7.0
# Threshold at which the LLM pass is triggered (default: medium+ files)
LLM_TRIGGER_THRESHOLD = float(os.getenv("LLM_TRIGGER_THRESHOLD", "4.0"))


@dataclass
class FileScore:
    """Weighted debt score for a single file."""
    path: str
    score: float                              # 0.0 – 10.0
    severity: str                             # "low" | "medium" | "high"
    static_issues: List[DebtIssue] = field(default_factory=list)
    llm_issues: List[dict] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)  # category → partial score
    content: str = ""                               # source code for code viewer

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "debt_score": round(self.score, 2),
            "severity": self.severity,
            "breakdown": self.breakdown,
            "issues": [
                {
                    "type": i.issue_type,
                    "description": i.description,
                    "line": i.line,
                    "severity": i.severity,
                    **i.extra,
                }
                for i in self.static_issues
            ],
            "llm_issues": self.llm_issues,
            "content": self.content,
        }


class DebtScorer:
    """
    Scoring weights (max 2 pts each, total max = 10):

        comments_score   = debt comments      (TODO/FIXME etc.)   → 0-2
        complexity_score = long functions     (>50 lines)         → 0-2
        nesting_score    = deep nesting       (≥4 levels)         → 0-2
        docs_score       = missing docstrings                     → 0-2
        structure_score  = large file + magic numbers             → 0-2
        llm_score        = LLM findings (only if static ≥ 4.0)   → 0-2
    """

    # Weight caps per category
    CAPS = {
        "comments":  2.0,
        "complexity": 2.0,
        "nesting":   2.0,
        "docs":      2.0,
        "structure": 2.0,
    }

    def score(self, path: str, issues: List[DebtIssue]) -> FileScore:
        """Compute a 0-10 debt score from static issues only."""
        breakdown = self._compute_breakdown(issues)
        static_score = sum(breakdown.values())
        severity = self._severity(static_score)

        return FileScore(
            path=path,
            score=static_score,
            severity=severity,
            static_issues=issues,
            breakdown=breakdown,
        )

    def _compute_breakdown(self, issues: List[DebtIssue]) -> dict:
        counts = {k: 0.0 for k in self.CAPS}

        for issue in issues:
            t = issue.issue_type
            w = issue.severity  # 1-3

            if t == "debt_comment":
                counts["comments"] += 0.4 * w
            elif t == "long_function":
                counts["complexity"] += 0.8 * w
            elif t == "deep_nesting":
                counts["nesting"] += 0.5 * w
            elif t == "missing_docstring":
                counts["docs"] += 0.3 * w
            elif t in ("large_file", "magic_number"):
                counts["structure"] += 0.4 * w

        # Cap each category
        return {k: round(min(v, self.CAPS[k]), 2) for k, v in counts.items()}

    @staticmethod
    def _severity(score: float) -> str:
        if score >= HIGH_THRESHOLD:
            return "high"
        if score >= LOW_THRESHOLD:
            return "medium"
        return "low"


class LLMDebtAnalyzer:
    """
    Pass-2 LLM analysis — called only on files whose static score ≥ 4.0
    to keep LLM costs under control.
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineer performing a technical debt review. "
        "Analyze the provided code and identify the top 3 most significant tech-debt issues. "
        "Return ONLY a valid JSON array with objects: "
        '[{"type": "string", "description": "string", "severity": 1-3, "line": int_or_0}]. '
        "severity: 1=minor, 2=moderate, 3=critical. "
        "Focus on: error handling gaps, architectural issues, coupling, "
        "security smells, deprecated patterns. Be concise."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=120.0
        )

    def analyze(self, content: str, filename: str) -> List[dict]:
        """Ask the LLM for top-3 debt issues. Returns empty list on any error."""
        # Budget check
        if not cost_tracker.check_budget():
            logger.warning("Skipping LLM for %s — daily budget exhausted", filename)
            return []

        # Truncate very large files to stay within context limits
        truncated = content[:6000]
        if len(content) > 6000:
            truncated += "\n\n... [truncated for analysis]"

        # Prompt isolation — protect against injection from source code
        isolated_code = sanitize_for_llm(truncated, filename)

        prompt = (
            f"{isolated_code}\n\n"
            "Return the JSON array now."
        )

        input_text = self.SYSTEM_PROMPT + prompt

        def _call_llm():
            return self.llm.chat([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])

        try:
            raw = self._circuit_breaker.execute(
                func=lambda: retry_with_backoff(_call_llm, max_retries=2, base_delay=1.0),
                fallback=lambda: "",
            )
            if not raw:
                return []

            # Track cost
            cost_tracker.record_call(input_text, raw)

            # Output validation — redact any sensitive data in LLM response
            raw = validate_llm_output(raw)

            issues = self._parse_response(raw)
            return issues[:3]  # Hard cap at 3
        except Exception as exc:
            logger.error("LLM analysis failed for %s: %s", filename, exc)
            return []

    def _parse_response(self, raw: str) -> List[dict]:
        """Extract JSON array from LLM response, ignoring surrounding text."""
        # Find JSON array in the response
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
            if not isinstance(data, list):
                return []
            result = []
            for item in data:
                if isinstance(item, dict) and "description" in item:
                    result.append({
                        "type": str(item.get("type", "llm_detected")),
                        "description": str(item.get("description", ""))[:300],
                        "severity": int(item.get("severity", 1)),
                        "line": int(item.get("line", 0)),
                    })
            return result
        except (json.JSONDecodeError, ValueError):
            return []

    def apply_llm_score(self, file_score: FileScore, llm_issues: List[dict]) -> FileScore:
        """Merge LLM findings into the FileScore, adding up to 2 extra points."""
        if not llm_issues:
            return file_score

        # Up to 2 extra points based on severity of LLM findings
        llm_pts = sum(min(i.get("severity", 1), 3) * 0.4 for i in llm_issues)
        llm_pts = min(llm_pts, 2.0)

        new_score = min(file_score.score + llm_pts, 10.0)
        file_score.score = round(new_score, 2)
        file_score.severity = DebtScorer._severity(new_score)
        file_score.llm_issues = llm_issues
        file_score.breakdown["llm"] = round(llm_pts, 2)
        return file_score
