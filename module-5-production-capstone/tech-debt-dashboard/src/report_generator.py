"""
Report generator — orchestrates GitHub download, static analysis, scoring,
and optional LLM pass to produce a full tech-debt report dict.
"""
import io
import os
import re
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError

from src.debt_detector import StaticAnalyzer
from src.debt_scorer import DebtScorer, FileScore, LLMDebtAnalyzer, LLM_TRIGGER_THRESHOLD
from src.llm_client import LLMClient


# ---------------------------------------------------------------------------
# File-type constants (reused from Module 4)
# ---------------------------------------------------------------------------
DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c", ".cs",
    ".rb", ".php", ".swift", ".kt",
    ".sql", ".sh", ".bash",
}

IGNORED_DIRECTORIES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".vercel", "vendor",
    "coverage", ".pytest_cache", "target",
}


class ReportGenerator:
    """
    Full pipeline:
        1. Download GitHub repo (or accept pre-loaded files)
        2. Walk files, run StaticAnalyzer on each
        3. Score each file with DebtScorer
        4. Run LLM pass on files with score ≥ HIGH_THRESHOLD (cost-controlled)
        5. Assemble and return a structured report dict
    """

    def __init__(self, llm: LLMClient, enable_llm_pass: bool = True):
        self.analyzer = StaticAnalyzer()
        self.scorer = DebtScorer()
        self.llm_analyzer = LLMDebtAnalyzer(llm) if enable_llm_pass else None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def analyze_github(self, repo_url: str) -> dict:
        """Download a public GitHub repo and run the full analysis pipeline."""
        owner, repo, branch = _parse_github_url(repo_url)
        files = _download_github_archive(owner, repo, branch)
        return self._run_analysis(files, source=repo_url)

    def analyze_files(self, files: Dict[str, str]) -> dict:
        """Analyze a dict of {filename: content} directly."""
        return self._run_analysis(files, source="direct_upload")

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run_analysis(self, files: Dict[str, str], source: str) -> dict:
        scored_files: List[FileScore] = []

        for path, content in files.items():
            ext = os.path.splitext(path)[1].lower()
            if ext not in DEFAULT_EXTENSIONS:
                continue

            issues = self.analyzer.analyze(content, path)
            file_score = self.scorer.score(path, issues)

            # Store truncated source for the code viewer (max 500 lines)
            src_lines = content.splitlines()
            if len(src_lines) > 500:
                file_score.content = "\n".join(src_lines[:500]) + "\n\n# ... truncated ..."
            else:
                file_score.content = content

            # LLM pass only on files above the trigger threshold (default: medium+)
            if self.llm_analyzer and file_score.score >= LLM_TRIGGER_THRESHOLD:
                llm_issues = self.llm_analyzer.analyze(content, path)
                file_score = self.llm_analyzer.apply_llm_score(file_score, llm_issues)

            scored_files.append(file_score)

        # Sort descending by debt score
        scored_files.sort(key=lambda f: f.score, reverse=True)

        return _build_report(scored_files, source)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _build_report(scored_files: List[FileScore], source: str) -> dict:
    if not scored_files:
        return {
            "analysis_id": str(uuid.uuid4()),
            "source": source,
            "analyzed_at": _now_iso(),
            "summary": {
                "total_files": 0,
                "avg_debt_score": 0.0,
                "high_debt_files": 0,
                "medium_debt_files": 0,
                "low_debt_files": 0,
                "top_issue_types": [],
            },
            "files": [],
        }

    scores = [f.score for f in scored_files]
    avg_score = round(sum(scores) / len(scores), 2)

    high = sum(1 for f in scored_files if f.severity == "high")
    medium = sum(1 for f in scored_files if f.severity == "medium")
    low = sum(1 for f in scored_files if f.severity == "low")

    # Top issue type categories
    all_issue_types = [i.issue_type for f in scored_files for i in f.static_issues]
    top_types = [t for t, _ in Counter(all_issue_types).most_common(5)]

    return {
        "analysis_id": str(uuid.uuid4()),
        "source": source,
        "analyzed_at": _now_iso(),
        "summary": {
            "total_files": len(scored_files),
            "avg_debt_score": avg_score,
            "high_debt_files": high,
            "medium_debt_files": medium,
            "low_debt_files": low,
            "top_issue_types": top_types,
        },
        "files": [f.to_dict() for f in scored_files],
    }


# ---------------------------------------------------------------------------
# GitHub helpers (adapted from Module 4 main.py)
# ---------------------------------------------------------------------------

def _parse_github_url(url: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse a GitHub URL into (owner, repo, branch).
    Accepted forms:
        https://github.com/owner/repo
        https://github.com/owner/repo/tree/branch
    """
    url = url.strip().rstrip("/")
    pattern = re.compile(
        r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+))?"
    )
    m = pattern.match(url)
    if not m:
        raise ValueError(f"Invalid GitHub URL: {url!r}")
    owner, repo = m.group(1), m.group(2)
    branch = m.group(3)  # may be None
    return owner, repo, branch


def _download_github_archive(
    owner: str,
    repo: str,
    branch: Optional[str],
) -> Dict[str, str]:
    """
    Download a public GitHub repo archive and return {relative_path: content}.
    Tries the provided branch, then 'main', then 'master'.
    """
    branches_to_try = []
    if branch:
        branches_to_try.append(branch)
    if "main" not in branches_to_try:
        branches_to_try.append("main")
    if "master" not in branches_to_try:
        branches_to_try.append("master")

    last_error: Exception = RuntimeError("No branch attempted")

    for b in branches_to_try:
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{b}.zip"
        try:
            req = Request(url, headers={"User-Agent": "tech-debt-dashboard/1.0"})
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
            return _extract_zip(data, repo, b)
        except URLError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not download {owner}/{repo}. Tried branches: {branches_to_try}. "
        f"Last error: {last_error}"
    )


def _extract_zip(data: bytes, repo: str, branch: str) -> Dict[str, str]:
    """Extract a GitHub ZIP archive, returning {relative_path: text_content}."""
    files: Dict[str, str] = {}
    prefix = f"{repo}-{branch}/"  # GitHub adds this prefix inside the zip

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            # Strip the root prefix
            rel = name[len(prefix):] if name.startswith(prefix) else name
            if not rel or rel.endswith("/"):
                continue

            # Skip ignored directories
            parts = rel.split("/")
            if any(p in IGNORED_DIRECTORIES for p in parts[:-1]):
                continue

            ext = os.path.splitext(rel)[1].lower()
            if ext not in DEFAULT_EXTENSIONS:
                continue

            try:
                content = zf.read(name).decode("utf-8", errors="replace")
                files[rel] = content
            except Exception:
                continue  # Skip binary or unreadable files

    return files


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
