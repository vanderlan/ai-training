"""Static analysis debt detector — regex-based, zero LLM cost."""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class DebtIssue:
    """A single detected tech-debt instance."""
    issue_type: str          # e.g. "todo_comment", "long_function"
    description: str
    line: int = 0            # 1-based line number (0 = whole-file issue)
    severity: int = 1        # 1 = low, 2 = medium, 3 = high
    extra: dict = field(default_factory=dict)


class StaticAnalyzer:
    """
    Pass-1 debt detector.  Runs entirely on raw source text — no LLM needed.

    Detectors:
        - TODO/FIXME/HACK/XXX comments
        - Long functions  (>50 lines)
        - Deep nesting    (≥4 indent levels)
        - Missing docstrings (Python functions/classes)
        - Magic numbers / bare literals
        - Large files     (>300 lines)
    """

    # Comment markers that indicate deferred work
    DEBT_COMMENT_RE = re.compile(
        r"#.*?\b(TODO|FIXME|HACK|XXX|REFACTOR|TEMP|DEPRECATED)\b"
        r"|//.*?\b(TODO|FIXME|HACK|XXX|REFACTOR|TEMP|DEPRECATED)\b"
        r"|/\*.*?\b(TODO|FIXME|HACK|XXX|REFACTOR|TEMP|DEPRECATED)\b",
        re.IGNORECASE,
    )

    # Python def/class opening lines
    PY_FUNC_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
    PY_CLASS_RE = re.compile(r"^(\s*)class\s+(\w+)", re.MULTILINE)

    # Magic number: a bare integer/float literal NOT in an assignment context
    MAGIC_NUM_RE = re.compile(
        r"(?<![=\[\(,\s\"])-?\b(\d{2,}(?:\.\d+)?)\b(?!\s*[=\w])",
    )

    def analyze(self, content: str, filename: str) -> List[DebtIssue]:
        """Run all static detectors on *content* and return a list of issues."""
        issues: List[DebtIssue] = []
        lines = content.splitlines()
        total_lines = len(lines)

        issues.extend(self._detect_debt_comments(lines))
        issues.extend(self._detect_large_file(total_lines))
        issues.extend(self._detect_deep_nesting(lines))
        issues.extend(self._detect_magic_numbers(lines, filename))

        lang = self._ext_to_lang(filename)
        if lang == "python":
            issues.extend(self._detect_long_python_functions(content))
            issues.extend(self._detect_missing_docstrings(content))
        elif lang in ("javascript", "typescript"):
            issues.extend(self._detect_long_js_functions(content))

        return issues

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _detect_debt_comments(self, lines: List[str]) -> List[DebtIssue]:
        issues = []
        for i, line in enumerate(lines, start=1):
            m = self.DEBT_COMMENT_RE.search(line)
            if m:
                keyword = next(g for g in m.groups() if g)
                issues.append(DebtIssue(
                    issue_type="debt_comment",
                    description=f"{keyword} comment found",
                    line=i,
                    severity=2,
                    extra={"keyword": keyword.upper(), "text": line.strip()[:120]},
                ))
        return issues

    def _detect_large_file(self, total_lines: int) -> List[DebtIssue]:
        if total_lines > 300:
            severity = 3 if total_lines > 600 else 2
            return [DebtIssue(
                issue_type="large_file",
                description=f"File has {total_lines} lines (>{300} threshold)",
                line=0,
                severity=severity,
                extra={"line_count": total_lines},
            )]
        return []

    def _detect_deep_nesting(self, lines: List[str]) -> List[DebtIssue]:
        """Flag lines with 4+ levels of indentation (4 spaces per level)."""
        issues = []
        threshold = 16  # 4 levels × 4 spaces
        for i, line in enumerate(lines, start=1):
            if line.strip() == "":
                continue
            indent = len(line) - len(line.lstrip())
            if indent >= threshold:
                issues.append(DebtIssue(
                    issue_type="deep_nesting",
                    description=f"Indentation depth ≥4 levels (indent={indent})",
                    line=i,
                    severity=2,
                    extra={"indent": indent},
                ))
        return issues

    def _detect_magic_numbers(self, lines: List[str], filename: str) -> List[DebtIssue]:
        """Detect bare numeric literals (skip common safe values: 0, 1, 2)."""
        lang = self._ext_to_lang(filename)
        safe = {"0", "1", "2", "100", "1000"}
        issues = []
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "/*")):
                continue
            for m in self.MAGIC_NUM_RE.finditer(line):
                val = m.group(1)
                if val not in safe:
                    issues.append(DebtIssue(
                        issue_type="magic_number",
                        description=f"Magic number {val} — consider a named constant",
                        line=i,
                        severity=1,
                        extra={"value": val},
                    ))
        return issues

    def _detect_long_python_functions(self, content: str) -> List[DebtIssue]:
        """Detect Python functions/methods longer than 50 lines."""
        issues = []
        lines = content.splitlines()

        for match in self.PY_FUNC_RE.finditer(content):
            indent = len(match.group(1))
            name = match.group(2)
            start_line = content[: match.start()].count("\n") + 1

            # Count lines until we return to same or lower indent level
            body_length = 0
            for line in lines[start_line:]:
                if line.strip() == "":
                    body_length += 1
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent and body_length > 0:
                    break
                body_length += 1

            if body_length > 50:
                severity = 3 if body_length > 100 else 2
                issues.append(DebtIssue(
                    issue_type="long_function",
                    description=f"Function '{name}' is {body_length} lines (>50 threshold)",
                    line=start_line,
                    severity=severity,
                    extra={"name": name, "line_count": body_length},
                ))
        return issues

    def _detect_long_js_functions(self, content: str) -> List[DebtIssue]:
        """Detect JS/TS functions longer than 50 lines (brace-counting approach)."""
        issues = []
        lines = content.splitlines()
        func_re = re.compile(
            r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>|(\w+)\s*\(.*?\)\s*\{)",
        )
        for i, line in enumerate(lines):
            m = func_re.search(line)
            if not m:
                continue
            name = m.group(1) or m.group(2) or m.group(3) or "anonymous"
            depth = 0
            length = 0
            for j in range(i, min(i + 500, len(lines))):
                depth += lines[j].count("{") - lines[j].count("}")
                length += 1
                if depth <= 0 and length > 2:
                    break
            if length > 50:
                severity = 3 if length > 100 else 2
                issues.append(DebtIssue(
                    issue_type="long_function",
                    description=f"Function '{name}' is ~{length} lines (>50 threshold)",
                    line=i + 1,
                    severity=severity,
                    extra={"name": name, "line_count": length},
                ))
        return issues

    def _detect_missing_docstrings(self, content: str) -> List[DebtIssue]:
        """Flag Python functions and classes that lack a docstring."""
        issues = []
        lines = content.splitlines()

        # Find all def/class lines
        for match in list(self.PY_FUNC_RE.finditer(content)) + list(self.PY_CLASS_RE.finditer(content)):
            start_line = content[: match.start()].count("\n") + 1  # 1-based
            # The line after the signature (may span multiple lines with parentheses)
            # Look for triple-quote on the next non-empty line inside the body
            found_docstring = False
            for line in lines[start_line:start_line + 5]:
                stripped = line.strip()
                if stripped.startswith(('"""', "'''", 'r"""', "r'''")):
                    found_docstring = True
                    break
                if stripped and not stripped.startswith("#"):
                    break  # Non-comment, non-docstring → no docstring

            if not found_docstring:
                name = match.group(2)
                issues.append(DebtIssue(
                    issue_type="missing_docstring",
                    description=f"'{name}' has no docstring",
                    line=start_line,
                    severity=1,
                    extra={"name": name},
                ))
        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ext_to_lang(filename: str) -> str:
        for ext, lang in [
            (".py", "python"), (".js", "javascript"), (".ts", "typescript"),
            (".jsx", "javascript"), (".tsx", "typescript"),
        ]:
            if filename.endswith(ext):
                return lang
        return "unknown"
