"""Quick integration tests — run while the server is on port 8001."""
import requests
import json
import time

BASE = "http://localhost:8001"
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name} {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")

# 1. Health
r = requests.get(f"{BASE}/health")
check("Health endpoint", r.status_code == 200, f"status={r.json()['status']}")

# 2. Dashboard serves HTML
r = requests.get(f"{BASE}/")
check("Dashboard HTML", r.status_code == 200 and "text/html" in r.headers.get("content-type", ""))

# 3. Clean file → low score, no LLM issues
r = requests.post(f"{BASE}/analyze/files", json={"files": {
    "clean.py": 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
}})
data = r.json()
f0 = data["files"][0]
check("Clean file low score", f0["debt_score"] < 4 and f0["severity"] == "low",
      f"score={f0['debt_score']}")

# 4. Debt-heavy file → medium+ score, LLM triggered
bad_code = (
    "# TODO: fix this hack\n"
    "# FIXME: memory leak\n"
    "# HACK: temporary workaround\n"
    "# XXX: needs refactoring\n"
    "def process_data(x):\n"
    "    if x:\n"
    "        if x > 0:\n"
    "            if x > 10:\n"
    "                if x > 100:\n"
    "                    if x > 1000:\n"
    "                        return x * 42 + 7 * 99\n"
    "    return None\n"
)
r = requests.post(f"{BASE}/analyze/files", json={"files": {"bad.py": bad_code}})
data = r.json()
f0 = data["files"][0]
check("Debt-heavy file scored", f0["debt_score"] >= 4.0,
      f"score={f0['debt_score']}, severity={f0['severity']}")
check("Static issues found", len(f0["issues"]) > 0,
      f"count={len(f0['issues'])}")
check("LLM issues returned", len(f0["llm_issues"]) > 0,
      f"count={len(f0['llm_issues'])}")

# 5. SSRF protection
r = requests.post(f"{BASE}/analyze/github", json={"repo_url": "https://evil.com/owner/repo"})
check("SSRF blocked", r.status_code == 422, f"status={r.status_code}")

# 6. Empty files rejected
r = requests.post(f"{BASE}/analyze/files", json={"files": {}})
check("Empty files rejected", r.status_code == 422, f"status={r.status_code}")

# 7. Cache clear
r = requests.delete(f"{BASE}/cache")
check("Cache clear", r.status_code == 200, f"response={r.json()}")

# 8. Rate limiting (hit the github endpoint 4 times quickly — 4th should 429)
# First clear cache
requests.delete(f"{BASE}/cache")
statuses = []
for i in range(4):
    try:
        r = requests.post(f"{BASE}/analyze/github",
                          json={"repo_url": "https://github.com/sindresorhus/is"},
                          timeout=60)
        statuses.append(r.status_code)
    except requests.exceptions.Timeout:
        statuses.append("timeout")
    if r.status_code == 429:
        break
check("Rate limit kicks in", 429 in statuses, f"statuses={statuses}")

# 9. Cost tracking appears in health
r = requests.get(f"{BASE}/health")
data = r.json()
check("Cost stats in health", "cost" in data and "total_calls" in data["cost"],
      f"calls={data['cost']['total_calls']}")

# 10. Input validation (path traversal in filename)
r = requests.post(f"{BASE}/analyze/files", json={"files": {
    "../../etc/passwd": "root:x:0:0:root:/root:/bin/bash"
}})
check("Path traversal blocked", r.status_code == 422, f"status={r.status_code}")

# 11. Input validation (oversized file)
r = requests.post(f"{BASE}/analyze/files", json={"files": {
    "big.py": "x = 1\n" * 200_000  # ~1.2 MB, over the 500KB limit
}})
check("Oversized file blocked", r.status_code == 422, f"status={r.status_code}")

# 12. Code content returned for code viewer
r = requests.post(f"{BASE}/analyze/files", json={"files": {
    "test.py": 'def hello():\n    """Greet."""\n    print("hi")\n'
}})
data = r.json()
f0 = data["files"][0]
check("Content in response", bool(f0.get("content")), f"has_content={bool(f0.get('content'))}")

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
