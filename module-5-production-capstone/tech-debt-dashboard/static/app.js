/**
 * Tech Debt Dashboard — app.js
 * Handles: form submission, API calls, chart rendering, table + filtering, modal
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const form         = document.getElementById("analyze-form");
const repoInput    = document.getElementById("repo-url");
const submitBtn    = document.getElementById("submit-btn");
const errorMsg     = document.getElementById("error-msg");
const loadingEl    = document.getElementById("loading");
const loadingText  = document.getElementById("loading-text");
const resultsEl    = document.getElementById("results");

// Summary
const sTotal    = document.getElementById("s-total");
const sAvg      = document.getElementById("s-avg");
const sHigh     = document.getElementById("s-high");
const sMedium   = document.getElementById("s-medium");
const sLow      = document.getElementById("s-low");
const sCached   = document.getElementById("s-cached");

// Top issues
const topIssuesTags = document.getElementById("top-issues-tags");
const topIssuesCard = document.getElementById("top-issues-card");

// Chart
const chartCanvas  = document.getElementById("debt-chart");
let chartInstance  = null;

// Table
const tbody          = document.getElementById("files-tbody");
const severityFilter = document.getElementById("severity-filter");
const searchInput    = document.getElementById("search-input");

// Code viewer modal
const modalOverlay    = document.getElementById("modal-overlay");
const modalClose      = document.getElementById("modal-close");
const modalPath       = document.getElementById("modal-path");
const modalScoreBadge = document.getElementById("modal-score-badge");
const modalSevBadge   = document.getElementById("modal-severity-badge");
const modalIssueCount = document.getElementById("modal-issue-count");
const codeViewerEl    = document.getElementById("code-viewer");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let allFiles = [];
let sortCol  = "debt_score";
let sortAsc  = false;

// ---------------------------------------------------------------------------
// Form submit
// ---------------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = repoInput.value.trim();
  if (!url) return;

  showError("");
  setLoading(true, "Cloning repository and analyzing…");
  resultsEl.classList.add("hidden");

  try {
    const res = await fetch("/analyze/github", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: url }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `Error ${res.status}`);
      return;
    }

    renderReport(data);
  } catch (err) {
    showError("Network error — is the server running?");
  } finally {
    setLoading(false);
  }
});

// ---------------------------------------------------------------------------
// Render full report
// ---------------------------------------------------------------------------
function renderReport(data) {
  const s = data.summary;

  // Summary cards
  sTotal.textContent  = s.total_files;
  sAvg.textContent    = s.avg_debt_score.toFixed(1);
  sHigh.textContent   = s.high_debt_files;
  sMedium.textContent = s.medium_debt_files;
  sLow.textContent    = s.low_debt_files;
  sCached.textContent = data.cached ? "✓ Cached" : "Fresh";

  // Top issue types
  topIssuesTags.innerHTML = "";
  if (s.top_issue_types && s.top_issue_types.length) {
    s.top_issue_types.forEach(t => {
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = formatIssueType(t);
      topIssuesTags.appendChild(span);
    });
    topIssuesCard.classList.remove("hidden");
  } else {
    topIssuesCard.classList.add("hidden");
  }

  // Store files for table + sorting
  allFiles = data.files || [];

  // Chart (top 20 by score)
  renderChart(allFiles.slice(0, 20));

  // Table
  renderTable();

  resultsEl.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Chart (horizontal bar, top 20 files)
// ---------------------------------------------------------------------------
function renderChart(files) {
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const labels    = files.map(f => shortPath(f.path));
  const scores    = files.map(f => f.debt_score);
  const colors    = files.map(f => severityColor(f.severity));
  const borders   = files.map(f => severityColor(f.severity, 1));

  // Adjust chart height based on number of bars
  const barHeight = 28;
  chartCanvas.height = Math.max(200, files.length * barHeight);

  chartInstance = new Chart(chartCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Debt Score",
        data: scores,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => files[items[0].dataIndex].path,
            label: (item) => ` Score: ${item.raw.toFixed(2)}  (${files[item.dataIndex].severity})`,
          },
        },
      },
      scales: {
        x: {
          min: 0, max: 10,
          ticks: { color: "#8892a4" },
          grid: { color: "rgba(255,255,255,.05)" },
        },
        y: {
          ticks: {
            color: "#8892a4",
            font: { family: "'Fira Code', monospace", size: 11 },
          },
          grid: { display: false },
        },
      },
      onClick: (_evt, elements) => {
        if (elements.length) openModal(allFiles[elements[0].index]);
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Table rendering + sorting + filtering
// ---------------------------------------------------------------------------
function renderTable() {
  const filterVal  = severityFilter.value;
  const searchVal  = searchInput.value.toLowerCase();

  let filtered = allFiles.filter(f => {
    const matchSev    = filterVal === "all" || f.severity === filterVal;
    const matchSearch = !searchVal || f.path.toLowerCase().includes(searchVal);
    return matchSev && matchSearch;
  });

  // Sort
  filtered.sort((a, b) => {
    let av = a[sortCol], bv = b[sortCol];
    if (typeof av === "string") av = av.toLowerCase();
    if (typeof bv === "string") bv = bv.toLowerCase();
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  });

  tbody.innerHTML = "";
  filtered.forEach(f => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="path-cell" title="${escHtml(f.path)}">${escHtml(f.path)}</td>
      <td class="score-cell" style="color:${severityColor(f.severity, .8)}">${f.debt_score.toFixed(1)}</td>
      <td><span class="badge ${f.severity}">${f.severity}</span></td>
      <td>${f.issues.length + (f.llm_issues ? f.llm_issues.length : 0)} issues</td>
    `;
    tr.addEventListener("click", () => openModal(f));
    tbody.appendChild(tr);
  });

  if (filtered.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" style="text-align:center;color:var(--text-muted);padding:2rem">No files match the current filter.</td>`;
    tbody.appendChild(tr);
  }
}

// Sorting
document.querySelectorAll("th.sortable").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (sortCol === col) {
      sortAsc = !sortAsc;
    } else {
      sortCol = col;
      sortAsc = col === "path";
    }
    renderTable();
  });
});

severityFilter.addEventListener("change", renderTable);
searchInput.addEventListener("input", renderTable);

// ---------------------------------------------------------------------------
// Issue Modal
// ---------------------------------------------------------------------------
function openModal(file) {
  // Header
  modalPath.textContent = file.path;
  modalScoreBadge.textContent = `Score: ${file.debt_score.toFixed(1)}`;
  modalScoreBadge.className = `badge ${file.severity}`;
  modalSevBadge.textContent = file.severity;
  modalSevBadge.className = `badge ${file.severity}`;

  const staticCount = (file.issues || []).length;
  const llmCount = (file.llm_issues || []).length;
  modalIssueCount.textContent = `${staticCount + llmCount} issue${staticCount + llmCount !== 1 ? "s" : ""}`;

  // Build issue map: lineNum → [issues]
  const issueMap = {};
  (file.issues || []).forEach(i => {
    const line = i.line || 0;
    if (!issueMap[line]) issueMap[line] = [];
    issueMap[line].push({ ...i, source: "static" });
  });
  (file.llm_issues || []).forEach(i => {
    const line = i.line || 0;
    if (!issueMap[line]) issueMap[line] = [];
    issueMap[line].push({ ...i, source: "llm" });
  });

  codeViewerEl.innerHTML = "";

  // File-level issues (line 0)
  if (issueMap[0]) {
    issueMap[0].forEach(issue => codeViewerEl.appendChild(createAnnotation(issue)));
  }

  if (file.content) {
    const lines = file.content.split("\n");
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      const lineIssues = issueMap[lineNum];

      // Code line
      const lineEl = document.createElement("div");
      if (lineIssues) {
        const maxSev = Math.max(...lineIssues.map(i => i.severity || 1));
        lineEl.className = `code-line ${maxSev >= 3 ? "issue-high" : maxSev >= 2 ? "issue-medium" : "issue-low"}`;
      } else {
        lineEl.className = "code-line";
      }

      const numEl = document.createElement("span");
      numEl.className = "line-num";
      numEl.textContent = lineNum;

      const codeEl = document.createElement("pre");
      codeEl.className = "line-code";
      codeEl.textContent = line;

      lineEl.appendChild(numEl);
      lineEl.appendChild(codeEl);
      codeViewerEl.appendChild(lineEl);

      // Issue annotations after the affected line
      if (lineIssues) {
        lineIssues.forEach(issue => codeViewerEl.appendChild(createAnnotation(issue)));
      }
    });
  } else {
    // Fallback when no source code is available
    const fb = document.createElement("div");
    fb.className = "no-content-fallback";
    fb.innerHTML = "<p>Code preview not available.</p>";
    codeViewerEl.appendChild(fb);
    Object.values(issueMap).flat().forEach(issue => codeViewerEl.appendChild(createAnnotation(issue)));
  }

  modalOverlay.classList.remove("hidden");

  // Auto-scroll to first issue
  const firstAnnotation = codeViewerEl.querySelector(".issue-annotation");
  if (firstAnnotation) {
    setTimeout(() => firstAnnotation.scrollIntoView({ behavior: "smooth", block: "center" }), 150);
  }
}

function createAnnotation(issue) {
  const el = document.createElement("div");
  const sevClass = (issue.severity || 1) >= 3 ? "high" : (issue.severity || 1) >= 2 ? "medium" : "low";
  const isLLM = issue.source === "llm";
  el.className = `issue-annotation ${sevClass}`;

  const icon = document.createElement("span");
  icon.className = "annotation-icon";
  icon.textContent = isLLM ? "\uD83E\uDD16" : "\u26A0\uFE0F";

  const typeBadge = document.createElement("span");
  typeBadge.className = "issue-type-badge";
  typeBadge.textContent = formatIssueType(issue.type);

  const desc = document.createElement("span");
  desc.className = "issue-description";
  desc.textContent = issue.description;

  el.appendChild(icon);
  el.appendChild(typeBadge);
  el.appendChild(desc);
  return el;
}

modalClose.addEventListener("click", () => modalOverlay.classList.add("hidden"));
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) modalOverlay.classList.add("hidden");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") modalOverlay.classList.add("hidden");
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setLoading(on, msg = "") {
  submitBtn.disabled = on;
  if (on) {
    loadingEl.classList.remove("hidden");
    loadingText.textContent = msg;
  } else {
    loadingEl.classList.add("hidden");
  }
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.toggle("hidden", !msg);
}

function severityColor(sev, alpha = 0.7) {
  const map = {
    high:   `rgba(239,68,68,${alpha})`,
    medium: `rgba(245,158,11,${alpha})`,
    low:    `rgba(34,197,94,${alpha})`,
  };
  return map[sev] || `rgba(99,102,241,${alpha})`;
}

function shortPath(p) {
  return p.length > 45 ? "…" + p.slice(-44) : p;
}

function formatIssueType(t) {
  return (t || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
