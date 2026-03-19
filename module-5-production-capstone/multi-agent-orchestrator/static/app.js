/**
 * Multi-Agent Orchestrator — app.js
 * Handles: form submission, API calls, timeline rendering, result display,
 *          SSE streaming, live agent visualization panel
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const form        = document.getElementById("task-form");
const taskInput   = document.getElementById("task-input");
const iterSelect  = document.getElementById("iterations-select");
const submitBtn   = document.getElementById("submit-btn");
const errorMsg    = document.getElementById("error-msg");
const loadingEl   = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const resultsEl   = document.getElementById("results");

// Summary
const sSteps  = document.getElementById("s-steps");
const sAgents = document.getElementById("s-agents");
const sCached = document.getElementById("s-cached");

// Note
const noteBanner = document.getElementById("note-banner");

// Timeline
const timelineEl = document.getElementById("timeline");

// Result
const resultContent = document.getElementById("result-content");

// Live panel
const liveToggle = document.getElementById("live-toggle");
const livePanel  = document.getElementById("live-panel");
const liveFeed   = document.getElementById("live-feed");
const liveDot    = document.querySelector(".live-dot");

// Agent nodes
const nodes = {
  supervisor: document.getElementById("node-supervisor"),
  researcher: document.getElementById("node-researcher"),
  writer:     document.getElementById("node-writer"),
  reviewer:   document.getElementById("node-reviewer"),
};
const statuses = {
  supervisor: document.getElementById("status-supervisor"),
  researcher: document.getElementById("status-researcher"),
  writer:     document.getElementById("status-writer"),
  reviewer:   document.getElementById("status-reviewer"),
};
const connectors = {
  researcher: document.getElementById("conn-researcher"),
  writer:     document.getElementById("conn-writer"),
  reviewer:   document.getElementById("conn-reviewer"),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let liveEnabled = false;
let startTime   = null;

// ---------------------------------------------------------------------------
// Toggle live view
// ---------------------------------------------------------------------------
liveToggle.addEventListener("change", () => {
  liveEnabled = liveToggle.checked;
  livePanel.classList.toggle("hidden", !liveEnabled);
  document.body.classList.toggle("live-open", liveEnabled);
});

// ---------------------------------------------------------------------------
// Form submit
// ---------------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const task = taskInput.value.trim();
  if (!task) return;

  showError("");
  setLoading(true);
  resultsEl.classList.add("hidden");
  startTime = Date.now();

  if (liveEnabled) {
    resetLivePanel();
    runStreaming(task, parseInt(iterSelect.value, 10));
  } else {
    runStandard(task, parseInt(iterSelect.value, 10));
  }
});

// ---------------------------------------------------------------------------
// Standard (non-streaming) request
// ---------------------------------------------------------------------------
async function runStandard(task, maxIter) {
  try {
    const res = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, max_iterations: maxIter }),
    });
    const data = await res.json();
    if (!res.ok) { showError(data.detail || "Error " + res.status); return; }
    renderReport(data);
  } catch (err) {
    showError("Network error — is the server running?");
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Streaming (SSE) request
// ---------------------------------------------------------------------------
async function runStreaming(task, maxIter) {
  liveDot.classList.add("active");
  addFeedItem("start", "System", "Starting multi-agent workflow…");

  try {
    const res = await fetch("/run/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, max_iterations: maxIter }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || "Error " + res.status);
      setLoading(false);
      liveDot.classList.remove("active");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const event = JSON.parse(line.slice(6));
            handleSSEEvent(event);
          } catch (_) { /* skip malformed */ }
        }
      }
    }
  } catch (err) {
    showError("Stream error — " + err.message);
  } finally {
    setLoading(false);
    liveDot.classList.remove("active");
    clearAllActive();
  }
}

// ---------------------------------------------------------------------------
// SSE event handler
// ---------------------------------------------------------------------------
function handleSSEEvent(evt) {
  const t = evt.type;

  if (t === "thinking") {
    activateNode("supervisor", "Thinking…");
    addFeedItem("thinking", "Supervisor", evt.message || "Deciding next step…");
  }

  else if (t === "delegating") {
    const to = (evt.to || "").toLowerCase();
    deactivateNode("supervisor");
    statuses.supervisor.textContent = "Delegating";
    if (connectors[to]) connectors[to].classList.add("active");
    addFeedItem("delegating", "Supervisor",
      "→ " + evt.to, evt.task);
  }

  else if (t === "working") {
    const agent = (evt.agent || "").toLowerCase();
    clearAllActive();
    activateNode(agent, "Working…");
    addFeedItem("working", evt.agent, evt.message || "Processing…");
  }

  else if (t === "result") {
    const agent = (evt.agent || "").toLowerCase();
    deactivateNode(agent);
    markDone(agent);
    clearConnectors();
    addFeedItem("result", evt.agent, "Completed ✓", evt.preview);
  }

  else if (t === "final") {
    clearAllActive();
    markDone("supervisor");
    addFeedItem("final", "Supervisor", "✨ Final result ready");

    // Render the full report
    renderReport({
      result: evt.result,
      steps: evt.steps || [],
      steps_taken: evt.steps_taken || 0,
      cached: false,
      note: evt.note || null,
    });
  }

  else if (t === "error") {
    addFeedItem("error", "System", evt.message || "An error occurred");
  }

  else if (t === "done") {
    setLoading(false);
    liveDot.classList.remove("active");
  }
}

// ---------------------------------------------------------------------------
// Live panel helpers
// ---------------------------------------------------------------------------
function resetLivePanel() {
  liveFeed.innerHTML = "";
  Object.values(nodes).forEach(n => { n.classList.remove("active", "done"); });
  Object.values(statuses).forEach(s => { s.textContent = ""; });
  Object.values(connectors).forEach(c => { c.classList.remove("active"); });
}

function activateNode(name, status) {
  const key = name.toLowerCase();
  if (nodes[key]) {
    nodes[key].classList.add("active");
    nodes[key].classList.remove("done");
  }
  if (statuses[key]) statuses[key].textContent = status || "";
}

function deactivateNode(name) {
  const key = name.toLowerCase();
  if (nodes[key]) nodes[key].classList.remove("active");
  if (statuses[key]) statuses[key].textContent = "";
}

function markDone(name) {
  const key = name.toLowerCase();
  if (nodes[key]) nodes[key].classList.add("done");
  if (statuses[key]) statuses[key].textContent = "Done ✓";
}

function clearAllActive() {
  Object.values(nodes).forEach(n => n.classList.remove("active"));
  clearConnectors();
}

function clearConnectors() {
  Object.values(connectors).forEach(c => c.classList.remove("active"));
}

function addFeedItem(type, agent, message, detail) {
  const icons = {
    start: "▶", thinking: "🧠", delegating: "→",
    working: "⚙", result: "✓", final: "✨", error: "✕",
  };

  const el = document.createElement("div");
  el.className = "feed-item";

  const elapsed = startTime ? ((Date.now() - startTime) / 1000).toFixed(1) + "s" : "";

  let html = '<div class="feed-icon ' + type + '">' + (icons[type] || "•") + "</div>";
  html += '<div class="feed-text">';
  html += '<span class="feed-agent">' + escapeHtml(agent) + "</span> ";
  html += escapeHtml(message);
  if (detail) {
    html += '<span class="feed-detail">' + escapeHtml(detail.substring(0, 200)) + "</span>";
  }
  html += "</div>";
  html += '<span class="feed-time">' + elapsed + "</span>";

  el.innerHTML = html;
  liveFeed.appendChild(el);
  liveFeed.scrollTop = liveFeed.scrollHeight;
}

// ---------------------------------------------------------------------------
// Render report (shared between standard + streaming)
// ---------------------------------------------------------------------------
function renderReport(data) {
  sSteps.textContent = data.steps_taken || 0;

  const agentsUsed = new Set();
  if (data.steps) {
    data.steps.forEach(s => { if (s.agent) agentsUsed.add(s.agent); });
  }
  sAgents.textContent = agentsUsed.size || "—";
  sCached.textContent = data.cached ? "Hit ✓" : "Miss";

  if (data.note) {
    noteBanner.textContent = "⚠ " + data.note;
    noteBanner.classList.remove("hidden");
  } else {
    noteBanner.classList.add("hidden");
  }

  renderTimeline(data.steps || []);
  const raw = data.result || "No result produced.";
  resultContent.innerHTML = typeof marked !== "undefined"
    ? DOMPurify.sanitize(marked.parse(raw))
    : escapeHtml(raw);
  resultsEl.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Render timeline
// ---------------------------------------------------------------------------
function renderTimeline(steps) {
  timelineEl.innerHTML = "";

  if (steps.length === 0) {
    timelineEl.innerHTML = '<p style="color:var(--text-muted)">No steps recorded.</p>';
    return;
  }

  steps.forEach((step) => {
    const el = document.createElement("div");
    el.className = "step";

    const dotClass = step.action === "final" ? "final" : "delegate";
    const agentLower = (step.agent || "").toLowerCase();

    let badgeClass = "supervisor";
    if (agentLower === "researcher") badgeClass = "researcher";
    else if (agentLower === "writer") badgeClass = "writer";
    else if (agentLower === "reviewer") badgeClass = "reviewer";

    let html = '<div class="step-dot ' + dotClass + '"></div>';
    html += '<div class="step-header">';
    html += '<span class="agent-badge ' + badgeClass + '">' + escapeHtml(step.agent || "Supervisor") + '</span>';
    html += '<span class="step-iteration">Step ' + step.iteration + '</span>';
    html += '</div>';

    if (step.action === "delegate" && step.task) {
      html += '<div class="step-task">' + escapeHtml(step.task) + '</div>';
    } else if (step.action === "final") {
      html += '<div class="step-task">Synthesized final output</div>';
    }

    if (step.result_preview) {
      html += '<div class="step-preview">' + escapeHtml(step.result_preview) + '</div>';
    }

    el.innerHTML = html;
    timelineEl.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setLoading(on) {
  loadingEl.classList.toggle("hidden", !on);
  submitBtn.disabled = on;
}

function showError(msg) {
  if (msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove("hidden");
  } else {
    errorMsg.textContent = "";
    errorMsg.classList.add("hidden");
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
