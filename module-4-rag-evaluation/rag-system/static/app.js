const API_BASE = window.location.origin;

// Configure marked for markdown rendering
document.addEventListener('DOMContentLoaded', () => {
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return code;
            }
        });
    }
    loadHealth();
    loadStats();

    document.getElementById('query-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            query();
        }
    });
});

async function loadHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        const badge = document.getElementById('provider-badge');
        badge.textContent = `Provider: ${data.provider}`;
        badge.className = `badge provider-${data.provider}`;
    } catch (error) {
        const badge = document.getElementById('provider-badge');
        badge.textContent = 'API offline';
        badge.className = 'badge';
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        document.getElementById('chunks-badge').textContent = `${data.count} chunks`;
    } catch (error) {
        document.getElementById('chunks-badge').textContent = '0 chunks';
    }
}

async function indexGithubRepo() {
    const url = document.getElementById('github-url').value.trim();
    const branch = document.getElementById('github-branch').value.trim();

    if (!url) {
        showStatus('index-status', 'Please enter a GitHub repository URL', 'error');
        return;
    }

    showStatus('index-status', 'Indexing...', 'info');

    try {
        const payload = { url };
        if (branch) {
            payload.branch = branch;
        }

        const response = await fetch(`${API_BASE}/index/github`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Indexing failed');
        }

        const data = await response.json();
        showStatus('index-status',
            `Indexed ${data.indexed_chunks} chunks from ${data.files_processed} files (${data.repository}@${data.branch})`,
            'success');
        loadStats();
    } catch (error) {
        showStatus('index-status', `Error: ${error.message}`, 'error');
    }
}

async function clearIndex() {
    showStatus('index-status', 'Clearing...', 'info');

    try {
        const response = await fetch(`${API_BASE}/index`, { method: 'DELETE' });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Clear failed');
        }

        showStatus('index-status', 'Index cleared', 'success');
        loadStats();
        document.getElementById('query-result').innerHTML = '';
    } catch (error) {
        showStatus('index-status', `Error: ${error.message}`, 'error');
    }
}

async function query() {
    const question = document.getElementById('query-input').value.trim();
    if (!question) return;

    const loadingEl = document.getElementById('query-loading');
    const resultEl = document.getElementById('query-result');

    loadingEl.classList.remove('hidden');
    resultEl.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, n_results: 3 })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        resultEl.innerHTML = `<div class="status-message error">Error: ${escapeHtml(error.message)}</div>`;
    } finally {
        loadingEl.classList.add('hidden');
    }
}

function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    return '<p>' + escapeHtml(text) + '</p>';
}

function parseContextChunks(contextUsed) {
    if (!contextUsed) return {};
    const chunks = {};
    const parts = contextUsed.split(/(?=File: )/);
    for (const part of parts) {
        const match = part.match(/^File:\s*(.+?)\s*\(line\s*(\d+)\)\s*\n```[\s\S]*?\n([\s\S]*?)```/);
        if (match) {
            const key = match[1].trim() + ':' + match[2];
            chunks[key] = match[3].trim();
        }
    }
    return chunks;
}

function displayResults(data) {
    const resultEl = document.getElementById('query-result');

    let html = `
        <div class="answer-box">
            <h3>Answer</h3>
            <div class="answer-text">${renderMarkdown(data.answer || '')}</div>
        </div>
    `;

    const contextChunks = parseContextChunks(data.context_used);

    if (data.sources && data.sources.length > 0) {
        html += '<div class="sources"><h3>Sources (Retrieved Chunks)</h3>';
        data.sources.forEach((source, idx) => {
            // Support both flat and nested metadata structures
            const sourceFile = source.file
                || (source.metadata && source.metadata.file)
                || (source.metadata && source.metadata.filename)
                || 'unknown';
            const line = source.line || (source.metadata && source.metadata.line) || '';
            const relevance = source.relevance || source.score || 0;
            const relevanceStr = relevance ? ` (relevance: ${(relevance * 100).toFixed(1)}%)` : '';

            // Try to get content from context_used parsing
            const chunkKey = sourceFile + ':' + line;
            const content = source.content || contextChunks[chunkKey] || '';

            html += `
                <div class="source-item">
                    <div class="source-header">
                        <span class="source-file">${escapeHtml(sourceFile)}${line ? ' (line ' + line + ')' : ''}</span>
                        <span class="source-score">Chunk ${idx + 1}${relevanceStr}</span>
                    </div>
                    ${content ? '<div class="source-content">' + escapeHtml(content) + '</div>' : ''}
                </div>
            `;
        });
        html += '</div>';
    }

    resultEl.innerHTML = html;

    // Apply syntax highlighting to code blocks in the answer
    if (typeof hljs !== 'undefined') {
        resultEl.querySelectorAll('.answer-text pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    }
}

function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `status-message ${type}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
