const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    loadHealth();

    document.getElementById('source-code').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            runMigration();
        }
    });
});

async function loadHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        document.getElementById('model-badge').textContent = `Model: ${data.model}`;
        document.getElementById('status-badge').textContent = data.status;
    } catch (error) {
        document.getElementById('model-badge').textContent = 'API offline';
        document.getElementById('status-badge').textContent = 'offline';
    }
}

function addFile() {
    const container = document.getElementById('files-container');
    const idx = container.children.length;
    const div = document.createElement('div');
    div.className = 'file-entry';
    div.innerHTML = `
        <div class="file-header">
            <input type="text" class="file-name" placeholder="filename.js" value="">
            <button class="btn-remove-file" onclick="removeFile(this)">Remove</button>
        </div>
        <textarea class="file-content" placeholder="Paste file content here..."></textarea>
    `;
    container.appendChild(div);
}

function removeFile(btn) {
    const entry = btn.closest('.file-entry');
    if (document.querySelectorAll('.file-entry').length > 1) {
        entry.remove();
    }
}

function loadExample() {
    document.getElementById('source-framework').value = 'express';
    document.getElementById('target-framework').value = 'fastapi';

    const container = document.getElementById('files-container');
    container.innerHTML = '';
    const div = document.createElement('div');
    div.className = 'file-entry';
    div.innerHTML = `
        <div class="file-header">
            <input type="text" class="file-name" value="server.js">
            <button class="btn-remove-file" onclick="removeFile(this)">Remove</button>
        </div>
        <textarea class="file-content">const express = require('express');
const app = express();

app.use(express.json());

app.get('/api/users', async (req, res) => {
    const users = [{id: 1, name: 'John'}];
    res.json(users);
});

app.post('/api/users', async (req, res) => {
    const { name } = req.body;
    res.status(201).json({ id: 2, name });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});</textarea>
    `;
    container.appendChild(div);
}

function collectFiles() {
    const entries = document.querySelectorAll('.file-entry');
    const files = {};
    entries.forEach(entry => {
        const name = entry.querySelector('.file-name').value.trim();
        const content = entry.querySelector('.file-content').value;
        if (name && content) {
            files[name] = content;
        }
    });
    return files;
}

async function runMigration() {
    const sourceFramework = document.getElementById('source-framework').value.trim();
    const targetFramework = document.getElementById('target-framework').value.trim();
    const files = collectFiles();

    if (!sourceFramework || !targetFramework) {
        showStatus('Please enter both source and target frameworks.', 'error');
        return;
    }

    if (Object.keys(files).length === 0) {
        showStatus('Please add at least one file with a name and content.', 'error');
        return;
    }

    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result-container').innerHTML = '';
    document.getElementById('status-msg').classList.add('hidden');
    document.querySelectorAll('button').forEach(b => b.disabled = true);
    document.getElementById('status-badge').textContent = 'Migrating…';

    try {
        const response = await fetch(`${API_BASE}/migrate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_framework: sourceFramework,
                target_framework: targetFramework,
                files: files,
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Migration failed');
        }

        displayResults(data);
        document.getElementById('status-badge').textContent = data.success ? 'Done' : 'Errors';
    } catch (error) {
        showStatus(`Error: ${escapeHtml(error.message)}`, 'error');
        document.getElementById('status-badge').textContent = 'Error';
    } finally {
        document.getElementById('loading').classList.add('hidden');
        document.querySelectorAll('button').forEach(b => b.disabled = false);
    }
}

function displayResults(data) {
    let html = '';

    // Success / failure banner
    if (data.success) {
        html += `<div class="status-message success" style="margin-bottom:16px;">
            ✅ Migration completed! ${data.source_framework} → ${data.target_framework} (${data.iterations} iterations, phase: ${data.phase})
        </div>`;
    } else {
        html += `<div class="status-message error" style="margin-bottom:16px;">
            ❌ Migration completed with errors (phase: ${data.phase})
        </div>`;
    }

    // Plan / Steps
    if (data.plan_executed && data.plan_executed.length > 0) {
        html += `
        <div class="answer-box">
            <h3>📋 Migration Plan (${data.plan_executed.length} steps)</h3>
            <div>
                ${data.plan_executed.map(step => `
                    <div class="step-item ${step.status}">
                        <div class="step-header">
                            <span class="step-badge ${step.status}">${step.status}</span>
                            <span class="step-desc">${escapeHtml(step.description)}</span>
                        </div>
                        ${step.error ? `<div style="color:#c62828;font-size:0.85em;margin-top:4px;">Error: ${escapeHtml(step.error)}</div>` : ''}
                    </div>`).join('')}
            </div>
        </div>`;
    }

    // Migrated files
    if (data.migrated_files && Object.keys(data.migrated_files).length > 0) {
        html += `
        <div class="answer-box" style="border-left-color:#4db6ac;">
            <h3 style="color:#00796b;">📁 Migrated Files (${Object.keys(data.migrated_files).length})</h3>
            ${Object.entries(data.migrated_files).map(([filename, content]) => `
                <div class="file-card">
                    <div class="file-card-header" onclick="toggleFileCode(this)">
                        <span class="file-card-name">${escapeHtml(filename)}</span>
                        <span class="file-card-toggle">▼</span>
                    </div>
                    <div class="file-card-code">${escapeHtml(content)}</div>
                </div>`).join('')}
        </div>`;
    }

    // Verification
    if (data.verification) {
        const v = data.verification;
        html += `
        <div class="answer-box" style="border-left-color:#7986cb;">
            <h3 style="color:#5c6bc0;">✅ Verification</h3>
            <div class="verification-grid">
                ${Object.entries(v).map(([key, val]) => `
                    <div class="verification-card">
                        <div class="verification-value">${escapeHtml(String(val))}</div>
                        <div class="verification-label">${escapeHtml(key.replace(/_/g, ' '))}</div>
                    </div>`).join('')}
            </div>
        </div>`;
    }

    // Errors
    if (data.errors && data.errors.length > 0) {
        html += `
        <div class="answer-box" style="border-left-color:#e57373;">
            <h3 style="color:#c62828;">⚠️ Errors</h3>
            <ul class="errors-list">
                ${data.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('')}
            </ul>
        </div>`;
    }

    document.getElementById('result-container').innerHTML = html;
}

function toggleFileCode(header) {
    const code = header.nextElementSibling;
    const toggle = header.querySelector('.file-card-toggle');
    if (code.style.display === 'none') {
        code.style.display = '';
        toggle.textContent = '▼';
    } else {
        code.style.display = 'none';
        toggle.textContent = '▶';
    }
}

function showStatus(message, type) {
    const el = document.getElementById('status-msg');
    el.textContent = message;
    el.className = `status-message ${type}`;
}

function clearAll() {
    document.getElementById('source-framework').value = '';
    document.getElementById('target-framework').value = '';
    document.getElementById('files-container').innerHTML = '';
    addFile();
    document.getElementById('result-container').innerHTML = '';
    document.getElementById('status-msg').className = 'hidden';
    document.getElementById('status-badge').textContent = 'Ready';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// Initialize with one empty file entry
loadExample();
