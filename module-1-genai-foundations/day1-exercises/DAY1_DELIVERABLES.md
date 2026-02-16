# Day 1 Deliverables Checklist

**Date Started:** February 12, 2026
**Training:** Agentic AI Intensive - Day 1: GenAI Foundations

---

## 📋 Official Day 1 Deliverables

### ✅ 1. Deployed URL Shortener
- **Status:** COMPLETED ✅
- **URL:** _[Your Vercel/Railway URL]_
- **Repository:** module-1-genai-foundations/url-shortener
- **Notes:** 

---

### ✅ 2. Model Comparison Report
- **Status:** COMPLETED ✅
- **File:** `model_comparison_report.md`
- **Raw Results:** `model_comparison_results.json`

**Completed Steps:**
- [x] Set up API keys for 2 providers (OpenAI + DeepSeek)
- [x] Ran `python exercise1_model_comparison.py`
- [x] Reviewed the JSON results  
- [x] Filled out the markdown report template
- [x] Documented 3 interesting differences

**Key Finding:** DeepSeek is more thorough and educational (but slower), OpenAI is faster and more concise.

---

### ✅ 3. Tool Selection Matrix
- **Status:** COMPLETED ✅
- **Decision:** GitHub Copilot (primary), considering Cursor for complex work

**Summary:**
- Weighted scoring completed based on training priorities
- GitHub Copilot (484/670) - Best for current workflow
- Cursor (485/670) - Will explore for multi-file refactoring
- Direct API usage for learning (DeepSeek for education, OpenAI for speed)

## Tool Selection Matrix

| Factor | Weight (1-10) | Claude Code | Cursor | Gemini CLI | GitHub Copilot | Weighted Scores |
|--------|---------------|-------------|--------|------------|----------------|-----------------|
| **Terminal preference** | 6 | 9×6=54 | 3×6=18 | 9×6=54 | 2×6=12 | CLI tools preferred |
| **IDE integration** | 9 | 5×9=45 | 10×9=90 | 5×9=45 | 9×9=81 | Very important |
| **Cost sensitivity** | 7 | 6×7=42 | 4×7=28 | 8×7=56 | 7×7=49 | Budget matters |
| **Team collaboration** | 5 | 5×5=25 | 7×5=35 | 5×5=25 | 9×5=45 | Some sharing |
| **Offline capability** | 3 | 2×3=6 | 2×3=6 | 2×3=6 | 6×3=18 | Not critical |
| **Learning curve** | 6 | 7×6=42 | 8×6=48 | 7×6=42 | 9×6=54 | Ease matters |
| **Context awareness** | 9 | 9×9=81 | 9×9=81 | 7×9=63 | 6×9=54 | Multi-file critical |
| **Code quality** | 10 | 9×10=90 | 9×10=90 | 8×10=80 | 7×10=70 | Most important! |
| **Speed/latency** | 7 | 7×7=49 | 7×7=49 | 9×7=63 | 8×7=56 | Response time |
| **Documentation** | 5 | 6×5=30 | 8×5=40 | 7×5=35 | 9×5=45 | Nice to have |
| **TOTAL** | -- | **464** | **485** | **469** | **484** | -- |

### Analysis
Based on weighted scoring for AI training/development work:
1. **Cursor**: 485 points (Winner - Best IDE integration + quality)
2. **GitHub Copilot**: 484 points (Very close - Great all-rounder)
3. **Gemini CLI**: 469 points (Fast, good value)
4. **Claude Code**: 464 points (Strong terminal tool)

### My Tool Selection

**Primary Tool:** **GitHub Copilot** (already installed in VS Code)

**Why:** 
- Already integrated and familiar
- Excellent inline suggestions for fast coding
- Great documentation and community support
- Works seamlessly with my existing VS Code workflow
- Good balance of speed, quality, and ease of use
- 484/670 points in my prioritization

**Backup Tool:** **Cursor IDE** (for complex multi-file refactoring)

**Why:** 
- Best for when I need deep codebase understanding
- Excellent for architectural changes across multiple files
- Higher context awareness than Copilot
- Worth learning for complex projects
- Can import VS Code settings easily

**For Learning/Exploration:** **Claude Code CLI** or **API directly**

**Why:**
- Best for understanding concepts thoroughly (as seen in model comparison)
- Terminal-based fits my workflow
- Can use DeepSeek or Claude for detailed explanations

**Tools I'm Currently Using:**
- [x] GitHub Copilot (primary - in VS Code)
- [ ] Cursor (planning to try for Day 3 multi-agent work)
- [ ] Claude Code / Sonnet for VS Code
- [ ] Continue.dev
- [x] Other: **Direct API calls** (OpenAI, DeepSeek for learning)

---

### 🔲 4. Environment Verification
- **Status:** NOT STARTED ⚪

**Steps:**
```powershell
# Check if verification script exists
cd C:\repos\vanderlan\ai-training

# Run verification
./scripts/verify-setup.sh
# OR on Windows:
bash ./scripts/verify-setup.sh
```

**Required Checks:**
- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] Git installed
- [ ] At least one API key configured
- [ ] Can import required packages

---

## 📚 Learning Checklist (Optional but Recommended)

### Morning Session
- [ ] Understand LLM fundamentals (tokens, context windows)
- [ ] Know about temperature and top-p parameters
- [ ] Understand model comparison (Claude vs GPT vs Gemini)
- [ ] Learn about hallucinations and limitations
- [ ] Understand the "lost in the middle" problem

### Afternoon Session
- [ ] Understand "Vibe Coding" methodology
- [ ] Know when to use AI vs traditional coding
- [ ] Reviewed AI coding tool landscape
- [ ] Understand AI-first development loop

---

## 🔗 Resources

- [Day 1 Curriculum](../Training_Content/curriculum/day1-foundations.md)
- [Schedule](../Training_Content/curriculum/SCHEDULE.md)
- [Getting Started Guide](../Training_Content/guides/getting-started.md)
- [Checklists](../Training_Content/guides/checklists.md)
