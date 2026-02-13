# Day 1 Exercise Setup

This directory contains the exercises for Day 1: GenAI Foundations & AI-First Engineering.

## Exercise 1: Model Comparison

**Goal:** Compare the behavior of different LLM providers to understand their strengths and weaknesses.

### Quick Start

1. **Set up your API keys** in the root `.env` file (or create one):
   ```bash
   cd c:\repos\vanderlan\ai-training
   # Copy example if needed
   cp Training_Content/.env.example .env
   # Then edit .env and add at least 2 API keys
   ```

2. **Install dependencies** (if not already done):
   ```bash
   # Using your existing venv
   cd c:\repos\vanderlan\ai-training\module-1-genai-foundations\url-shortener
   .\venv\Scripts\activate
   
   # Install additional packages if needed
   pip install openai anthropic google-generativeai
   ```

3. **Run the comparison**:
   ```bash
   cd c:\repos\vanderlan\ai-training\module-1-genai-foundations\day1-exercises
   python exercise1_model_comparison.py
   ```

4. **Review results**:
   - `model_comparison_results.json` - Raw API responses
   - `model_comparison_report.md` - Template to fill out your analysis

### What Gets Tested

The script tests each model on:
1. **Code Generation** - Writing a palindrome finder function
2. **Reasoning** - Logic puzzle about sheep
3. **Refactoring** - Improving existing code
4. **Ambiguous Requests** - How they handle unclear prompts

### Expected Output

```
===========================================================================
DAY 1 - EXERCISE 1: MODEL COMPARISON
===========================================================================

Checking available providers...
  ✅ Openai is available
  ✅ Anthropic is available
  ✅ Gemini is available

Running comparison with 3 providers...
...
```

### After Running

1. Open `model_comparison_results.json` to see all responses
2. Fill out `model_comparison_report.md` with your analysis
3. Document 3 interesting differences you noticed

### Time Estimate

- Setup: 5 minutes
- Running: 5-10 minutes (depends on API response times)
- Analysis: 20-30 minutes
- **Total: ~45 minutes**

---

## Files in This Directory

```
day1-exercises/
├── README.md                          # This file
├── exercise1_model_comparison.py      # Main exercise script
├── model_comparison_results.json      # Generated: Raw results
├── model_comparison_report.md         # Generated: Your analysis template
└── DAY1_DELIVERABLES.md              # Your complete checklist
```

---

## Troubleshooting

### "No API keys found"
- Make sure you have a `.env` file in the root directory
- Add at least one of: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`

### "Module not found"
```bash
pip install openai anthropic google-generativeai
```

### "You need at least 2 providers to compare"
- The exercise requires at least 2 LLM providers
- Add API keys for at least 2 different services
- Free options: Google AI Studio, Groq

### Import errors
Make sure you're in the right virtual environment:
```bash
cd c:\repos\vanderlan\ai-training\module-1-genai-foundations\url-shortener
.\venv\Scripts\activate
```

---

## Next Steps

After completing this exercise:
1. ✅ Check off "Model Comparison Report" in [DAY1_DELIVERABLES.md](DAY1_DELIVERABLES.md)
2. 📝 Fill out the Tool Selection Matrix
3. ✅ Run environment verification
4. 🚀 Move on to Day 2!
