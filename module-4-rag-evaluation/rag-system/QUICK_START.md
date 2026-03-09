# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Setup Environment

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: full ChromaDB backend
# (requires Python 3.11 or MSVC Build Tools on Windows)
pip install -r requirements-chroma.txt

# Create .env file
cp .env.example .env
```

### Step 2: Configure API Keys

Edit `.env` file and add your API key(s):

```bash
# At minimum, you need ONE of these:
ANTHROPIC_API_KEY=sk-ant-...        # For Claude (recommended)
OPENAI_API_KEY=sk-...               # For GPT-4 and embeddings

# Optional:
LLM_PROVIDER=anthropic              # anthropic, openai, or gemini
```

**Free Option:** If you don't have API keys, the system will automatically use sentence-transformers for embeddings (slower but free). You'll still need an LLM API key for answer generation.

**Windows + Python 3.12 note:** `requirements.txt` intentionally skips Chroma to avoid `chroma-hnswlib` native build failures. The app still runs with a fallback backend.

### Step 3: Run Tests

```bash
python test_rag.py
```

You should see:
```
🚀 RAG System Test Suite

Testing Indexing
============================================================
✓ Indexed 15 chunks
✓ Collection stats: {'count': 15, 'name': 'test_codebase'}

Testing Queries
============================================================
📝 Question: How do I load data from a file?
💡 Answer: Use the DataProcessor class's load_data method...
...
```

## 🎯 What Next?

### Start the API Server
```bash
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

### Try the API

**Index your own code:**
```bash
curl -X POST "http://localhost:8000/index/directory" \
  -H "Content-Type: application/json" \
  -d '{"directory": "./src"}'
```

**Ask questions:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the chunker work?", "n_results": 5}'
```

### Run Evaluation
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d @data/test_queries.json
```

## 📚 Documentation

- [README.md](README.md) - Full project documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guides
- [DELIVERABLES.md](DELIVERABLES.md) - Feature checklist

## 🐛 Troubleshooting

**Import errors?**
```bash
# Make sure virtual environment is activated
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**ChromaDB errors?**
```bash
# Try removing the database and re-indexing
rm -rf chroma_db
python test_rag.py
```

**API key errors?**
- Check that .env file exists and has your keys
- Make sure keys are valid and have credits
- Try the free option: remove OPENAI_API_KEY from .env

## 💡 Tips

1. **Start small:** Test with sample data first
2. **Check logs:** Errors are printed to console
3. **Use /docs:** FastAPI auto-generates interactive docs
4. **Monitor costs:** Embeddings and LLM calls use API credits

## 🎓 Learning Path

1. ✅ Run tests to see how it works
2. 📖 Read the code in `src/` to understand architecture
3. 🔧 Modify chunking strategies for your needs
4. 🚀 Deploy to Railway or your preferred platform
5. 📊 Experiment with evaluation metrics
