# RAG System with Evaluation

**Module 4 Project: RAG & Evaluation**

## 🎯 Project Overview

A complete Retrieval-Augmented Generation (RAG) system with comprehensive evaluation metrics. This implementation covers intelligent code chunking, vector embeddings, semantic search, and robust evaluation frameworks including LLM-as-judge patterns.

## ✅ Features Implemented

### Core Features
- ✅ Document ingestion and preprocessing
- ✅ Intelligent code chunking (Python, JavaScript/TypeScript, generic)
- ✅ Embedding generation (OpenAI + sentence-transformers fallback)
- ✅ Vector database integration (ChromaDB with persistence)
- ✅ Semantic search and retrieval
- ✅ Response generation with context

### Evaluation Features
- ✅ Precision@K and Recall@K metrics
- ✅ Mean Reciprocal Rank (MRR)
- ✅ LLM-as-judge evaluation
- ✅ FastAPI-based REST API
- ✅ Test suite with sample data

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Vector DB:** ChromaDB with persistence
- **Embeddings:** OpenAI text-embedding-3-small (with sentence-transformers fallback)
- **LLM:** DeepSeek / Anthropic Claude / OpenAI GPT / Google Gemini
- **Framework:** FastAPI
- **Deployment:** Docker, Railway, Vercel
- **Live URL:** https://rag-system-vanderlan-lab4.vercel.app

## 📁 Project Structure

```
rag-system/
├── README.md
├── DEPLOYMENT.md
├── requirements.txt
├── Dockerfile
├── main.py                 # FastAPI application
├── test_rag.py            # Test suite
├── .env.example           # Environment variables template
├── src/
│   ├── __init__.py
│   ├── chunker.py         # Intelligent code chunking
│   ├── vector_store.py    # ChromaDB integration
│   ├── pipeline.py        # Main RAG pipeline
│   ├── evaluation.py      # Evaluation metrics
│   └── llm_client.py      # LLM abstraction layer
└── data/
    ├── sample_code/       # Sample code files
    │   ├── data_processor.py
    │   ├── api_handler.py
    │   └── user_service.ts
    └── test_queries.json  # Evaluation queries
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs a cross-platform default setup.

- On Python 3.12+ (especially Windows), Chroma is skipped to avoid native `chroma-hnswlib` build errors.
- The app still runs using the built-in fallback vector backend.

For full ChromaDB support:

```bash
pip install -r requirements-chroma.txt
```

If that fails on Windows, use Python 3.11 or install Microsoft C++ Build Tools.

### 3. Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env — set LLM_PROVIDER to one of: deepseek, anthropic, openai, gemini
# Then add the matching API key:
# DEEPSEEK_API_KEY   — for DeepSeek (default provider)
# ANTHROPIC_API_KEY  — for Claude
# OPENAI_API_KEY     — for GPT (also used for embeddings)
# GOOGLE_API_KEY     — for Gemini
```

### 4. Run Test Suite

```bash
python test_rag.py
```

This will:
- Index the sample code files
- Run test queries
- Display evaluation metrics

### 5. Start API Server

```bash
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## 🧪 Using the API

### Index a local directory

```bash
curl -X POST "http://localhost:8000/index/directory" \
  -H "Content-Type: application/json" \
  -d '{"directory": "./data/sample_code"}'
```

### Index a public GitHub repository

```bash
curl -X POST "http://localhost:8000/index/github" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/owner/repo"}'
```

### Query the codebase

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a new user?", "n_results": 5}'
```

### Run evaluation

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d @data/test_queries.json
```

### Statistics and housekeeping

```bash
curl http://localhost:8000/stats
curl -X DELETE http://localhost:8000/index
```

## 📊 Evaluation Metrics

### Retrieval Metrics
- **Precision@K:** Fraction of retrieved documents that are relevant
- **Recall@K:** Fraction of relevant documents that were retrieved
- **MRR:** Mean Reciprocal Rank - position of first relevant result

### Generation Metrics (LLM-as-Judge)
- **Relevance:** How well the answer addresses the question (1-5 scale)
- **Accuracy:** How well the answer matches expected answer (1-5 scale)

## 🏗️ Architecture

### Chunking Strategy
- **Python:** Splits by function/class definitions with decorators
- **JavaScript/TypeScript:** Splits by function/class declarations
- **Generic:** Size-based chunking with configurable overlap

### Vector Store
- Uses ChromaDB for persistent vector storage when available
- Falls back to an in-memory lexical retriever when Chroma is unavailable
- Cosine similarity for semantic search (Chroma mode)

### RAG Pipeline
1. **Index:** Chunk code → Generate embeddings → Store in vector DB
2. **Retrieve:** Query → Find similar chunks → Return top K
3. **Generate:** Build context → Send to LLM → Return answer

## 🧪 Sample Test Results

```
Testing Indexing
============================================================
✓ Indexed 15 chunks
✓ Collection stats: {'count': 15, 'name': 'codebase'}

Testing Queries
============================================================
📝 Question: How do I load data from a file?
💡 Answer: Use the DataProcessor class's load_data method...
📚 Sources: 3 files
   - data_processor.py (load_data) - relevance: 0.892

Retrieval Metrics:
   precision@5: 0.867
   recall@5: 0.933
   mrr: 0.889
   
Generation Metrics:
   relevance: 0.900
   accuracy: 0.850
```

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment instructions.

**Live production URL:** https://rag-system-vanderlan-lab4.vercel.app

## 🎓 Learning Objectives Achieved

✅ Implement complete RAG pipeline  
✅ Design effective chunking strategies for code  
✅ Optimize retrieval with embeddings  
✅ Build comprehensive evaluation frameworks  
✅ Use LLM-as-judge patterns  
✅ Create production-ready REST API  

---

**Part of Taller AI Training Program - Module 4**
