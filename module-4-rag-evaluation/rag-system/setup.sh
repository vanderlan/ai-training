#!/bin/bash
# Quick Start Setup Script for RAG System (Mac/Linux)
# Run: bash setup.sh

echo "🚀 Setting up RAG System..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1)
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys!"
    echo "   Required: ANTHROPIC_API_KEY or OPENAI_API_KEY"
    echo "   The .env file has been created. Please add your keys before running tests."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Run: python test_rag.py (to test the system)"
echo "3. Run: uvicorn main:app --reload (to start the API server)"
echo ""
