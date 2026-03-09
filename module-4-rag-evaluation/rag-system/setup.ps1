# Quick Start Setup Script for RAG System
# Run this script to set up your environment

Write-Host "🚀 Setting up RAG System..." -ForegroundColor Cyan
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit .env file and add your API keys!" -ForegroundColor Red
    Write-Host "   Required: ANTHROPIC_API_KEY or OPENAI_API_KEY" -ForegroundColor Red
    Write-Host "   The .env file has been created. Please add your keys before running tests." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file and add your API keys"
Write-Host "2. Run: python test_rag.py (to test the system)"
Write-Host "3. Run: uvicorn main:app --reload (to start the API server)"
Write-Host ""
