# Dockerfile for Migration Workflow System

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application (uses PORT env var from Railway, defaults to 8000)
CMD python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
