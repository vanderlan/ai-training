"""
FastAPI application for Migration Workflow System.

REST API for submitting and monitoring code migrations.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agent import MigrationAgent
from src.llm_client import LLMClient
from src.models import MigrationRequest, MigrationResponse, MigrationStepResponse, HealthResponse
from src.state import MigrationState, Phase

# Load environment variables
load_dotenv()

# Global state
llm_client: LLMClient = None
agent: MigrationAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager for startup/shutdown."""
    global llm_client, agent

    # Startup
    print("🚀 Starting Migration Workflow System...")
    print("Using OpenAI provider")
    llm_client = LLMClient()
    agent = MigrationAgent(llm_client)
    print("✓ LLM client initialized")
    print(f"✓ Using model: {llm_client.model}")

    yield

    # Shutdown
    print("🛑 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Migration Workflow System",
    description="AI-powered code migration system using multi-phase agent architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and LLM availability."""
    return HealthResponse(
        status="healthy",
        model=llm_client.model if llm_client else "not-initialized",
        message="Migration Workflow System is running with OpenAI",
    )


# Main migration endpoint
@app.post("/migrate", response_model=MigrationResponse)
async def migrate(request: MigrationRequest) -> MigrationResponse:
    """
    Execute a complete code migration workflow.

    The migration goes through 4 phases:
    1. **Analysis**: Analyze source code structure and patterns
    2. **Planning**: Create step-by-step migration plan
    3. **Execution**: Execute each migration step
    4. **Verification**: Verify the migrated code

    Returns the complete migration result with all generated files.
    """
    try:
        # Validate input
        if not request.files:
            raise HTTPException(status_code=400, detail="At least one source file is required")

        if not request.source_framework or not request.target_framework:
            raise HTTPException(status_code=400, detail="Both source and target frameworks are required")

        print(f"\n{'='*60}")
        print(f"🎯 Starting migration: {request.source_framework} → {request.target_framework}")
        print(f"{'='*60}\n")

        # Create initial state
        state = MigrationState(
            source_framework=request.source_framework,
            target_framework=request.target_framework,
            source_files=request.files,
        )

        # Run the migration agent
        state = agent.run(state)

        # Convert steps to response format
        plan_executed = [
            MigrationStepResponse(
                id=step.id,
                description=step.description,
                status=step.status,
                input_files=step.input_files,
                output_files=step.output_files,
                result=step.result,
                error=step.error,
            )
            for step in state.plan
        ]

        # Create response
        response = MigrationResponse(
            success=len(state.errors) == 0 and state.phase == Phase.COMPLETE,
            source_framework=state.source_framework,
            target_framework=state.target_framework,
            phase=state.phase.value,
            plan_executed=plan_executed,
            migrated_files=state.migrated_files or {},
            verification=state.verification_result,
            errors=state.errors,
            iterations=state.iterations,
        )

        print(f"\n{'='*60}")
        if response.success:
            print(f"✅ Migration completed successfully!")
            print(f"Generated {len(state.migrated_files)} files")
        else:
            print(f"❌ Migration completed with errors")
            for error in state.errors:
                print(f"   - {error}")
        print(f"{'='*60}\n")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


# Status endpoint (for async/polling pattern)
@app.get("/status/{migration_id}")
async def get_status(migration_id: str) -> dict[str, Any]:
    """Get status of a migration (placeholder for future async implementation)."""
    return {
        "migration_id": migration_id,
        "status": "not-implemented",
        "message": "This endpoint is for future async migration tracking",
    }


# Example migrations endpoint
@app.get("/examples")
async def get_examples() -> dict[str, Any]:
    """Get example migrations."""
    return {
        "examples": [
            {
                "name": "Express to FastAPI",
                "description": "Migrate a Node.js Express API to Python FastAPI",
                "source_framework": "express",
                "target_framework": "fastapi",
                "example_code": "const app = require('express')(); app.get('/api/users', (req, res) => res.json([]));",
            },
            {
                "name": "Flask to FastAPI",
                "description": "Migrate from Flask to FastAPI",
                "source_framework": "flask",
                "target_framework": "fastapi",
            },
            {
                "name": "Vue to React",
                "description": "Migrate Vue.js components to React",
                "source_framework": "vue",
                "target_framework": "react",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.main:app", host="0.0.0.0", port=port, reload=True, reload_dirs=["src"]
    )
