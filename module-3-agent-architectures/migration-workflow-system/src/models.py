"""
Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MigrationRequest(BaseModel):
    """Request to start a migration."""

    source_framework: str = Field(..., description="Source framework (e.g., 'express', 'django')")
    target_framework: str = Field(..., description="Target framework (e.g., 'fastapi', 'nest')")
    files: Dict[str, str] = Field(..., description="Source files as {filename: content} dict")

    class Config:
        json_schema_extra = {
            "example": {
                "source_framework": "express",
                "target_framework": "fastapi",
                "files": {
                    "server.js": "const express = require('express');\nconst app = express();\napp.get('/users', (req, res) => res.json([]));\napp.listen(3000);"
                },
            }
        }


class MigrationStepResponse(BaseModel):
    """Single step in the migration plan."""

    id: int
    description: str
    status: str
    input_files: List[str] = []
    output_files: List[str] = []
    result: Optional[str] = None
    error: Optional[str] = None


class MigrationResponse(BaseModel):
    """Response from migration endpoint."""

    success: bool = Field(..., description="Whether migration completed successfully")
    source_framework: str
    target_framework: str
    phase: str = Field(..., description="Current migration phase")
    plan_executed: List[MigrationStepResponse] = Field(..., description="Steps executed")
    migrated_files: Dict[str, str] = Field(..., description="Generated migrated files")
    verification: Optional[Dict[str, Any]] = Field(None, description="Verification results")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    iterations: int = Field(..., description="Number of iterations completed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    message: str
