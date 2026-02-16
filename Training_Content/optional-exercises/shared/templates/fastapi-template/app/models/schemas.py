"""
Pydantic Models for Request/Response Validation

These models provide:
- Automatic request validation
- Clear API documentation
- Type safety
- Serialization/deserialization
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """Single message in a conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

    @validator('role')
    def validate_role(cls, v):
        """Ensure role is valid."""
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message to send to AI"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)"
    )

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "message": "Explain quantum computing in simple terms",
                "max_tokens": 1024,
                "temperature": 1.0
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="AI-generated response")
    model: str = Field(..., description="Model used for generation")
    tokens_used: int = Field(..., description="Total tokens used")
    finish_reason: str = Field(..., description="Why generation stopped")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "message": "Quantum computing uses quantum mechanics...",
                "model": "claude-3-5-sonnet-20241022",
                "tokens_used": 150,
                "finish_reason": "end_turn"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    path: Optional[str] = Field(None, description="Request path that caused error")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "error": "Invalid API key",
                "status_code": 401,
                "path": "/api/chat"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
