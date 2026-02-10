from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
from typing import Optional


class URLCreate(BaseModel):
    """Request model for creating a shortened URL"""
    url: HttpUrl
    custom_alias: Optional[str] = Field(None, min_length=3, max_length=20, pattern="^[a-zA-Z0-9_-]+$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.example.com/very/long/url/path",
                "custom_alias": "mylink"
            }
        }


class URLResponse(BaseModel):
    """Response model for URL operations"""
    id: int
    original_url: str
    short_code: str
    short_url: str
    clicks: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class URLStats(BaseModel):
    """Model for URL statistics"""
    original_url: str
    short_code: str
    short_url: str
    clicks: int
    created_at: datetime
