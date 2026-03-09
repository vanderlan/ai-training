"""Sample Python file for RAG testing - API Handler Module."""
from typing import Dict, List, Optional
import json


class APIHandler:
    """Handle API requests and responses."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize API handler.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.retry_count = 3

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to API.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        # Simulate API call
        return {"status": "success", "data": {}}

    def post(self, endpoint: str, data: Dict) -> Dict:
        """Make POST request to API.
        
        Args:
            endpoint: API endpoint path
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        # Simulate API call
        return {"status": "success", "id": "12345"}

    def handle_error(self, error: Exception) -> Dict:
        """Handle API errors gracefully.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error response dictionary
        """
        return {
            "status": "error",
            "message": str(error),
            "retry": self.retry_count > 0
        }


def format_response(data: Dict, format_type: str = "json") -> str:
    """Format API response data.
    
    Args:
        data: Response data to format
        format_type: Output format type (json, xml, etc)
        
    Returns:
        Formatted response string
    """
    if format_type == "json":
        return json.dumps(data, indent=2)
    return str(data)


def validate_request(request_data: Dict, required_fields: List[str]) -> bool:
    """Validate that request contains required fields.
    
    Args:
        request_data: Request data to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    return all(field in request_data for field in required_fields)
