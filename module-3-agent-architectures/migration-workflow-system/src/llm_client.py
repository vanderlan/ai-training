"""
LLM Client for interacting with OpenAI API.

Handles all communication with the OpenAI API.
"""

import json
import os
from typing import Optional, Dict, Any
import openai


class LLMClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """Initialize the LLM client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("MODEL", "gpt-4-turbo")
        openai.api_key = self.api_key

    def generate_analysis(self, system_prompt: str, files_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate analysis of source files."""
        files_str = "\n\n".join([f"## File: {name}\n```\n{content}\n```" for name, content in files_content.items()])

        response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Analyze these files:\n\n{files_str}",
                }
            ],
        )

        response_text = response.choices[0].message.content
        return self._parse_json_response(response_text)

    def generate_plan(
        self, system_prompt: str, source_framework: str, target_framework: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate migration plan."""
        analysis_str = json.dumps(analysis, indent=2)

        response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""Source Framework: {source_framework}
Target Framework: {target_framework}

Analysis Results:
{analysis_str}

Create a detailed migration plan.""",
                }
            ],
        )

        response_text = response.choices[0].message.content
        return self._parse_json_response(response_text)

    def execute_step(
        self, system_prompt: str, step_description: str, input_files: Dict[str, str], target_framework: str
    ) -> Dict[str, Any]:
        """Execute a single migration step."""
        files_str = "\n\n".join([f"## File: {name}\n```\n{content}\n```" for name, content in input_files.items()])

        response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=3500,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""Target Framework: {target_framework}

Step to Execute:
{step_description}

Source Files to Transform:
{files_str}

Transform these files according to the step requirements.""",
                }
            ],
        )

        response_text = response.choices[0].message.content
        return self._parse_json_response(response_text)

    def verify_migration(self, system_prompt: str, migrated_files: Dict[str, str], target_framework: str) -> Dict[str, Any]:
        """Verify the migrated code."""
        files_str = "\n\n".join([f"## File: {name}\n```\n{content}\n```" for name, content in migrated_files.items()])

        response = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""Target Framework: {target_framework}

Migrated Code to Verify:
{files_str}

Verify this migrated code.""",
                }
            ],
        )

        response_text = response.choices[0].message.content
        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with robust error handling."""
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from LLM")
        
        json_str = None
        
        # Try 1: Find JSON in code block
        if "```json" in response_text:
            try:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    json_str = response_text[start:end].strip()
            except:
                pass
        
        # Try 2: Find JSON in curly braces
        if not json_str and "{" in response_text:
            try:
                start = response_text.find("{")
                # Find matching closing brace
                brace_count = 0
                for i in range(start, len(response_text)):
                    if response_text[i] == "{":
                        brace_count += 1
                    elif response_text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response_text[start:i+1]
                            break
            except:
                pass
        
        # Try 3: Use the whole response
        if not json_str:
            json_str = response_text.strip()
        
        # Attempt to parse
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If JSON is malformed, create a valid response structure
                print(f"⚠️  Warning: JSON parsing failed - {str(e)}")
                print(f"Response text: {json_str[:200]}...")
                return {
                    "status": "completed",
                    "generated_files": {},
                    "summary": "Step completed with response parsing issues",
                    "issues_encountered": ["Response was not valid JSON"],
                    "notes": "Fallback response due to LLM formatting"
                }
        
        raise ValueError("No JSON found in response and fallback failed")
