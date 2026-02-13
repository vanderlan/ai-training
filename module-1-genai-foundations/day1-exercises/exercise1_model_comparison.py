"""
Exercise 1: Model Comparison
Compare behavior across different LLM providers to understand their differences.
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

# Load environment variables from root .env file
from dotenv import load_dotenv

# Find and load .env from project root (2 levels up)
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

print(f"📁 Loading .env from: {env_path}")
if env_path.exists():
    print(f"   ✅ Found .env file")
else:
    print(f"   ⚠️  .env file not found at expected location")

# Simple LLM client implementations
class LLMClient:
    """Abstract base for LLM clients."""
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model
            self.available = True
        except Exception as e:
            print(f"⚠️  OpenAI not available: {e}")
            self.available = False

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.available:
            return "ERROR: OpenAI client not available"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

class DeepSeekClient(LLMClient):
    def __init__(self, model: str = "deepseek-chat"):
        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise Exception("No DeepSeek API key found")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            self.model = model
            self.available = True
        except Exception as e:
            print(f"⚠️  DeepSeek not available: {e}")
            self.available = False

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.available:
            return "ERROR: DeepSeek client not available"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model
            self.available = True
        except Exception as e:
            print(f"⚠️  Anthropic not available: {e}")
            self.available = False

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.available:
            return "ERROR: Anthropic client not available"
        
        # Anthropic uses 'system' separately
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 4096),
            system=system,
            messages=filtered_messages
        )
        return response.content[0].text

class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-pro"):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise Exception("No Google/Gemini API key found")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.available = True
        except Exception as e:
            print(f"⚠️  Gemini not available: {e}")
            self.available = False

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.available:
            return "ERROR: Gemini client not available"
        
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"Instructions: {msg['content']}\n")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}\n")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}\n")
        
        prompt = "\n".join(prompt_parts)
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"ERROR: Gemini response failed: {e}"

def get_llm_client(provider: str) -> Optional[LLMClient]:
    """Factory function to get the appropriate LLM client."""
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
        "deepseek": DeepSeekClient,
    }

    if provider not in providers:
        print(f"❌ Unknown provider: {provider}")
        return None

    return providers[provider]()

# Test prompts covering different capabilities
TEST_PROMPTS = [
    {
        "name": "code_generation",
        "prompt": "Write a Python function that finds the longest palindromic substring in a given string. Include type hints and a docstring.",
        "evaluate": ["correctness", "code_quality", "documentation"]
    },
    {
        "name": "reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain your reasoning step by step.",
        "evaluate": ["correct_answer", "explanation_quality"]
    },
    {
        "name": "refactoring",
        "prompt": """Refactor this code to be more idiomatic and efficient:

def get_evens(numbers):
    result = []
    for i in range(len(numbers)):
        if numbers[i] % 2 == 0:
            result.append(numbers[i])
    return result
""",
        "evaluate": ["improvement", "explanation"]
    },
    {
        "name": "ambiguous_request",
        "prompt": "I need to process some data. Can you help?",
        "evaluate": ["handling_ambiguity", "clarifying_questions"]
    }
]

def run_comparison():
    """Run the model comparison exercise."""
    print("=" * 70)
    print("DAY 1 - EXERCISE 1: MODEL COMPARISON")
    print("=" * 70)
    print()
    
    # Try all providers
    providers = ["openai", "deepseek", "anthropic", "gemini"]
    results = {}
    available_providers = []

    print("Checking available providers...")
    for provider in providers:
        client = get_llm_client(provider)
        if client and client.available:
            available_providers.append(provider)
            print(f"  ✅ {provider.capitalize()} is available")
        else:
            print(f"  ❌ {provider.capitalize()} is not configured")
    
    print()
    
    if len(available_providers) < 2:
        print("⚠️  WARNING: You need at least 2 providers configured to compare!")
        print("   Please add API keys to your .env file:")
        print("   - OPENAI_API_KEY")
        print("   - DEEPSEEK_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY or GEMINI_API_KEY")
        return None

    print(f"Running comparison with {len(available_providers)} providers...")
    print()

    for test in TEST_PROMPTS:
        results[test["name"]] = {}
        print(f"\n{'='*70}")
        print(f"TEST: {test['name'].replace('_', ' ').title()}")
        print(f"{'='*70}")
        print(f"PROMPT: {test['prompt'][:100]}...")
        print()

        for provider in available_providers:
            try:
                client = get_llm_client(provider)
                if not client or not client.available:
                    continue
                    
                messages = [
                    {"role": "system", "content": "You are a helpful programming assistant."},
                    {"role": "user", "content": test["prompt"]}
                ]

                print(f"\n--- {provider.upper()} ---")
                start_time = time.time()
                response = client.chat(messages)
                elapsed = time.time() - start_time
                
                results[test["name"]][provider] = {
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 2)
                }

                # Show truncated response
                if len(response) > 500:
                    print(response[:500] + f"\n... (truncated, {len(response)} chars total)")
                else:
                    print(response)
                
                print(f"\n⏱️  Response time: {elapsed:.2f}s")

            except Exception as e:
                results[test["name"]][provider] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"\n--- {provider.upper()} ---")
                print(f"❌ Error: {e}")

    # Save results
    output_file = "model_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*70}")
    
    return results

def generate_report_template():
    """Generate a markdown template for the comparison report."""
    template = """# Model Comparison Report - Day 1 Exercise

**Date:** {date}
**Models Tested:** [List the models you tested]

---

## Code Generation Test

**Prompt:** "Write a Python function that finds the longest palindromic substring..."

### Results

| Criteria | OpenAI | Anthropic | Gemini | Notes |
|----------|--------|-----------|--------|-------|
| Correctness (1-5) | | | | Does the code work? |
| Code Quality (1-5) | | | | Is it readable and efficient? |
| Documentation (1-5) | | | | Good docstrings and comments? |
| **Total** | | | | |

**Winner:** 

**Why:** 

---

## Reasoning Test

**Prompt:** "A farmer has 17 sheep. All but 9 die. How many sheep are left?"

### Results

| Criteria | OpenAI | Anthropic | Gemini | Notes |
|----------|--------|-----------|--------|-------|
| Correct Answer (Y/N) | | | | Did it get 9? |
| Explanation Quality (1-5) | | | | Clear reasoning? |
| Step-by-step (1-5) | | | | Showed work? |
| **Total** | | | | |

**Winner:** 

**Why:** 

---

## Refactoring Test

**Prompt:** "Refactor this code to be more idiomatic..."

### Results

| Criteria | OpenAI | Anthropic | Gemini | Notes |
|----------|--------|-----------|--------|-------|
| Improvement (1-5) | | | | Better than original? |
| Explanation (1-5) | | | | Explained changes? |
| Modern idioms (1-5) | | | | Used best practices? |
| **Total** | | | | |

**Winner:** 

**Why:** 

---

## Ambiguous Request Handling

**Prompt:** "I need to process some data. Can you help?"

### Results

| Criteria | OpenAI | Anthropic | Gemini | Notes |
|----------|--------|-----------|--------|-------|
| Handling (1-5) | | | | Dealt with ambiguity well? |
| Questions (1-5) | | | | Asked clarifying questions? |
| Helpfulness (1-5) | | | | Useful response? |
| **Total** | | | | |

**Winner:** 

**Why:** 

---

## Overall Performance

### Response Times

| Provider | Avg Response Time | Notes |
|----------|------------------|-------|
| OpenAI | | |
| Anthropic | | |
| Gemini | | |

**Fastest:** 

---

## Three Most Interesting Differences

1. **[Difference 1 Title]**
   - Description: 
   - Impact:

2. **[Difference 2 Title]**
   - Description:
   - Impact:

3. **[Difference 3 Title]**
   - Description:
   - Impact:

---

## My Conclusions

### Best for Code Generation
**Choice:** 
**Reason:** 

### Best for Reasoning
**Choice:** 
**Reason:** 

### Best for Refactoring
**Choice:** 
**Reason:** 

### Most Helpful with Ambiguous Requests
**Choice:** 
**Reason:** 

### Personal Preference
**Choice:** 
**Reason:** 

---

## Next Steps

- [ ] Consider these findings for Day 2 projects
- [ ] Test my preferred model on my own use cases
- [ ] Bookmark this comparison for future reference

""".format(date=datetime.now().strftime("%Y-%m-%d"))
    
    with open("model_comparison_report.md", "w") as f:
        f.write(template)
    
    print(f"✅ Report template created: model_comparison_report.md")
    print("   Fill this out after reviewing the results!")

if __name__ == "__main__":
    # Check for environment variables
    print("\n🔍 Checking API keys...")
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    
    print(f"  OPENAI_API_KEY: {'✅ Found' if has_openai else '❌ Missing'}")
    print(f"  DEEPSEEK_API_KEY: {'✅ Found' if has_deepseek else '❌ Missing'}")
    print(f"  ANTHROPIC_API_KEY: {'✅ Found' if has_anthropic else '❌ Missing'}")
    print(f"  GOOGLE_API_KEY: {'✅ Found' if has_gemini else '❌ Missing'}")
    print()
    
    if not (has_openai or has_deepseek or has_anthropic or has_gemini):
        print("❌ ERROR: No API keys found!")
        print("   Please set at least one API key in your .env file")
        exit(1)
    
    # Run the comparison
    results = run_comparison()
    
    if results:
        print()
        generate_report_template()
        print()
        print("📋 NEXT STEPS:")
        print("   1. Review model_comparison_results.json")
        print("   2. Fill out model_comparison_report.md with your analysis")
        print("   3. Note 3 interesting differences you observed")
