# Model Comparison Report - Day 1 Exercise

**Date:** 2026-02-12
**Models Tested:** OpenAI (gpt-4o-mini), DeepSeek (deepseek-chat)

**Note:** Anthropic and Gemini were not available due to API configuration issues.

---

## Code Generation Test

**Prompt:** "Write a Python function that finds the longest palindromic substring..."

### Results

| Criteria | OpenAI | DeepSeek | Notes |
|----------|--------|----------|-------|
| Correctness (1-5) | 5 | 5 | Both used "expand around center" approach correctly |
| Code Quality (1-5) | 5 | 5 | Both clean, readable, well-structured |
| Documentation (1-5) | 4 | 5 | DeepSeek provided extensive docstrings + examples |
| Completeness (1-5) | 4 | 5 | DeepSeek added DP alternative + test suite |
| **Total** | 18/20 | 20/20 | DeepSeek more comprehensive but slower |

**Winner:** DeepSeek (by thoroughness)

**Why:** While both provided correct, high-quality code, DeepSeek went far beyond by including:
- Bonus dynamic programming implementation for comparison
- Complete test suite with edge cases
- Extensive examples and explanations
- However, took 41s vs OpenAI's 9s 

---

## Reasoning Test

**Prompt:** "A farmer has 17 sheep. All but 9 die. How many sheep are left?"

### Results

| Criteria | OpenAI | DeepSeek | Notes |
|----------|--------|----------|-------|
| Correct Answer (Y/N) | Y | Y | Both correctly answered: 9 sheep |
| Explanation Quality (1-5) | 5 | 4 | OpenAI more formal, DeepSeek more casual |
| Step-by-step (1-5) | 5 | 5 | Both showed clear step-by-step reasoning |
| Clarity (1-5) | 5 | 5 | Both very clear and easy to follow |
| **Total** | 15/15 | 14/15 | Essentially tied |

**Winner:** Tie (slight edge to OpenAI for polish)

**Why:** Both models correctly solved the logic puzzle. OpenAI used more formal structure with numbered steps. DeepSeek was more conversational but equally clear. Response times: OpenAI 3.4s vs DeepSeek 5.7s 

---

## Refactoring Test

**Prompt:** "Refactor this code to be more idiomatic..."

### Results
DeepSeek | Notes |
|----------|--------|----------|-------|
| Improvement (1-5) | 5 | 5 | Both gave list comprehension solution |
| Explanation (1-5) | 4 | 5 | DeepSeek explained improvements in detail |
| Modern idioms (1-5) | 5 | 5 | Both used proper Python idioms |
| Alternatives (1-5) | 3 | 5 | DeepSeek showed filter() and type hints versions |
| **Total** | 17/20 | 20/20 | DeepSeek more educational |

**Winner:** DeepSeek (more comprehensive)

**Why:** Both provided the same core solution (list comprehension), but DeepSeek added:
- Detailed breakdown of improvements made
- Alternative approaches (filter(), type hinting)
- Explained why list comprehension is preferred
- Response: OpenAI 3.4s, DeepSeek 9.6s
**Why:** 

---

## Ambiguous Request HDeepSeek | Notes |
|----------|--------|----------|-------|
| Handling (1-5) | 3 | 5 | DeepSeek much more proactive |
| Questions (1-5) | 2 | 5 | OpenAI asked 1 vague question, DeepSeek asked 4 specific |
| Helpfulness (1-5) | 3 | 5 | DeepSeek provided structured approach |
| Detail (1-5) | 2 | 5 | DeepSeek covered tools, formats, goals |
| **Total** | 10/20 | 20/20 | Clear winner |

**Winner:** DeepSeek (significantly better)

**Why:** This showed the biggest difference:
- **OpenAI**: Brief response, asked generic question about data type and processing
- **DeepSeek**: Asked 4 specific clarifying questions:
  1. What type of data? (with examples)
  2. What is the goal? (with use cases)
  3. Can you share a sample?
  4. What tools/languages do you prefer?
- Much more actionable and helpful for real-world scenariosons (1-5) | | | | Asked clarifying questions? |
| Helpfulness (1-5) | | | | UsefuMin | Max | Notes |
|----------|------------------|-----|-----|-------|
| OpenAI | 4.3s | 1.2s | 9.5s | Consistently fast |
| DeepSeek | 15.7s | 5.7s | 41.4s | Slower but more thorough |

**Fastest:** OpenAI (3.6x faster on average)

### Speed vs Quality Tradeoff
- **OpenAI**: Fast, concise, correct answers. Great for quick iterations.
- **DeepSeek**: Slower but provides more comprehensive responses with alternatives, examples, and educational value.
---

## OvResponse Verbosity and Educational Value**
   - **Description:** DeepSeek consistently provided more detailed, educational responses. For the palindrome function, DeepSeek gave 5,423 characters including alternative implementations and test cases, while OpenAI gave 2,216 characters with just the main solution.
   - **Impact:** DeepSeek is better for learning and understanding multiple approaches. OpenAI is better when you just need a working solution quickly. In production, the extra context from DeepSeek could help with maintenance and understanding edge cases.

2. **Ambiguous Request Handling Philosophy**
   - **Description:** When faced with "I need to process some data," OpenAI gave a generic "sure, what type?" response (205 chars), while DeepSeek asked 4 structured clarifying questions with specific examples (620 chars).
   - **Impact:** This reveals different design philosophies. DeepSeek acts more like a consultant proactively structuring the problem space. OpenAI waits for more input. For real-world pair programming or requirements gathering, DeepSeek's approach would save time and prevent ambiguity.

3. **Speed vs Thoroughness Tradeoff**
   - **Description:** OpenAI averaged 4.3 seconds per response with concise answers. DeepSeek averaged 15.7 seconds (one response took 41s!) but included alternatives, examples, and explanations.
   - **Impact:** This is the classic speed-quality tradeoff. For rapid prototyping or simple questions, OpenAI's speed wins. For complex problems where you need to understand tradeoffs or learn best practices, DeepSeek's thoroughness justifies the wait. Cost-wise, longer responses = higher token usage. 

---

## Three Most Interesting Differences

1. **[Difference 1 Title]**
   - DescripDeepSeek
**Reason:** While both produce correct code, DeepSeek's comprehensive approach with examples, tests, and alternative implementations is invaluable for production code. The extra 30 seconds is worth it when you're learning or need robust solutions.

### Best for Reasoning
**Choice:** Tie (slight edge to OpenAI)
**Reason:** Both solved the logic puzzle correctly with clear explanations. OpenAI's more formal structure might be slightly better for documentation, but the difference is minimal. Choose based on preference for formal vs casual tone.

### Best for Refactoring
**Choice:** DeepSeek
**Reason:** Both gave the same core answer, but DeepSeek explained WHY each change improves the code and offered alternatives. This educational approach helps you become a better developer rather than just copying solutions.

### Most Helpful with Ambiguous Requests
**Choice:** DeepSeek (by a landslide)
**Reason:** The difference here was stark. DeepSeek's structured clarifying questions demonstrate better real-world problem-solving skills. It's like talking to a senior engineer who asks the right questions vs a junior who waits for explicit instructions.

### Personal Preference
**Choice:** Use both strategically
**Reason:** 
- **OpenAI for:** Quick iterations, simple tasks, when speed matters, production latency-sensitive apps
- **DeepSeek for:** Learning, complex problems, exploratory work, when I need to understand tradeoffs, code reviews

**Cost consideration:** DeepSeek is typically more affordable per token, but generates more tokens. For my use case (learning and training), DeepSeek's thoroughness is worth the time.

---

## Key Takeaway for This Training

**Pattern Observed:** DeepSeek seems optimized for educational use and thorough explanations, while OpenAI (gpt-4o-mini) is optimized for speed and conciseness. This mirrors the difference between getting answers from a teacher vs getting answers from a smart peer.

For **Days 2-5 of this training**, I'll likely use:
- DeepSeek when learning new concepts or needing detailed explanations
- OpenAI when I understand the concept and just need to implement quickly

---

## Next Steps

- [x] Model comparison completed
- [x] Documented findings and preferences
- [ ] Apply these insights to Day 2 prompting exercises
- [ ] Consider using DeepSeek for complex agent design in Day 3
- [ ] Bookmark this comparison for future project decisions

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

