# Lab 02: Code Analyzer Agent

**Module 2: Advanced Prompting for Engineering**

[![Deployed on Vercel](https://img.shields.io/badge/Deployed-Vercel-black?logo=vercel)](https://code-analyzer-agent.vercel.app)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue?logo=typescript)](https://www.typescriptlang.org/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-green?logo=openai)](https://openai.com/)

**Live Demo:** 🚀 [https://code-analyzer-agent.vercel.app](https://code-analyzer-agent.vercel.app)

An intelligent code analysis agent that leverages advanced prompting techniques to analyze code and provide structured, actionable feedback.

## 🎯 Project Objectives

### Learning Goals
- Master RCFG framework (Role, Context, Format, Goal) for effective prompting
- Implement Chain-of-Thought prompting for structured code analysis
- Design specialized prompts for different analysis types (general, security, performance)
- Extract structured output using Zod validation
- Build a multi-provider LLM client abstraction
- Deploy a production-ready API to Vercel

### What This Agent Does

Provides comprehensive code analysis including:
- **Code Quality Summary**: 2-3 sentence overview of what the code does and its overall quality
- **Issues Detection**: Categorized by severity (critical, high, medium, low) and type (bug, security, performance, style, maintainability)
- **Improvement Suggestions**: Actionable recommendations for better code
- **Quality Metrics**: Complexity, readability, and test coverage estimates

## 📁 Project Structure

```
code-analyzer-agent/
├── README.md              # This file
├── LAB02_DELIVERABLES.md  # Lab completion documentation
├── DEPLOYMENT.md          # Deployment guide
├── package.json           # Dependencies and scripts
├── tsconfig.json          # TypeScript configuration
├── vercel.json            # Vercel deployment configuration
├── .env.example           # Environment variables template
├── .env                   # Your API keys (git-ignored)
├── .vercelignore          # Vercel deployment exclusions
├── api/
│   └── index.ts           # Vercel serverless API endpoint
├── public/
│   └── index.html         # Web UI for code analysis
└── src/
    ├── index.ts           # CLI entry point (npm run analyze)
    ├── analyzer.ts        # CodeAnalyzer class with 3 analysis modes
    ├── llm-client.ts      # Multi-provider LLM abstraction (OpenAI/Anthropic/Google)
    ├── prompts.ts         # RCFG-structured system prompts
    └── types.ts           # Zod schemas for type-safe output validation
```

## 🚀 Quick Start

### 1. Set up your API keys

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key(s)
```

Your `.env` file should contain:
```bash
LLM_PROVIDER=anthropic  # Choose: anthropic, openai, or google

# Add at least one API key
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

### 2. Install dependencies

```bash
npm install
```

### 3. Run the analyzer

```bash
npm run analyze
```

This command runs the analyzer with sample code, performing three types of analysis:
- **General Analysis**: Overall code quality review
- **Security Analysis**: Focus on security vulnerabilities
- **Performance Analysis**: Focus on performance issues

Results are displayed in the terminal and saved to `analysis_results.json`.

---

## 📊 What Gets Analyzed

The script analyzes sample Python and TypeScript code for:

1. **General Code Quality**
   - Code structure and organization
   - Variable naming and readability
   - Best practices
   - Potential bugs

2. **Security Issues**
   - SQL injection vulnerabilities
   - Command injection risks
   - Hardcoded secrets
   - Input validation problems
   - Authentication/authorization issues

3. **Performance Problems**
   - Algorithm complexity (Big O)
   - Unnecessary iterations
   - Memory usage
   - Optimization opportunities

---

## 🌐 Web Interface & API

### Web UI

Visit [https://code-analyzer-agent.vercel.app](https://code-analyzer-agent.vercel.app) for an interactive web interface featuring:

- **Beautiful gradient UI** with code input textarea
- **Multi-language support**: Python, JavaScript, TypeScript, Java, C#, Go, Rust
- **Three analysis modes** accessible via buttons:
  - 📊 General Analysis
  - 🔒 Security Analysis
  - ⚡ Performance Analysis
- **Real-time results** with color-coded severity levels
- **Responsive design** that works on desktop and mobile

### API Endpoints

All endpoints accept POST requests with JSON body:

#### 1. General Code Analysis
```bash
POST https://code-analyzer-agent.vercel.app/api/analyze

Body:
{
  "code": "def hello():\n    print('world')",
  "language": "python"
}

Response:
{
  "success": true,
  "analysis": {
    "summary": "...",
    "issues": [...],
    "suggestions": [...],
    "metrics": {...}
  }
}
```

#### 2. Security-Focused Analysis
```bash
POST https://code-analyzer-agent.vercel.app/api/analyze/security
```

#### 3. Performance-Focused Analysis
```bash
POST https://code-analyzer-agent.vercel.app/api/analyze/performance
```

### Example cURL Request

```bash
curl -X POST "https://code-analyzer-agent.vercel.app/api/analyze/security" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    return db.execute(query)",
    "language": "python"
  }'
```

---

## 🧠 Advanced Prompting Techniques Used

### 1. RCFG Framework

All prompts use the RCFG structure:

- **Role**: "You are an expert code reviewer"
- **Context**: Analysis type (general/security/performance) and language
- **Format**: Explicit JSON schema with Zod validation
- **Goal**: "Analyze the provided code and return a structured analysis"

### 2. Chain-of-Thought Prompting

The system prompt instructs the LLM to:
1. Understand what the code does (summary)
2. Identify specific issues (with line numbers and categories)
3. Provide actionable suggestions
4. Estimate quality metrics

### 3. Specialized Analysis Modes

**Security Focus Prompt**:
- SQL injection, command injection, path traversal
- Hardcoded secrets and credentials
- Input validation and sanitization
- Authentication and authorization flaws

**Performance Focus Prompt**:
- Algorithm complexity (Big O)
- Inefficient loops and iterations
- Memory allocation issues
- Database query optimization
- Caching opportunities

### 4. Structured Output Extraction

Uses Zod schemas to enforce:
- Type-safe results
- Consistent format across all LLM responses
- Validation of severity levels, categories, and metrics
- Proper error handling for malformed LLM responses

---

## 🏗️ Architecture & Implementation

### Multi-Provider LLM Client

Supports three LLM providers with unified interface:

```typescript
// Automatically selects provider from environment
const llm = getLLMClient(process.env.LLM_PROVIDER); // 'openai' | 'anthropic' | 'google'
const analyzer = new CodeAnalyzer(llm);
```

**Supported Models:**
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **Google**: Gemini 1.5 Pro, Gemini 1.5 Flash

### CodeAnalyzer Class

```typescript
class CodeAnalyzer {
  async analyze(code: string, language: string): Promise<AnalysisResult>
  async analyzeSecurity(code: string, language: string): Promise<AnalysisResult>
  async analyzePerformance(code: string, language: string): Promise<AnalysisResult>
}
```

Each method:
1. Constructs a prompt using RCFG framework
2. Sends to LLM with system + user messages
3. Parses JSON response (handles markdown code blocks)
4. Validates with Zod schema
5. Returns type-safe `AnalysisResult`

---

## 📚 Sample Output

When you run `npm run analyze`, you'll see output like this:

```
================================================================================
CODE ANALYZER AGENT - Lab 02
================================================================================

Using LLM Provider: OPENAI

────────────────────────────────────────────────────────────────────────────────
ANALYSIS 1: General Code Analysis (Python)
────────────────────────────────────────────────────────────────────────────────

📊 Summary:
The code contains three functions handling data processing, total calculation, and
database querying. Overall, it works as intended but has room for improvements in
security, performance, and maintainability.

🐛 Issues Found: 4

1. [LOW] performance
   Line: 2
   Issue: Using 'range(len(data))' is less efficient and more error-prone than
          iterating directly over the list.
   Fix: Use 'for item in data:' instead of 'for i in range(len(data)):' and
        access item directly.

2. [HIGH] security
   Line: 16
   Issue: The code is vulnerable to SQL injection as it concatenates user input
          directly into the SQL query.
   Fix: Use parameterized queries or prepared statements to avoid SQL injection
        vulnerabilities.

3. [MEDIUM] maintainability
   Line: 14
   Issue: Importing a module within a function is generally discouraged unless
          necessary, as it can make dependency management more complex.
   Fix: Move the 'import sqlite3' statement to the top of the module.

4. [MEDIUM] performance
   Line: N/A
   Issue: The database connection is not closed, which can lead to resource leaks.
   Fix: Ensure the database connection is closed after operations are completed,
        potentially using a context manager.

💡 Suggestions: 2
1. Add error handling in the database interaction to manage cases where the
   database is inaccessible or the query fails.
2. Consider adding type hints to function definitions for better clarity and
   maintainability.

📈 Metrics:
   Complexity: low
   Readability: fair
   Test Coverage: none

────────────────────────────────────────────────────────────────────────────────
ANALYSIS 2: Security-Focused Analysis (Python)
────────────────────────────────────────────────────────────────────────────────

📊 Summary:
The code provides three functions to process data, calculate a total price from a
list of items, and retrieve a user from a database. The code is functional but
contains a critical security vulnerability related to SQL injection.

🔒 Security Issues: 1

1. [CRITICAL] security
   Line: 13
   Issue: The function `get_user` is vulnerable to SQL injection because it uses
          string interpolation to directly insert `user_id` into the SQL query
          without sanitization.
   Fix: Use parameterized queries to safely pass the user_id to the database,
        for example: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))

────────────────────────────────────────────────────────────────────────────────
ANALYSIS 3: Performance-Focused Analysis (TypeScript)
────────────────────────────────────────────────────────────────────────────────

📊 Summary:
The code includes two functions: `findUser`, which searches for a specific user by
ID from a loaded list, and `processOrders`, which processes an array of orders
asynchronously. The overall quality is straightforward but has notable performance
concerns.

⚡ Performance Issues: 2

1. [HIGH] performance
   Line: 3
   Issue: The `findUser` function calls `loadAllUsers` function, which potentially
          loads all users into memory. This can be inefficient if the user database
          is large.
   Fix: Instead of loading all users, consider implementing a method to fetch a
        user by ID directly, reducing memory usage and improving performance.

2. [MEDIUM] performance
   Line: 13
   Issue: The `processOrders` function processes each order sequentially, which can
          be slow if there are many orders.
   Fix: Consider using `Promise.all` to process all orders in parallel, taking
        advantage of concurrency.

================================================================================
✅ Analysis complete! Results saved to analysis_results.json
================================================================================
```

---

## 🧪 Testing & Validation

### Local Testing

```bash
# Run the CLI analyzer with sample code
npm run analyze

# Build TypeScript
npm run build

# Run development mode
npm run dev
```

### API Testing

Test the deployed API with PowerShell:

```powershell
$body = @{
    code = "def hello():`n    print('world')"
    language = "python"
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://code-analyzer-agent.vercel.app/api/analyze" `
  -Method POST `
  -Body $body `
  -ContentType "application/json"
```

Or with curl:

```bash
curl -X POST "https://code-analyzer-agent.vercel.app/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"code":"def test():\n    pass","language":"python"}'
```

---

## 🚀 Deployment to Vercel

### Prerequisites

- Vercel account ([sign up free](https://vercel.com/signup))
- Vercel CLI: `npm install -g vercel`

### Deploy Steps

```bash
# 1. Deploy to Vercel
vercel --prod

# 2. Set environment variables in Vercel dashboard
# Go to: https://vercel.com/[your-username]/code-analyzer-agent/settings/environment-variables

# Add these variables:
# - LLM_PROVIDER = openai (or anthropic/google)
# - OPENAI_API_KEY = sk-...
# - ANTHROPIC_API_KEY = sk-ant-... (optional)
# - GOOGLE_API_KEY = ... (optional)

# 3. Redeploy to apply environment variables
vercel --prod
```

### Configuration Files

- **vercel.json**: Routes and build configuration
- **.vercelignore**: Excludes test files and node_modules
- **api/index.ts**: Serverless function handler
- **public/index.html**: Web UI

---

## 🎓 Key Learning Outcomes

### 1. RCFG Framework (Role, Context, Format, Goal)

**Role**: "You are an expert code reviewer"
- Sets expertise level and perspective
- Influences tone and depth of analysis

**Context**: Language-specific code analysis
- Provides background: "This is Python/TypeScript code"
- Sets scope: general, security, or performance focus

**Format**: Structured JSON with Zod validation
- Explicit schema ensures consistent output
- Type-safe parsing and validation

**Goal**: "Find issues and suggest improvements"
- Clear objective for the LLM
- Specific deliverables (issues, suggestions, metrics)

### 2. Chain-of-Thought Prompting

The system prompt guides the LLM through logical steps:
1. **Understand**: What does the code do? (summary)
2. **Analyze**: What problems exist? (issues with severity/category)
3. **Recommend**: How to improve? (suggestions)
4. **Evaluate**: How is the quality? (metrics)

### 3. Structured Output Extraction

```typescript
// Zod schema enforces structure
export const AnalysisResultSchema = z.object({
  summary: z.string(),
  issues: z.array(IssueSchema),
  suggestions: z.array(z.string()),
  metrics: MetricsSchema,
});

// Type-safe parsing
const result = AnalysisResultSchema.parse(jsonResponse);
```

### 4. Multi-Provider LLM Abstraction

```typescript
// Unified interface works with any provider
const llm = getLLMClient('openai');  // or 'anthropic' | 'google'
const response = await llm.chat(messages);
```

---

## 🔧 Customization Ideas

### Analyze Your Own Code

Edit `src/index.ts` and replace the sample code:

```typescript
const MY_CODE = `
// Paste your code here
function myFunction() {
  // ...
}
`;

const result = await analyzer.analyze(MY_CODE, 'javascript');
console.log(result);
```

### Add Custom Analysis Types

Create new specialized analysis modes:

```typescript
// In prompts.ts
export const ACCESSIBILITY_FOCUS_PROMPT = `Focus on accessibility issues:
- Missing alt text
- Insufficient color contrast
- Missing ARIA labels
- Keyboard navigation problems`;

// In analyzer.ts
async analyzeAccessibility(code: string, language: string) {
  const userPrompt = `Analyze this ${language} code for accessibility:\n...\n${ACCESSIBILITY_FOCUS_PROMPT}`;
  // ... rest of implementation
}
```

### Support More Languages

Add language-specific analysis rules:

```typescript
const LANGUAGE_SPECIFIC_RULES = {
  python: 'Check PEP 8 compliance, use of typing module',
  javascript: 'Check ESLint rules, use of modern ES6+ features',
  java: 'Check Java naming conventions, use of streams vs loops',
  // ... more languages
};
```

---

## 🚧 Extension Challenges

1. **Multi-File Analysis**: Accept multiple files and analyze relationships/dependencies
2. **Diff Analysis**: Compare two versions of code and analyze changes
3. **Language Auto-Detection**: Automatically detect programming language
4. **Result Caching**: Cache analysis results to avoid redundant LLM calls
5. **Batch Processing**: Analyze entire directories or repositories
6. **Custom Rules**: Allow users to define custom analysis rules
7. **Integration**: GitHub webhook to analyze pull requests
8. **Metrics Dashboard**: Web UI showing trends over time
9. **CI/CD Integration**: Run as part of automated testing
10. **Export Formats**: Support markdown, PDF, or HTML reports

---

## 📖 Documentation Files

- **README.md** (this file): Complete project documentation
- **LAB02_DELIVERABLES.md**: Lab completion checklist and reflections
- **DEPLOYMENT.md**: Deployment guide and API documentation
- **analysis_results.json**: Sample analysis results output

---

## 🐛 Troubleshooting

### Issue: "Unknown provider" error

**Solution**: Check that `LLM_PROVIDER` in `.env` is set to `openai`, `anthropic`, or `google`

### Issue: "API key not found" error

**Solution**: Ensure the corresponding API key environment variable is set:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

### Issue: Invalid JSON from LLM

**Solution**: The parser handles markdown code blocks. If it still fails:
1. Check the LLM response in console
2. Verify the prompt includes "Return as valid JSON"
3. Try a different model (e.g., GPT-4 vs GPT-3.5)

### Issue: Deployment fails on Vercel

**Solution**: 
1. Ensure `vercel.json` is properly configured
2. Check that all dependencies are in `package.json`
3. Verify environment variables are set in Vercel dashboard
4. Check build logs: https://vercel.com/dashboard

---

## 📚 Additional Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library)
- [Vercel Deployment Documentation](https://vercel.com/docs)
- [Zod Schema Validation](https://zod.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)

---

## 💡 Key Takeaways

1. **RCFG Framework** provides a systematic approach to prompt engineering
2. **Structured output** with schema validation ensures reliable LLM responses
3. **Specialized prompts** (security, performance) yield better focused results
4. **Multi-provider support** allows flexibility and comparison across models
5. **Serverless deployment** makes AI applications accessible via web APIs

---

## ✅ Lab Completion Checklist

See [LAB02_DELIVERABLES.md](LAB02_DELIVERABLES.md) for the complete checklist.

**Core Requirements:**
- [x] TypeScript implementation complete
- [x] LLM integration working (OpenAI)
- [x] Three analysis modes (general, security, performance)
- [x] Structured JSON output with Zod validation
- [x] Sample code analyzed successfully
- [x] Results saved to `analysis_results.json`
- [x] Deployed to Vercel with working API
- [x] Web UI created for interactive use

---

**Built with Advanced Prompting Techniques | Module 2: Advanced Prompting for Engineering**

**Live Demo:** 🚀 [https://code-analyzer-agent.vercel.app](https://code-analyzer-agent.vercel.app)
```

### Add New Analysis Types

Create a new prompt in `src/prompts.ts`:

```typescript
export const READABILITY_FOCUS_PROMPT = `Focus on readability:
- Variable and function naming
- Code organization
- Comment quality
- Documentation
...`;
```

Then add a method to `CodeAnalyzer`:

```typescript
async analyzeReadability(code: string, language: string) {
  const userPrompt = `Analyze for readability...
  ${READABILITY_FOCUS_PROMPT}`;
  // ...
}
```

### Use a Different LLM Provider

In your `.env` file:
```bash
# Use OpenAI instead
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or use Google
LLM_PROVIDER=google
GOOGLE_API_KEY=...
```

---

## 📝 Deliverables

- [x] Working code analyzer with TypeScript
- [x] System prompts for different analysis types
- [x] Structured JSON output with Zod validation
- [x] Multi-provider LLM support (Anthropic, OpenAI, Google)
- [x] Example demonstrating all three analysis modes
- [ ] Test with your own code
- [ ] Customize prompts for your needs
- [ ] Compare results across different LLM providers

---

## 🚀 Extension Ideas

1. **File Input**: Accept file paths instead of hardcoded strings
2. **Diff Analysis**: Compare two versions of code
3. **Multi-file Analysis**: Analyze relationships between files
4. **Custom Rules**: Add project-specific coding standards
5. **API Version**: Build a REST API (like the lab examples)
6. **CLI Tool**: Create a command-line interface
7. **IDE Integration**: VS Code extension

---

## 📚 Related Resources

- [Day 2 Curriculum](../../Training_Content/curriculum/day2-prompting.md)
- [Lab 02 Full Labs](../../Training_Content/labs/lab02-code-analyzer-agent/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

---

**Part of AI Training Program - Module 2: Advanced Prompting**
