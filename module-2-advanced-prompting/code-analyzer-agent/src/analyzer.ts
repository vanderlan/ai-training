/**
 * Code Analyzer implementation
 */
import type { LLMClient } from './llm-client.js';
import type { AnalysisResult, Message } from './types.js';
import { AnalysisResultSchema } from './types.js';
import {
  CODE_ANALYZER_SYSTEM,
  SECURITY_FOCUS_PROMPT,
  PERFORMANCE_FOCUS_PROMPT,
} from './prompts.js';

/**
 * CodeAnalyzer class - analyzes code using LLM
 */
export class CodeAnalyzer {
  private llm: LLMClient;
  private systemPrompt: string;

  constructor(llmClient: LLMClient) {
    this.llm = llmClient;
    this.systemPrompt = CODE_ANALYZER_SYSTEM;
  }

  /**
   * Analyze code and return structured result
   */
  async analyze(code: string, language: string = 'python'): Promise<AnalysisResult> {
    const userPrompt = `Analyze this ${language} code:

\`\`\`${language}
${code}
\`\`\`

Return your analysis as JSON.`;

    const messages: Message[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userPrompt },
    ];

    const response = await this.llm.chat(messages);
    return this.parseResponse(response);
  }

  /**
   * Security-focused analysis
   */
  async analyzeSecurity(code: string, language: string = 'python'): Promise<AnalysisResult> {
    const userPrompt = `Analyze this ${language} code for security vulnerabilities:

\`\`\`${language}
${code}
\`\`\`

${SECURITY_FOCUS_PROMPT}`;

    const messages: Message[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userPrompt },
    ];

    const response = await this.llm.chat(messages);
    return this.parseResponse(response);
  }

  /**
   * Performance-focused analysis
   */
  async analyzePerformance(code: string, language: string = 'python'): Promise<AnalysisResult> {
    const userPrompt = `Analyze this ${language} code for performance issues:

\`\`\`${language}
${code}
\`\`\`

${PERFORMANCE_FOCUS_PROMPT}`;

    const messages: Message[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userPrompt },
    ];

    const response = await this.llm.chat(messages);
    return this.parseResponse(response);
  }

  /**
   * Parse LLM response into structured result
   */
  private parseResponse(response: string): AnalysisResult {
    let jsonStr = response.trim();

    // Handle markdown code blocks
    if (jsonStr.includes('```json')) {
      const parts = jsonStr.split('```json');
      if (parts.length > 1) {
        jsonStr = parts[1].split('```')[0];
      }
    } else if (jsonStr.includes('```')) {
      const parts = jsonStr.split('```');
      if (parts.length > 1) {
        jsonStr = parts[1].split('```')[0];
      }
    }

    // Parse and validate with Zod
    const data = JSON.parse(jsonStr.trim());
    return AnalysisResultSchema.parse(data);
  }
}
