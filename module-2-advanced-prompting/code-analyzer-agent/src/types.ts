/**
 * Type definitions and Zod schemas for Code Analyzer Agent
 */
import { z } from 'zod';

// Zod Schemas for validation
export const IssueSchema = z.object({
  severity: z.enum(['critical', 'high', 'medium', 'low']),
  line: z.number().nullable(),
  category: z.enum(['bug', 'security', 'performance', 'style', 'maintainability']),
  description: z.string(),
  suggestion: z.string(),
});

export const MetricsSchema = z.object({
  complexity: z.enum(['low', 'medium', 'high']),
  readability: z.enum(['poor', 'fair', 'good', 'excellent']),
  test_coverage_estimate: z.enum(['none', 'partial', 'good']),
});

export const AnalysisResultSchema = z.object({
  summary: z.string(),
  issues: z.array(IssueSchema),
  suggestions: z.array(z.string()),
  metrics: MetricsSchema,
});

// TypeScript types inferred from schemas
export type Issue = z.infer<typeof IssueSchema>;
export type Metrics = z.infer<typeof MetricsSchema>;
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;

// Message structure for LLM API calls
export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

// LLM Provider types
export type LLMProvider = 'anthropic' | 'openai' | 'google';
