import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getLLMClient } from '../src/llm-client.js';
import { CodeAnalyzer } from '../src/analyzer.js';
import type { LLMProvider } from '../src/types.js';

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Credentials': 'true',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,OPTIONS,PATCH,DELETE,POST,PUT',
  'Access-Control-Allow-Headers': 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version',
};

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).json({ message: 'OK' });
  }

  // Root endpoint - documentation
  if (req.method === 'GET' && req.url === '/') {
    return res.status(200).json({
      name: 'Code Analyzer API',
      version: '1.0.0',
      endpoints: {
        'POST /api/analyze': {
          description: 'Analyze code for general issues',
          body: {
            code: 'string (required)',
            language: 'string (optional, default: python)',
          },
        },
        'POST /api/analyze/security': {
          description: 'Analyze code for security vulnerabilities',
          body: {
            code: 'string (required)',
            language: 'string (optional, default: python)',
          },
        },
        'POST /api/analyze/performance': {
          description: 'Analyze code for performance issues',
          body: {
            code: 'string (required)',
            language: 'string (optional, default: python)',
          },
        },
      },
    });
  }

  // Parse request body
  const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  const { code, language = 'python' } = body || {};

  if (!code) {
    return res.status(400).json({
      error: 'Missing required field: code',
    });
  }

  try {
    // Initialize LLM client
    const provider = (process.env.LLM_PROVIDER as LLMProvider) || 'openai';
    const llm = getLLMClient(provider);
    const analyzer = new CodeAnalyzer(llm);

    // Determine analysis type from path
    const path = req.url || '';
    let result;

    if (path.includes('/security')) {
      result = await analyzer.analyzeSecurity(code, language);
    } else if (path.includes('/performance')) {
      result = await analyzer.analyzePerformance(code, language);
    } else {
      result = await analyzer.analyze(code, language);
    }

    return res.status(200).json({
      success: true,
      analysis: result,
    });
  } catch (error: any) {
    console.error('Analysis error:', error);
    return res.status(500).json({
      error: 'Analysis failed',
      message: error.message || 'Unknown error',
    });
  }
}
