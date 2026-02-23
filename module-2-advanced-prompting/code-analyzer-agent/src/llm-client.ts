/**
 * LLM client abstraction supporting multiple providers
 */
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import type { Message, LLMProvider } from './types.js';

/**
 * Abstract base class for LLM clients
 */
export abstract class LLMClient {
  abstract chat(messages: Message[]): Promise<string>;
}

/**
 * Anthropic Claude client
 */
export class AnthropicClient extends LLMClient {
  private client: Anthropic;
  private model: string;

  constructor(model: string = 'claude-3-5-sonnet-20241022') {
    super();
    this.client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    this.model = model;
  }

  async chat(messages: Message[]): Promise<string> {
    // Extract system message and user/assistant messages
    const systemMessage = messages.find((m) => m.role === 'system');
    const conversationMessages = messages.filter((m) => m.role !== 'system');

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      system: systemMessage?.content,
      messages: conversationMessages.map((m) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
    });

    return response.content[0].type === 'text' ? response.content[0].text : '';
  }
}

/**
 * OpenAI client (GPT-4, GPT-4o, etc.)
 */
export class OpenAIClient extends LLMClient {
  private client: OpenAI;
  private model: string;

  constructor(model: string = 'gpt-4o') {
    super();
    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    this.model = model;
  }

  async chat(messages: Message[]): Promise<string> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
    });

    return response.choices[0].message.content || '';
  }
}

/**
 * Google Generative AI client (Gemini)
 */
export class GoogleClient extends LLMClient {
  private model: any;

  constructor(modelName: string = 'gemini-1.5-flash') {
    super();
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
      throw new Error('GOOGLE_API_KEY environment variable is required');
    }
    const genAI = new GoogleGenerativeAI(apiKey);
    this.model = genAI.getGenerativeModel({ model: modelName });
  }

  async chat(messages: Message[]): Promise<string> {
    // Extract system and build prompt
    const systemMessage = messages.find((m) => m.role === 'system');
    const userMessage = messages.find((m) => m.role === 'user');

    let prompt = '';
    if (systemMessage) {
      prompt += systemMessage.content + '\n\n';
    }
    if (userMessage) {
      prompt += userMessage.content;
    }

    const result = await this.model.generateContent(prompt);
    const response = await result.response;
    return response.text();
  }
}

/**
 * Factory function to create LLM client based on provider
 */
export function getLLMClient(provider: LLMProvider = 'anthropic'): LLMClient {
  switch (provider) {
    case 'anthropic':
      return new AnthropicClient();
    case 'openai':
      return new OpenAIClient();
    case 'google':
      return new GoogleClient();
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}
