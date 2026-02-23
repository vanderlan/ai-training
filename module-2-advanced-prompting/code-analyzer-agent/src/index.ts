/**
 * Code Analyzer Agent - Main Entry Point
 * Lab 02: Advanced Prompting for Engineering
 */
import 'dotenv/config';
import { getLLMClient } from './llm-client.js';
import { CodeAnalyzer } from './analyzer.js';
import type { LLMProvider } from './types.js';
import { writeFileSync } from 'fs';

// Sample code to analyze
const SAMPLE_CODE = `
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result

def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price'] * item['quantity']
    return total

def get_user(user_id):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
`;

const SAMPLE_CODE_TS = `
function findUser(userId: string) {
  const users = loadAllUsers();
  for (let i = 0; i < users.length; i++) {
    if (users[i].id === userId) {
      return users[i];
    }
  }
  return null;
}

async function processOrders(orders: any[]) {
  const results = [];
  for (const order of orders) {
    const result = await processOrder(order);
    results.push(result);
  }
  return results;
}
`;

/**
 * Main function to run code analysis
 */
async function main() {
  console.log('='.repeat(80));
  console.log('CODE ANALYZER AGENT - Lab 02');
  console.log('='.repeat(80));
  console.log();

  // Get LLM provider from environment or default to anthropic
  const provider = (process.env.LLM_PROVIDER as LLMProvider) || 'anthropic';
  console.log(`Using LLM Provider: ${provider.toUpperCase()}`);
  console.log();

  try {
    // Initialize LLM client and analyzer
    const llm = getLLMClient(provider);
    const analyzer = new CodeAnalyzer(llm);

    // Analysis 1: General Analysis
    console.log('─'.repeat(80));
    console.log('ANALYSIS 1: General Code Analysis (Python)');
    console.log('─'.repeat(80));
    const result1 = await analyzer.analyze(SAMPLE_CODE, 'python');
    console.log('\n📊 Summary:');
    console.log(result1.summary);
    console.log('\n🐛 Issues Found:', result1.issues.length);
    result1.issues.forEach((issue, index) => {
      console.log(`\n${index + 1}. [${issue.severity.toUpperCase()}] ${issue.category}`);
      console.log(`   Line: ${issue.line || 'N/A'}`);
      console.log(`   Issue: ${issue.description}`);
      console.log(`   Fix: ${issue.suggestion}`);
    });
    console.log('\n💡 Suggestions:', result1.suggestions.length);
    result1.suggestions.forEach((suggestion, index) => {
      console.log(`${index + 1}. ${suggestion}`);
    });
    console.log('\n📈 Metrics:');
    console.log(`   Complexity: ${result1.metrics.complexity}`);
    console.log(`   Readability: ${result1.metrics.readability}`);
    console.log(`   Test Coverage: ${result1.metrics.test_coverage_estimate}`);

    // Analysis 2: Security Focus
    console.log('\n\n' + '─'.repeat(80));
    console.log('ANALYSIS 2: Security-Focused Analysis (Python)');
    console.log('─'.repeat(80));
    const result2 = await analyzer.analyzeSecurity(SAMPLE_CODE, 'python');
    console.log('\n📊 Summary:');
    console.log(result2.summary);
    console.log('\n🔒 Security Issues:', result2.issues.length);
    result2.issues.forEach((issue, index) => {
      console.log(`\n${index + 1}. [${issue.severity.toUpperCase()}] ${issue.category}`);
      console.log(`   Line: ${issue.line || 'N/A'}`);
      console.log(`   Issue: ${issue.description}`);
      console.log(`   Fix: ${issue.suggestion}`);
    });

    // Analysis 3: Performance Focus (TypeScript)
    console.log('\n\n' + '─'.repeat(80));
    console.log('ANALYSIS 3: Performance-Focused Analysis (TypeScript)');
    console.log('─'.repeat(80));
    const result3 = await analyzer.analyzePerformance(SAMPLE_CODE_TS, 'typescript');
    console.log('\n📊 Summary:');
    console.log(result3.summary);
    console.log('\n⚡ Performance Issues:', result3.issues.length);
    result3.issues.forEach((issue, index) => {
      console.log(`\n${index + 1}. [${issue.severity.toUpperCase()}] ${issue.category}`);
      console.log(`   Line: ${issue.line || 'N/A'}`);
      console.log(`   Issue: ${issue.description}`);
      console.log(`   Fix: ${issue.suggestion}`);
    });

    // Save results to JSON file
    const results = {
      timestamp: new Date().toISOString(),
      provider,
      analyses: [
        { type: 'general', language: 'python', result: result1 },
        { type: 'security', language: 'python', result: result2 },
        { type: 'performance', language: 'typescript', result: result3 },
      ],
    };

    writeFileSync(
      'analysis_results.json',
      JSON.stringify(results, null, 2),
      'utf-8'
    );

    console.log('\n\n' + '='.repeat(80));
    console.log('✅ Analysis complete! Results saved to analysis_results.json');
    console.log('='.repeat(80));
  } catch (error) {
    console.error('\n❌ Error during analysis:');
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

// Run main function
main().catch(console.error);
