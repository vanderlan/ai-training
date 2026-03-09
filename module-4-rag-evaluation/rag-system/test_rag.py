"""Test script for RAG system."""
import json
from dotenv import load_dotenv
from src.llm_client import get_llm_client
from src import CodebaseRAG, RAGEvaluator, create_eval_dataset

load_dotenv()


def test_indexing():
    """Test indexing sample code."""
    print("=" * 60)
    print("Testing Indexing")
    print("=" * 60)
    
    llm = get_llm_client("anthropic")
    rag = CodebaseRAG(llm, collection_name="test_codebase")
    
    # Clear any existing index
    rag.clear_index()
    
    # Index sample code directory
    count = rag.index_directory("./data/sample_code")
    print(f"\n✓ Indexed {count} chunks")
    
    # Get stats
    stats = rag.get_stats()
    print(f"✓ Collection stats: {stats}")
    
    return rag


def test_querying(rag: CodebaseRAG):
    """Test querying the indexed code."""
    print("\n" + "=" * 60)
    print("Testing Queries")
    print("=" * 60)
    
    test_questions = [
        "How do I load data from a file?",
        "How can I create a new user?",
        "How do I make a POST request to an API?"
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        result = rag.query(question, n_results=3)
        print(f"💡 Answer: {result['answer'][:200]}...")
        print(f"📚 Sources: {len(result['sources'])} files")
        for source in result['sources']:
            print(f"   - {source['file']} ({source['name']}) - relevance: {source['relevance']}")


def test_evaluation(rag: CodebaseRAG):
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("Testing Evaluation")
    print("=" * 60)
    
    # Load test queries
    with open("./data/test_queries.json", "r") as f:
        test_data = json.load(f)
    
    examples = create_eval_dataset(test_data[:3])  # Use first 3 examples
    
    llm = get_llm_client("anthropic")
    evaluator = RAGEvaluator(rag, llm)
    
    # Evaluate retrieval
    print("\n📊 Retrieval Metrics:")
    retrieval_metrics = evaluator.evaluate_retrieval(examples, k=5)
    for metric, value in retrieval_metrics.items():
        print(f"   {metric}: {value}")
    
    # Evaluate generation
    print("\n📊 Generation Metrics:")
    generation_metrics = evaluator.evaluate_generation(examples)
    for metric, value in generation_metrics.items():
        print(f"   {metric}: {value}")


if __name__ == "__main__":
    print("\n🚀 RAG System Test Suite\n")
    
    # Test indexing
    rag = test_indexing()
    
    # Test querying
    test_querying(rag)
    
    # Test evaluation
    test_evaluation(rag)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
