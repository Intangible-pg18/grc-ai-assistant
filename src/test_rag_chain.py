"""
Test RAG chain with various GRC queries
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinecone_manager import PineconeManager
from rag_chain import GRCRAGChain

def test_rag_chain():
    # Initialize components
    print("🚀 Initializing RAG Chain...")
    retriever = PineconeManager()
    rag_chain = GRCRAGChain(retriever)
    
    # Test queries
    test_questions = [
        "What are the GDPR Article 33 notification requirements for data breaches?",
        "What is the timeline for notifying supervisory authorities about a data breach?",
        "Who should be notified in case of a security incident?",
    ]
    
    # Test single query with full details
    print("\n" + "█"*80)
    print("SINGLE QUERY TEST (with sources)")
    print("█"*80)
    
    result = rag_chain.query(
        test_questions[0],
        top_k=3,
        return_sources=True
    )
    
    print(f"\n💬 ANSWER:")
    print(result['answer'])
    
    print(f"\n📚 SOURCES:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['file_name']} (Score: {source['score']:.4f})")
        print(f"   {source['text']}")
    
    print(f"\n📊 METRICS:")
    for key, value in result['metrics'].items():
        print(f"   {key}: {value}")
    
    # Test batch queries
    print("\n" + "█"*80)
    print("BATCH QUERY TEST")
    print("█"*80)
    
    batch_results = rag_chain.batch_query(test_questions, top_k=3)
    
    print("\n📋 BATCH RESULTS SUMMARY:")
    for i, (question, result) in enumerate(zip(test_questions, batch_results), 1):
        print(f"\n{i}. Q: {question}")
        print(f"   A: {result['answer'][:150]}...")
        print(f"   Latency: {result['metrics']['total_latency_ms']:.2f}ms")
        print(f"   Tokens: {result['metrics']['total_tokens']}")

if __name__ == "__main__":
    test_rag_chain()