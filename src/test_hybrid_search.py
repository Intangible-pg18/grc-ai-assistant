import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinecone_manager import PineconeManager

def print_results(results, search_type):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"{search_type} RESULTS")
    print('='*80)
    
    if not results:
        print("❌ No results found")
        return
    
    for i, match in enumerate(results, 1):
        print(f"\n{i}. SCORE: {match['score']:.4f}")
        print(f"   SOURCE: {match['metadata'].get('file_name', 'Unknown')}")
        print(f"   TEXT: {match['metadata']['text'][:200]}...")
        print()

def test_queries():
    manager = PineconeManager()
    
    # Test queries designed to show differences
    test_cases = [
        {
            "query": "GDPR Article 33 notification requirements",
            "description": "Exact regulatory citation - should favor keyword matching"
        },
        {
            "query": "What should we do when customer data is accidentally exposed?",
            "description": "Natural language question - should favor semantic matching"
        },
        {
            "query": "72 hours breach notification timeline",
            "description": "Mixed: specific timeline (keyword) + concept (semantic)"
        }
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        desc = test_case["description"]
        
        print("\n" + "█"*80)
        print(f"TEST QUERY: {query}")
        print(f"RATIONALE: {desc}")
        print("█"*80)
        
        # 1. Hybrid Search (uses both)
        hybrid_results = manager.search(query, top_k=3)
        print_results(hybrid_results, "🔄 HYBRID (Dense + Sparse)")
        
        # 2. Semantic Only
        semantic_results = manager.search_semantic_only(query, top_k=3)
        print_results(semantic_results, "🧠 SEMANTIC ONLY (Dense)")
        
        # 3. Keyword Only
        keyword_results = manager.search_keyword_only(query, top_k=3)
        print_results(keyword_results, "🔑 KEYWORD ONLY (BM25)")
        
        # Analysis
        print("\n" + "─"*80)
        print("📊 ANALYSIS:")
        
        # Compare top results
        if hybrid_results and semantic_results and keyword_results:
            hybrid_top = hybrid_results[0]['metadata'].get('file_name', 'Unknown')
            semantic_top = semantic_results[0]['metadata'].get('file_name', 'Unknown')
            keyword_top = keyword_results[0]['metadata'].get('file_name', 'Unknown')
            
            print(f"   Hybrid picked: {hybrid_top}")
            print(f"   Semantic picked: {semantic_top}")
            print(f"   Keyword picked: {keyword_top}")
            
            if hybrid_top == semantic_top and hybrid_top == keyword_top:
                print("   ✅ All methods agree - strong consensus")
            elif hybrid_top == semantic_top:
                print("   ⚡ Hybrid followed semantic similarity")
            elif hybrid_top == keyword_top:
                print("   ⚡ Hybrid followed keyword matching")
            else:
                print("   🎯 Hybrid found unique best result combining both signals")
        
        print("─"*80)
        input("\n⏸️  Press ENTER to continue to next test...\n")

if __name__ == "__main__":
    test_queries()