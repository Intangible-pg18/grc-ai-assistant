"""
Script to populate Pinecone index with document embeddings
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from pinecone_manager import PineconeManager

def main():
    load_dotenv()
    
    print("[1/3] Loading documents...")
    processor = DocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
    )
    
    policy_nodes = processor.process_directory("data/policies")
    reg_nodes = processor.process_directory("data/regulations")
    all_nodes = policy_nodes + reg_nodes
    
    print(f"Loaded {len(all_nodes)} total text nodes")
    print(f"Policies: {len(policy_nodes)} nodes")
    print(f"Regulations: {len(reg_nodes)} nodes\n")
    
    print("[2/3] Setting up Pinecone...")
    manager = PineconeManager()
    manager.create_index_if_not_exists()
    
    print("\n[3/3] Uploading to Pinecone...")
    manager.upload_nodes_to_pinecone(all_nodes, batch_size=100)
    
    print("\nPipeline complete! Your GRC knowledge base is ready.")
    print(f"Index: {manager.index_name}")
    print(f"Vectors: {len(all_nodes)}")
    print(f"Embedding model: {manager.embedding_model}")
    print(f"Dimensions: {manager.embedding_dimension}")

if __name__ == "__main__":
    main()