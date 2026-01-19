import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from llama_index.core.schema import TextNode
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
import time
import pickle

load_dotenv()

class PineconeManager:
    """Manages Pinecone vector database operations with hybrid search (dense + sparse)"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "grc-policies")
        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        self.bm25_encoder = BM25Encoder()
        self.bm25_fitted = False
        self.bm25_path = "data/bm25_encoder.pkl"

        # Try to load BM25 encoder
        loaded = self._load_bm25()
        
        print("Pinecone, BM25Encoder and OpenAI clients initialized")
        if loaded:
            print("BM25 encoder loaded from disk")
        else:
            print("BM25 encoder not found. Run 'python src/populate_index.py' first.")
    
    def _save_bm25(self):
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok = True)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25_encoder, f)
        print(f"BM25 encoder saved to {self.bm25_path}")

    def _load_bm25(self):
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, 'rb') as f:
                self.bm25_encoder = pickle.load(f)
            self.bm25_fitted = True
            return True
        return False

    def create_index_if_not_exists(self):
        print(f"\nChecking if index '{self.index_name}' exists...")
        
        if self.index_name in self.pc.list_indexes().names():
            print(f"Index '{self.index_name}' already exists")
            return
        
        print(f"Creating new index '{self.index_name}'...")
        print(f"Dimension: {self.embedding_dimension}")
        print(f"Metric: dotproduct (for hybrid search)")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=self.embedding_dimension,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        print("Waiting for index to be ready...")
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        print(f"Index '{self.index_name}' created successfully!")
    
    def generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"Generating dense embeddings for {len(texts)} texts using OpenAI...")
        
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        print(f"Generated {len(embeddings)} dense embeddings")
        return embeddings
    
    def fit_bm25(self, texts: List[str]):
        print(f"Fitting BM25 encoder on {len(texts)} documents...")
        self.bm25_encoder.fit(texts)
        self.bm25_fitted = True
        self._save_bm25()
        print("BM25 encoder fitted and saved")

    def generate_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.bm25_fitted:
            raise ValueError("BM25 encoder not fitted. Call fit_bm25() first.")
        
        print(f"Generating sparse embeddings for {len(texts)} texts...")
        sparse_embeddings = self.bm25_encoder.encode_documents(texts)
        print(f"Generated {len(sparse_embeddings)} sparse embeddings")
        return sparse_embeddings

    def upload_nodes_to_pinecone(self, nodes: List[TextNode], batch_size: int = 100):
        print(f"\nUploading {len(nodes)} nodes to Pinecone...")
        print(f"Batch size: {batch_size}")
        
        index = self.pc.Index(self.index_name)
        texts = [node.get_content() for node in nodes]

        self.fit_bm25(texts)
        dense_embeddings = self.generate_dense_embeddings(texts)
        sparse_embeddings = self.generate_sparse_embeddings(texts)
        
        vectors = []
        for i, (node, dense_emb, sparse_emb) in enumerate(zip(nodes, dense_embeddings, sparse_embeddings)):
            vector = {
                "id": node.node_id,
                "values": dense_emb,
                "sparse_values": sparse_emb,
                "metadata": {
                    "text": node.get_content(),
                    **node.metadata
                }
            }
            vectors.append(vector)
        
        print("\nUploading to Pinecone...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        
        print(f"\nSuccessfully uploaded {len(vectors)} hybrid vectors to Pinecone!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25_fitted:
            raise ValueError(
                "BM25 encoder not fitted.\n"
                "Please run: python src/populate_index.py\n"
                "This will fit the BM25 encoder on your document corpus."
            )
        
        print(f"\n Hybrid search: '{query}'")
        
        dense_query = self.generate_dense_embeddings([query])[0]
        sparse_query = self.bm25_encoder.encode_queries([query])[0]
        
        index = self.pc.Index(self.index_name)
        results = index.query(
            vector=dense_query,
            sparse_vector=sparse_query,
            top_k=top_k,
            include_metadata=True
        )
        print(f"Found {len(results['matches'])} results")
        return results['matches']
    
    def search_semantic_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"\n Semantic-only search: '{query}'")
        
        dense_query = self.generate_dense_embeddings([query])[0]
        
        index = self.pc.Index(self.index_name)
        results = index.query(
            vector=dense_query,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"Found {len(results['matches'])} results")
        return results['matches']
    
    def search_keyword_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25_fitted:
            raise ValueError("BM25 encoder not fitted.")
        
        print(f"\n Keyword-only search: '{query}'")
        
        sparse_query = self.bm25_encoder.encode_queries([query])[0]
        
        # For sparse-only, we need a zero dense vector
        zero_dense = [0.0] * self.embedding_dimension
        
        index = self.pc.Index(self.index_name)
        results = index.query(
            vector=zero_dense,
            sparse_vector=sparse_query,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"Found {len(results['matches'])} results")
        return results['matches']