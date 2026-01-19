import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from llama_index.core.schema import TextNode
from dotenv import load_dotenv
import time

load_dotenv()

class PineconeManager:
    """Manages Pinecone vector database operations with OpenAI embeddings"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "grc-policies")
        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        
        print("Pinecone and OpenAI clients initialized")
    
    def create_index_if_not_exists(self):
        print(f"\nChecking if index '{self.index_name}' exists...")
        
        if self.index_name in self.pc.list_indexes().names():
            print(f"Index '{self.index_name}' already exists")
            return
        
        print(f"Creating new index '{self.index_name}'...")
        print(f"Dimension: {self.embedding_dimension}")
        print(f"Metric: dotproduct (required for hybrid search)")
        
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
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"Generating embeddings for {len(texts)} texts using OpenAI...")
        
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def upload_nodes_to_pinecone(self, nodes: List[TextNode], batch_size: int = 100):
        print(f"\nUploading {len(nodes)} nodes to Pinecone...")
        print(f"Batch size: {batch_size}")
        
        index = self.pc.Index(self.index_name)
        texts = [node.get_content() for node in nodes]
        embeddings = self.generate_embeddings(texts)
        
        vectors = []
        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            vector = {
                "id": node.node_id,
                "values": embedding,
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
        
        print(f"\nSuccessfully uploaded {len(vectors)} vectors to Pinecone!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.generate_embeddings([query])[0]
        
        index = self.pc.Index(self.index_name)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']