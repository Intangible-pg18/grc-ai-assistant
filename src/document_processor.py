"""
Document loading and chunking using LlamaIndex
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

load_dotenv()

class DocumentProcessor:
    """Process GRC policy documents into chunks"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = SentenceSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
    
    def load_documents(self, data_dir: str) -> List[Document]:
        print(f"Loading documents from {data_dir}...")

        reader = SimpleDirectoryReader(
            input_dir = data_dir,
            recursive = True,
            required_exts=[".txt"]
        )

        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        print(f"Chunking documents (size={self.chunk_size}, overlap = {self.chunk_overlap})...")
        nodes = self.splitter.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} chunks")

        if nodes:
            print(f"\n Sample chunk:")
            print(f"Text: {nodes[0].text[:200]}...")
            print(f"Metadata: {nodes[0].metadata}")
        
        return nodes

    def process_directory(self, data_dir: str) -> List[TextNode]:
        documents = self.load_documents(data_dir)
        nodes = self.chunk_documents(documents)
        return nodes
    
if __name__ == "__main__":
    processor = DocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
    )

    policy_nodes = processor.process_directory("data/policies")
    reg_nodes = processor.process_directory("data/regulations")
    print(f"\n Total nodes: {len(policy_nodes) + len(reg_nodes)}")