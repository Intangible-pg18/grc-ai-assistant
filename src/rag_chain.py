"""
RAG chain with AWS Bedrock, LangSmith tracing, and performance metrics
"""
import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import traceable
import json

load_dotenv()

class TokenCounterCallback(BaseCallbackHandler):
    """Callback to capture token usage from Bedrock"""
    
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        """Extract tokens from LLM response metadat"""
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('usage', {})
            self.input_tokens = usage.get('input_tokens', 0)
            self.output_tokens = usage.get('output_tokens', 0)

class GRCRAGChain:
    def __init__(self, pinecone_manager):
        self.retriever = pinecone_manager

        self.llm = ChatBedrock(
            model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
            region_name = os.getenv("AWS_REGION", "ap-south-1"),
            model_kwargs={
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_p": 0.9
            }
        )

        self.prompt = self._build_prompt_template()
        self.chain = self._build_chain()
        print("RAG Chain initialized")
        print(f"Model: {os.getenv('BEDROCK_MODEL_ID')}")
        print(f"Region: {os.getenv('AWS_REGION')}")
        print(f"Tracing: LangSmith enabled")

    def _build_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = """You are a GRC (Governance, Risk and Compliance) AI assistant for Diligent One platform.
        Your role is to answer questions about organizational policies, regulatory requirements and compliance obligations based ONLY on the provided context documents.
        CRITICAL INSTRUCTIONS:
        1. Answer ONLY using information from the context provided below
        2. If the context doesn't contain enough information, say "I don't have enough information in the policies to answer that"
        3. Always cite your sources using the format: [Source: POLICY-ID or REG-ID]
        4. For regulatory questions, cite specific articles/sections when mentioned
        5. Be precise and factual - this is for compliance purposes
        6. If multiple policies are relevant, mention all of them
        7. Use professional, clear language appropriate for compliance officers
        CONTEXT DOCUMENTS:
        {context}
        Remember: Only use information from the context above. Do not use external knowledge."""

        user_prompt = """Question: {question}
        Please provide a detailed answer based on the context documents provided, including relevant policy citations."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    
    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string with metadata"""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk['metadata']
            text = metadata['text']
            source = metadata.get('file_name', 'Unknown')
            score = chunk['score']

            context_parts.append(
                f"[{i}] SOURCE: {source} | Relevance Score: {score:.3f}\n"
                f"{text}\n"
            )
        return "\n".join(context_parts)
    
    def _build_chain(self):
        LCEL_chain = (
            {
                "context": lambda x: self._format_context(x["retrieved_chunks"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return LCEL_chain

    @traceable(name="hybrid_retieval")
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using hybrid search"""
        start_time = time.time()

        results = self.retriever.search(query, top_k = top_k)
        retrieval_latency = time.time() - start_time

        print(f"\n Retrieval Metrics:")
        print(f"Latency: {retrieval_latency*1000:.2f}ms")
        print(f"Chunks retrieved: {len(results)}")
        if results:
            print(f"Top score: {results[0]['score']:.4f}")
        return results
    
    @traceable(name="rag_query")
    def query(self, question: str, top_k: int = 5, return_sources: bool = True) -> Dict[str, Any]:
        """Execute full RAG pipeline"""
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print('='*80)

        total_start = time.time()
        retrieved_chunks = self.retrieve(question, top_k = top_k)

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
                "metrics": {
                    "total_latency_ms": 0,
                    "retrieval_latency_ms": 0,
                    "llm_latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
        retrieval_time = time.time() - total_start

        llm_start = time.time()
        token_callback = TokenCounterCallback()
        answer = self.chain.invoke({
            "retrieved_chunks": retrieved_chunks,
            "question": question
            }, 
            config = {"callbacks": [token_callback]}
        )

        llm_time = time.time() - llm_start
        total_time = time.time() - total_start

        actual_input_tokens = token_callback.input_tokens
        actual_output_tokens = token_callback.output_tokens

        metrics = {
            "total_latency_ms": total_time * 1000,
            "retrieval_latency_ms": retrieval_time * 1000,
            "llm_latency_ms": llm_time * 1000,
            "input_tokens": actual_input_tokens,
            "output_tokens": actual_output_tokens,
            "total_tokens": actual_input_tokens + actual_output_tokens,
            "chunks_retrieved": len(retrieved_chunks)
        }

        print(f"\n📈 LLM Metrics:")
        print(f"   Latency: {llm_time*1000:.2f}ms")
        print(f"   Input tokens: {actual_input_tokens}")
        print(f"   Output tokens: {actual_output_tokens}")
        print(f"   Total tokens: {metrics['total_tokens']}")

        response = {
            "answer": answer,
            "metrics": metrics
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "file_name": chunk['metadata'].get('file_name', 'Unknown'),
                    "score": chunk['score'],
                    "text": chunk['metadata']['text'][:300] + "..."
                }
                for chunk in retrieved_chunks
            ]
        
        return response

    @traceable(name = "batch_query")
    def batch_query(self, questions: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        results = []
        print(f"\nProcessing {len(questions)} questions in batch...")
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]")
            result = self.query(question, top_k = top_k, return_sources = False)
            results.append(result)

        total_tokens = sum(r['metrics']['total_tokens'] for r in results)
        avg_latency = sum(r['metrics']['total_latency_ms'] for r in results) / len(results)

        print(f"\n Batch Metrics:")
        print(f"Questions processed: {len(questions)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average latency: {avg_latency:.2f}ms")
        
        return results