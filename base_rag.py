"""
Base RAG System: Document retrieval and response generation
"""

import os
import logging
from typing import List, Dict, Optional
import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore
import numpy as np # type: ignore
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentStore:
    """Handles document storage and retrieval using FAISS."""
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Initializing DocumentStore with {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.index = None
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def add_documents(self, documents: List[str]):
        """Add documents to the store."""
        logger.info(f"Adding {len(documents)} documents to store")
        
        if not documents:
            logger.warning("No documents provided")
            return
        
        # Store documents
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Total documents in store: {len(self.documents)}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """Retrieve most relevant documents for a query."""
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in store")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        k = min(top_k, len(self.documents))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(1 / (1 + distance)),  # Convert distance to similarity
                    'index': int(idx)
                })
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def save(self, path: str):
        """Save the document store."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save documents
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Document store saved to {path}")
    
    def load(self, path: str):
        """Load the document store."""
        # Load FAISS index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load documents
        docs_path = os.path.join(path, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
        
        logger.info(f"Document store loaded from {path}")


class BaseRAG:
    """Base RAG system for document retrieval and response generation."""
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        logger.info(f"Initializing BaseRAG with model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize document store
        self.doc_store = DocumentStore(embedding_model)
        
        # Initialize generation model
        logger.info("Loading generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use pipeline for easier generation
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        logger.info("BaseRAG initialized successfully")
    
    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base."""
        self.doc_store.add_documents(documents)
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context for a query."""
        results = self.doc_store.retrieve(query, top_k)
        
        if not results:
            return "No relevant context found."
        
        # Concatenate top documents
        context = "\n\n".join([
            f"[Document {i+1}]: {result['document']}"
            for i, result in enumerate(results)
        ])
        
        return context
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Generate a response to a query using retrieved context.
        
        Args:
            query: User query
            context: Optional pre-retrieved context
            top_k: Number of documents to retrieve if context not provided
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve context if not provided
        if context is None:
            context = self.retrieve_context(query, top_k)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        logger.info("Generating response...")
        outputs = self.generator(
            prompt,
            max_length=512,
            num_return_sequences=1
        )
        
        response = outputs[0]['generated_text']
        
        return {
            'response': response,
            'context': context,
            'query': query
        }
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the generation model."""
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def save(self, path: str):
        """Save the RAG system."""
        self.doc_store.save(path)
    
    def load(self, path: str):
        """Load the RAG system."""
        self.doc_store.load(path)