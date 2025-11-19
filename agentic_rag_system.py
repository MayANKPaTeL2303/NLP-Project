"""
Agentic RAG System: Main orchestrator with iterative refinement loop
"""

import logging
from typing import Dict, List, Optional
import time
from datetime import datetime

from base_rag import BaseRAG
from slm_evaluator import SLMEvaluator
from agentic_tools import AgenticRAGTools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticRAGSystem:
    """
    Orchestrates the complete Agentic RAG system with:
    1. Base RAG for initial response generation
    2. SLM Evaluator for quality scoring
    3. Agentic loop for iterative refinement
    """
    
    def __init__(
        self,
        base_model: str = "google/flan-t5-base",
        evaluator_model: str = "microsoft/deberta-v3-small",
        score_threshold: float = 7.0,
        max_iterations: int = 3
    ):
        """
        Initialize the Agentic RAG System.
        
        Args:
            base_model: Model for response generation
            evaluator_model: Model for quality evaluation
            score_threshold: Minimum acceptable quality score (0-10)
            max_iterations: Maximum refinement iterations
        """
        logger.info("=" * 80)
        logger.info("Initializing Agentic RAG System")
        logger.info("=" * 80)
        
        self.score_threshold = score_threshold
        self.max_iterations = max_iterations
        
        # Initialize components
        logger.info("Loading Base RAG System...")
        self.base_rag = BaseRAG(model_name=base_model)
        
        logger.info("Loading SLM Evaluator...")
        self.evaluator = SLMEvaluator(model_name=evaluator_model)
        
        logger.info("Initializing Agentic Tools...")
        self.agentic_tools = AgenticRAGTools()
        
        # Tracking
        self.query_history = []
        
        logger.info("=" * 80)
        logger.info("Agentic RAG System initialized successfully!")
        logger.info(f"Score Threshold: {score_threshold}")
        logger.info(f"Max Iterations: {max_iterations}")
        logger.info("=" * 80)
    
    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base."""
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        self.base_rag.add_documents(documents)
    
    def query(
        self,
        user_query: str,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Process a user query with iterative refinement.
        
        Args:
            user_query: The user's question
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with final response and metadata
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"NEW QUERY: {user_query}")
        logger.info("=" * 80 + "\n")
        
        start_time = time.time()
        
        # Reset agentic tools
        self.agentic_tools.reset_iteration()
        
        # Track iteration history
        iteration_history = []
        
        # Initial response generation
        current_context = None
        current_response = None
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*80}\n")
            
            # Generate response
            if iteration == 0:
                # First iteration: use base RAG
                logger.info("Generating initial response...")
                result = self.base_rag.generate_response(user_query)
                current_response = result['response']
                current_context = result['context']
            else:
                # Subsequent iterations: use enhanced prompt with additional context
                logger.info("Generating refined response with enhanced context...")
                
                # Create enhanced prompt
                enhanced_prompt = self.agentic_tools.create_enhanced_prompt(
                    user_query,
                    current_response,
                    current_context,
                    missing_aspects
                )
                
                # Generate with enhanced context
                result = self.base_rag.generate_response(
                    enhanced_prompt,
                    context=current_context
                )
                current_response = result['response']
            
            logger.info(f"\nGenerated Response:\n{'-'*80}\n{current_response}\n{'-'*80}\n")
            
            # Evaluate response
            logger.info("Evaluating response quality...")
            scores = self.evaluator.evaluate_response(
                user_query,
                current_response,
                current_context
            )
            
            # Log scores
            self._log_scores(scores, iteration + 1)
            
            # Store iteration data
            iteration_data = {
                'iteration': iteration + 1,
                'response': current_response,
                'context': current_context,
                'scores': scores,
                'timestamp': datetime.now().isoformat()
            }
            iteration_history.append(iteration_data)
            
            # Check if score exceeds threshold
            if self.evaluator.is_score_above_threshold(scores, self.score_threshold):
                logger.info(f"\n{'='*80}")
                logger.info(f"✓ SUCCESS: Score {scores['overall']:.2f} exceeds threshold {self.score_threshold}")
                logger.info(f"{'='*80}\n")
                break
            else:
                logger.info(f"\n{'='*80}")
                logger.info(f"✗ Score {scores['overall']:.2f} below threshold {self.score_threshold}")
                logger.info(f"{'='*80}\n")
                
                # If not last iteration, perform agentic refinement
                if iteration < self.max_iterations - 1:
                    logger.info("Initiating agentic refinement...")
                    
                    # Identify missing information
                    missing_aspects = self.agentic_tools.identify_missing_info(
                        user_query,
                        current_response,
                        current_context,
                        scores
                    )
                    
                    logger.info(f"\nMissing Aspects Identified:")
                    for i, aspect in enumerate(missing_aspects, 1):
                        logger.info(f"  {i}. {aspect}")
                    
                    # Generate search queries
                    search_queries = self.agentic_tools.generate_search_queries(
                        user_query,
                        missing_aspects,
                        current_context
                    )
                    
                    logger.info(f"\nSearch Queries Generated:")
                    for i, query in enumerate(search_queries, 1):
                        logger.info(f"  {i}. {query}")
                    
                    # Fetch additional context
                    logger.info("\nFetching additional context...")
                    additional_docs = []
                    for search_query in search_queries:
                        retrieved = self.base_rag.doc_store.retrieve(search_query, top_k=3)
                        additional_docs.extend([r['document'] for r in retrieved])
                    
                    # Deduplicate and combine with existing context
                    additional_docs = list(set(additional_docs))
                    logger.info(f"Retrieved {len(additional_docs)} additional documents")
                    
                    additional_context = "\n\n".join([
                        f"[Additional Doc {i+1}]: {doc}"
                        for i, doc in enumerate(additional_docs)
                    ])
                    
                    # Augment context
                    current_context = f"{current_context}\n\n{additional_context}"
                    
                else:
                    logger.info(f"\nMax iterations ({self.max_iterations}) reached.")
        
        # Calculate total time
        elapsed_time = time.time() - start_time
        
        # Prepare final result
        final_result = {
            'query': user_query,
            'final_response': current_response,
            'final_scores': scores,
            'iterations_used': len(iteration_history),
            'max_iterations': self.max_iterations,
            'threshold': self.score_threshold,
            'success': scores['overall'] >= self.score_threshold,
            'elapsed_time': elapsed_time,
            'iteration_history': iteration_history
        }
        
        # Store in history
        self.query_history.append(final_result)
        
        # Final summary
        self._log_final_summary(final_result)
        
        return final_result
    
    def _log_scores(self, scores: Dict[str, float], iteration: int):
        """Log evaluation scores in a formatted way."""
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION SCORES - Iteration {iteration}")
        logger.info(f"{'='*80}")
        logger.info(f"  Relevance:     {scores['relevance']:.2f} / 10.0")
        logger.info(f"  Faithfulness:  {scores['faithfulness']:.2f} / 10.0")
        logger.info(f"  Completeness:  {scores['completeness']:.2f} / 10.0")
        logger.info(f"  Coherence:     {scores['coherence']:.2f} / 10.0")
        logger.info(f"  " + "-" * 40)
        logger.info(f"  OVERALL:       {scores['overall']:.2f} / 10.0")
        logger.info(f"{'='*80}\n")
    
    def _log_final_summary(self, result: Dict[str, any]):
        """Log final summary of the query processing."""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Query: {result['query']}")
        logger.info(f"Iterations Used: {result['iterations_used']}/{result['max_iterations']}")
        logger.info(f"Final Score: {result['final_scores']['overall']:.2f}")
        logger.info(f"Threshold: {result['threshold']}")
        logger.info(f"Success: {'✓ YES' if result['success'] else '✗ NO'}")
        logger.info(f"Elapsed Time: {result['elapsed_time']:.2f} seconds")
        logger.info("=" * 80)
        logger.info(f"\nFINAL RESPONSE:")
        logger.info("-" * 80)
        logger.info(result['final_response'])
        logger.info("-" * 80 + "\n")
    
    def get_query_history(self) -> List[Dict[str, any]]:
        """Get history of all queries processed."""
        return self.query_history
    
    def save(self, path: str):
        """Save the system state."""
        logger.info(f"Saving system to {path}")
        self.base_rag.save(path)
    
    def load(self, path: str):
        """Load the system state."""
        logger.info(f"Loading system from {path}")
        self.base_rag.load(path)