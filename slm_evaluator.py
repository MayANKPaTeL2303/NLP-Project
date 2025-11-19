"""
SLM Evaluator: Small Language Model for Response Quality Evaluation
Uses microsoft/deberta-v3-small for efficient quality scoring
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLMEvaluator:
    """
    Evaluates response quality using a small language model.
    Returns scores for relevance, completeness, accuracy, and coherence.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-small"):
        """
        Initialize the SLM evaluator.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading SLM Evaluator: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # For this implementation, we'll use DeBERTa for NLI-based evaluation
        # Alternatively, you can use vectara/hallucination_evaluation_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate_response(
        self, 
        query: str, 
        response: str, 
        context: str
    ) -> Dict[str, float]:
        """
        Evaluate response quality across multiple dimensions.
        
        Args:
            query: User's original query
            response: Generated response to evaluate
            context: Retrieved context used for generation
            
        Returns:
            Dictionary with scores and overall quality score (0-10)
        """
        logger.info("Evaluating response quality...")
        
        scores = {}
        
        # 1. Relevance: How relevant is the response to the query?
        scores['relevance'] = self._score_relevance(query, response)
        
        # 2. Faithfulness: Is the response faithful to the context?
        scores['faithfulness'] = self._score_faithfulness(context, response)
        
        # 3. Completeness: Does the response fully address the query?
        scores['completeness'] = self._score_completeness(query, response, context)
        
        # 4. Coherence: Is the response well-structured and coherent?
        scores['coherence'] = self._score_coherence(response)
        
        # Calculate overall score (weighted average)
        weights = {
            'relevance': 0.3,
            'faithfulness': 0.3,
            'completeness': 0.3,
            'coherence': 0.1
        }
        
        overall_score = sum(scores[k] * weights[k] for k in weights.keys())
        scores['overall'] = overall_score
        
        logger.info(f"Evaluation scores: {scores}")
        return scores
    
    def _score_relevance(self, query: str, response: str) -> float:
        """Score how relevant the response is to the query."""
        # Use NLI approach: query as premise, response as hypothesis
        prompt = f"Query: {query}\nResponse: {response}"
        return self._compute_entailment_score(query, response)
    
    def _score_faithfulness(self, context: str, response: str) -> float:
        """Score whether the response is faithful to the context."""
        if not context or len(context.strip()) == 0:
            return 0.5  # Neutral score if no context
        
        # Check if response is grounded in context
        return self._compute_entailment_score(context[:512], response[:512])
    
    def _score_completeness(self, query: str, response: str, context: str) -> float:
        """Score whether the response completely addresses the query."""
        # Check response length and information density
        response_length = len(response.split())
        
        # Heuristic: longer responses tend to be more complete
        length_score = min(response_length / 50, 1.0)  # Cap at 50 words
        
        # Check if key query terms appear in response
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        term_overlap = len(query_terms & response_terms) / max(len(query_terms), 1)
        
        # Combine scores
        completeness = (length_score * 0.4 + term_overlap * 0.6)
        return completeness * 10  # Scale to 0-10
    
    def _score_coherence(self, response: str) -> float:
        """Score the coherence and structure of the response."""
        # Simple heuristics for coherence
        sentences = response.split('.')
        num_sentences = len([s for s in sentences if len(s.strip()) > 0])
        
        # Well-structured responses have multiple sentences
        sentence_score = min(num_sentences / 3, 1.0)
        
        # Check for reasonable length
        words = response.split()
        length_score = 1.0 if 10 <= len(words) <= 200 else 0.5
        
        coherence = (sentence_score * 0.5 + length_score * 0.5)
        return coherence * 10  # Scale to 0-10
    
    def _compute_entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment score using the model.
        Returns score between 0-10.
        """
        try:
            # Prepare input
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # For binary classification (entailment vs not)
                # If 3-class (contradiction, neutral, entailment), use entailment class
                if probs.shape[-1] == 3:
                    entailment_score = probs[0][2].item()  # Entailment class
                else:
                    entailment_score = probs[0][1].item()  # Positive class
            
            # Scale to 0-10
            return entailment_score * 10
            
        except Exception as e:
            logger.error(f"Error computing entailment score: {e}")
            return 5.0  # Return neutral score on error
    
    def is_score_above_threshold(self, scores: Dict[str, float], threshold: float = 7.0) -> bool:
        """
        Check if overall score exceeds threshold.
        
        Args:
            scores: Dictionary of scores from evaluate_response
            threshold: Minimum acceptable score (0-10 scale)
            
        Returns:
            True if score is above threshold, False otherwise
        """
        return scores['overall'] >= threshold


# Alternative: Hallucination Detection Model
class HallucinationEvaluator:
    """
    Alternative evaluator using Vectara's hallucination detection model.
    """
    
    def __init__(self):
        logger.info("Loading Hallucination Evaluator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("vectara/hallucination_evaluation_model")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "vectara/hallucination_evaluation_model"
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Could not load Vectara model: {e}. Falling back to DeBERTa.")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v3-small"
            )
            self.model.to(self.device)
            self.model.eval()
    
    def detect_hallucination(self, context: str, response: str) -> float:
        """
        Detect hallucination in the response.
        
        Returns:
            Score from 0-1, where higher means less hallucination
        """
        try:
            inputs = self.tokenizer(
                context,
                response,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Higher score = less hallucination
                factual_score = probs[0][1].item() if probs.shape[-1] == 2 else probs[0][2].item()
            
            return factual_score
            
        except Exception as e:
            logger.error(f"Error detecting hallucination: {e}")
            return 0.5