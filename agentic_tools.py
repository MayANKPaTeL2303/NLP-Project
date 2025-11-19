"""
Agentic RAG Tools: Components for identifying missing information and fetching additional context
"""

import logging
from typing import List, Dict, Tuple
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenticRAGTools:
    """
    Tools for analyzing response gaps and generating targeted queries
    to fetch additional context.
    """
    
    def __init__(self):
        self.iteration_count = 0
        self.search_history = []
    
    def identify_missing_info(
        self, 
        query: str, 
        response: str, 
        context: str,
        scores: Dict[str, float]
    ) -> List[str]:
        """
        Analyze the response and identify what information is missing.
        
        Args:
            query: Original user query
            response: Generated response
            context: Retrieved context used
            scores: Evaluation scores
            
        Returns:
            List of missing information aspects
        """
        logger.info("Identifying missing information...")
        
        missing_aspects = []
        
        # 1. Check for incomplete query coverage
        query_keywords = self._extract_keywords(query)
        response_keywords = self._extract_keywords(response)
        
        missing_keywords = [kw for kw in query_keywords if kw not in response_keywords]
        if missing_keywords:
            missing_aspects.append(f"Keywords not addressed: {', '.join(missing_keywords)}")
        
        # 2. Check response length (too short might indicate missing details)
        if len(response.split()) < 30:
            missing_aspects.append("Response too brief, likely missing details")
        
        # 3. Analyze specific score deficiencies
        if scores.get('relevance', 10) < 7:
            missing_aspects.append("Low relevance - response may be off-topic or need refocusing")
        
        if scores.get('faithfulness', 10) < 7:
            missing_aspects.append("Low faithfulness - need more grounded information from sources")
        
        if scores.get('completeness', 10) < 7:
            missing_aspects.append("Incomplete answer - need additional information to fully address query")
        
        # 4. Check for question words in query that aren't answered
        question_patterns = {
            'what': 'definition or explanation',
            'why': 'reasons or causes',
            'how': 'methods or processes',
            'when': 'time or timeline',
            'where': 'location or context',
            'who': 'people or entities involved'
        }
        
        query_lower = query.lower()
        for qword, info_type in question_patterns.items():
            if qword in query_lower and not self._check_answer_type(response, qword):
                missing_aspects.append(f"Missing {info_type} ('{qword}' question)")
        
        # 5. Check for vague or generic responses
        if self._is_response_vague(response):
            missing_aspects.append("Response is too vague or generic, needs specific information")
        
        logger.info(f"Identified {len(missing_aspects)} missing aspects")
        return missing_aspects
    
    def generate_search_queries(
        self, 
        query: str, 
        missing_aspects: List[str],
        previous_context: str
    ) -> List[str]:
        """
        Generate targeted search queries to fill information gaps.
        
        Args:
            query: Original user query
            missing_aspects: List of identified missing information
            previous_context: Context from previous iterations
            
        Returns:
            List of search queries
        """
        logger.info("Generating targeted search queries...")
        
        search_queries = []
        
        # 1. Base query with refinements
        base_keywords = self._extract_keywords(query)
        
        # 2. Generate queries for each missing aspect
        for aspect in missing_aspects:
            if "keywords not addressed" in aspect.lower():
                # Extract the missing keywords
                keywords = aspect.split(":")[-1].strip()
                search_queries.append(f"{query} {keywords}")
            
            elif "brief" in aspect.lower() or "details" in aspect.lower():
                search_queries.append(f"{query} detailed explanation")
                search_queries.append(f"{query} comprehensive guide")
            
            elif "definition" in aspect.lower():
                search_queries.append(f"what is {query}")
                search_queries.append(f"{query} definition meaning")
            
            elif "reasons" in aspect.lower() or "causes" in aspect.lower():
                search_queries.append(f"why {query}")
                search_queries.append(f"{query} reasons causes")
            
            elif "methods" in aspect.lower() or "processes" in aspect.lower():
                search_queries.append(f"how to {query}")
                search_queries.append(f"{query} process steps")
            
            elif "time" in aspect.lower() or "timeline" in aspect.lower():
                search_queries.append(f"when {query}")
                search_queries.append(f"{query} timeline history")
            
            elif "location" in aspect.lower():
                search_queries.append(f"where {query}")
                search_queries.append(f"{query} location place")
            
            elif "people" in aspect.lower() or "entities" in aspect.lower():
                search_queries.append(f"who {query}")
                search_queries.append(f"{query} people involved")
            
            elif "vague" in aspect.lower() or "generic" in aspect.lower():
                search_queries.append(f"{query} specific examples")
                search_queries.append(f"{query} case studies")
            
            else:
                # Generic refinement
                search_queries.append(f"{query} additional information")
        
        # 3. Add alternative phrasings if we have few queries
        if len(search_queries) < 2:
            search_queries.append(f"{query} overview")
            search_queries.append(f"{query} key points")
        
        # 4. Remove duplicates and limit number of queries
        search_queries = list(dict.fromkeys(search_queries))[:5]
        
        # 5. Avoid repeating previous searches
        search_queries = [q for q in search_queries if q not in self.search_history]
        self.search_history.extend(search_queries)
        
        logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")
        return search_queries
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'about',
            'what', 'when', 'where', 'which', 'who', 'how', 'why'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get most common keywords
        word_freq = Counter(keywords)
        top_keywords = [word for word, freq in word_freq.most_common(10)]
        
        return top_keywords
    
    def _check_answer_type(self, response: str, question_word: str) -> bool:
        """Check if response appropriately answers the question type."""
        response_lower = response.lower()
        
        # Simple heuristics for answer type checking
        patterns = {
            'what': ['is', 'are', 'means', 'refers to', 'defined as'],
            'why': ['because', 'due to', 'reason', 'cause', 'since'],
            'how': ['by', 'through', 'using', 'process', 'method', 'steps'],
            'when': ['in', 'during', 'at', 'time', 'date', 'year'],
            'where': ['in', 'at', 'located', 'place', 'location'],
            'who': ['person', 'people', 'individual', 'organization', 'name']
        }
        
        if question_word in patterns:
            return any(pattern in response_lower for pattern in patterns[question_word])
        
        return True
    
    def _is_response_vague(self, response: str) -> bool:
        """Check if response is too vague or generic."""
        vague_indicators = [
            'generally', 'typically', 'usually', 'often', 'sometimes',
            'may', 'might', 'could', 'possibly', 'perhaps',
            'it depends', 'various', 'different', 'several'
        ]
        
        response_lower = response.lower()
        vague_count = sum(1 for indicator in vague_indicators if indicator in response_lower)
        
        # If more than 2 vague indicators and response is short, it's too vague
        return vague_count > 2 and len(response.split()) < 50
    
    def create_enhanced_prompt(
        self,
        original_query: str,
        previous_response: str,
        additional_context: str,
        missing_aspects: List[str]
    ) -> str:
        """
        Create an enhanced prompt for the next iteration.
        
        Args:
            original_query: Original user query
            previous_response: Previous response that was insufficient
            additional_context: Newly retrieved context
            missing_aspects: What was missing from previous response
            
        Returns:
            Enhanced prompt string
        """
        prompt = f"""Original Query: {original_query}

Previous Response Issues:
{chr(10).join(f"- {aspect}" for aspect in missing_aspects)}

Additional Context Retrieved:
{additional_context}

Please provide a comprehensive answer that addresses all aspects of the query, 
incorporating the additional context and ensuring completeness, relevance, and accuracy.
Focus especially on the missing aspects identified above.

Answer:"""
        
        return prompt
    
    def reset_iteration(self):
        """Reset iteration counter and search history."""
        self.iteration_count = 0
        self.search_history = []