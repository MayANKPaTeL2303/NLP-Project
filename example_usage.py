"""
Example usage of the Agentic RAG System
"""

import logging
from agentic_rag_system import AgenticRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Sample documents for the knowledge base
    sample_documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
        "It includes learning, reasoning, and self-correction. AI can be categorized into "
        "narrow AI, general AI, and superintelligent AI.",
        
        "Machine Learning is a subset of AI that enables systems to learn and improve from "
        "experience without being explicitly programmed. It focuses on developing computer "
        "programs that can access data and use it to learn for themselves.",
        
        "Deep Learning is a subset of machine learning based on artificial neural networks. "
        "It uses multiple layers to progressively extract higher-level features from raw input. "
        "Applications include image recognition, natural language processing, and autonomous vehicles.",
        
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, "
        "interpret and manipulate human language. NLP draws from many disciplines, including "
        "computer science and computational linguistics.",
        
        "Reinforcement Learning is an area of machine learning where an agent learns to behave "
        "in an environment by performing actions and seeing the results. The agent receives "
        "rewards or penalties for its actions and learns to maximize cumulative reward.",
        
        "Computer Vision is a field of AI that trains computers to interpret and understand "
        "the visual world. Using digital images from cameras and videos and deep learning models, "
        "machines can accurately identify and classify objects.",
        
        "Neural Networks are computing systems inspired by biological neural networks. "
        "They consist of interconnected nodes (neurons) that process information using a "
        "connectionist approach to computation. They can learn complex patterns in data.",
        
        "Supervised Learning is a machine learning approach where the model is trained on "
        "labeled data. The algorithm learns from examples with known outcomes and can then "
        "make predictions on new, unseen data.",
        
        "Unsupervised Learning involves training models on unlabeled data. The system tries "
        "to learn patterns and structure from the data without explicit guidance. Common "
        "applications include clustering and dimensionality reduction.",
        
        "Transfer Learning is a machine learning technique where a model developed for one "
        "task is reused as the starting point for a model on a second task. It's especially "
        "useful when you have limited data for the target task."
    ]
    
    print("\n" + "="*80)
    print("AGENTIC RAG SYSTEM - EXAMPLE USAGE")
    print("="*80 + "\n")
    
    # Initialize the system
    print("Initializing Agentic RAG System...")
    system = AgenticRAGSystem(
        base_model="google/flan-t5-base",
        evaluator_model="microsoft/deberta-v3-small",
        score_threshold=7.0,
        max_iterations=3
    )
    
    # Add documents to knowledge base
    print("\nAdding documents to knowledge base...")
    system.add_documents(sample_documents)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "How does deep learning relate to neural networks and what are its applications?",
    ]
    
    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}/{len(queries)}")
        print(f"{'='*80}\n")
        
        result = system.query(query, verbose=True)
        
        # Display results
        print("\n" + "="*80)
        print("RESULT SUMMARY")
        print("="*80)
        print(f"Success: {result['success']}")
        print(f"Iterations: {result['iterations_used']}/{result['max_iterations']}")
        print(f"Final Score: {result['final_scores']['overall']:.2f}/{result['threshold']}")
        print(f"Time: {result['elapsed_time']:.2f}s")
        print("="*80 + "\n")
        
        input("Press Enter to continue to next query...")
    
    # Show query history
    print("\n" + "="*80)
    print("QUERY HISTORY SUMMARY")
    print("="*80)
    
    history = system.get_query_history()
    for i, record in enumerate(history, 1):
        print(f"\nQuery {i}: {record['query']}")
        print(f"  Success: {record['success']}")
        print(f"  Iterations: {record['iterations_used']}")
        print(f"  Final Score: {record['final_scores']['overall']:.2f}")
        print(f"  Time: {record['elapsed_time']:.2f}s")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()