"""
Command-line interface for querying the Agentic RAG System
"""

import argparse
import logging
from pathlib import Path
from agentic_rag_system import AgenticRAGSystem
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def load_documents_from_file(file_path: str):
    """Load documents from a text file (one document per line)."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(line)
    return documents


def main():
    parser = argparse.ArgumentParser(
        description='Query the Agentic RAG System'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='The query to process'
    )
    
    parser.add_argument(
        '--documents',
        type=str,
        default=None,
        help='Path to documents file (one document per line)'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='google/flan-t5-base',
        help='Base model for response generation'
    )
    
    parser.add_argument(
        '--evaluator-model',
        type=str,
        default='microsoft/deberta-v3-small',
        help='Model for quality evaluation'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=7.0,
        help='Quality score threshold (0-10)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum number of refinement iterations'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--load-state',
        type=str,
        default=None,
        help='Path to load saved system state'
    )
    
    parser.add_argument(
        '--save-state',
        type=str,
        default=None,
        help='Path to save system state after processing'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AGENTIC RAG SYSTEM - QUERY INTERFACE")
    print("="*80 + "\n")
    
    # Initialize system
    print("Initializing Agentic RAG System...")
    system = AgenticRAGSystem(
        base_model=args.base_model,
        evaluator_model=args.evaluator_model,
        score_threshold=args.threshold,
        max_iterations=args.max_iterations
    )
    
    # Load state if specified
    if args.load_state:
        print(f"\nLoading system state from: {args.load_state}")
        system.load(args.load_state)
    
    # Load documents if specified
    if args.documents:
        print(f"\nLoading documents from: {args.documents}")
        documents = load_documents_from_file(args.documents)
        print(f"Loaded {len(documents)} documents")
        system.add_documents(documents)
    else:
        # Use default sample documents
        print("\nNo document file specified. Using default sample documents...")
        default_docs = [
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
            "Machine Learning is a subset of AI that enables systems to learn from experience.",
            "Deep Learning uses neural networks with multiple layers for complex pattern recognition.",
            "Natural Language Processing helps computers understand and process human language.",
            "Computer Vision enables machines to interpret and understand visual information."
        ]
        system.add_documents(default_docs)
    
    # Process query
    print(f"\nProcessing query: {args.query}")
    result = system.query(args.query, verbose=True)
    
    # Display final result
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['final_response']}")
    print(f"\nScores:")
    for key, value in result['final_scores'].items():
        print(f"  {key.capitalize()}: {value:.2f}")
    print(f"\nSuccess: {result['success']}")
    print(f"Iterations: {result['iterations_used']}/{result['max_iterations']}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    print("="*80 + "\n")
    
    # Save output if specified
    if args.output:
        print(f"Saving results to: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            # Remove non-serializable fields
            output_data = {
                'query': result['query'],
                'final_response': result['final_response'],
                'final_scores': result['final_scores'],
                'iterations_used': result['iterations_used'],
                'max_iterations': result['max_iterations'],
                'threshold': result['threshold'],
                'success': result['success'],
                'elapsed_time': result['elapsed_time']
            }
            json.dump(output_data, f, indent=2)
        print("Results saved!")
    
    # Save state if specified
    if args.save_state:
        print(f"\nSaving system state to: {args.save_state}")
        system.save(args.save_state)
        print("State saved!")


if __name__ == "__main__":
    main()