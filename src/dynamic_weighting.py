#!/usr/bin/env python3
"""
Dynamic weighting algorithm for hybrid retrieval system.

Key Features:
1. Online learning of optimal BM25/SBERT weights
2. Bandit algorithm for exploration-exploitation tradeoff
3. Contextual weighting based on query characteristics
4. Performance monitoring and feedback loop
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import util
import faiss

# Constants
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = DATA_DIR / "indexes"
CONFIG_DIR = Path(__file__).parent.parent / "configs"

class HybridRetriever:
    """Handle dynamic weighting between BM25 and SBERT retrieval"""
    
    def __init__(self):
        """
        Initialize hybrid retriever components
        """
        # Load models and index
        self.model = util.SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(str(INDEX_DIR / "social_index.faiss"))
        
        # Initialize weights (BM25, SBERT)
        self.weights = np.array([0.5, 0.5])
        
        # Contextual features
        self.feature_weights = {
            'query_length': 0.1,
            'query_entropy': 0.2,
            'query_type': 0.3  # 0=factual, 1=opinion, 2=conversational
        }
        
        # Performance tracking
        self.performance_history = []
        self.min_improvement = 0.01

    def calculate_query_features(self, query: str) -> Dict:
        """
        Extract features from query for contextual weighting
        
        Args:
            query: Input search query
            
        Returns:
            Dictionary of query features
        """
        # Basic features
        features = {
            'query_length': len(query.split()),
            'query_entropy': self._calculate_entropy(query),
            'query_type': self._classify_query_type(query)
        }
        return features

    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text
        
        Args:
            text: Input text
            
        Returns:
            Entropy value
        """
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        return -sum(p * np.log2(p) for p in prob if p > 0)

    def _classify_query_type(self, query: str) -> int:
        """
        Classify query into 3 types:
        0 - Factual (who, what, when)
        1 - Opinion (why, how, thoughts)
        2 - Conversational (social phrases)
        
        Args:
            query: Input query
            
        Returns:
            Query type (0, 1, or 2)
        """
        factual_words = {'who', 'what', 'when', 'where'}
        opinion_words = {'why', 'how', 'thought', 'opinion'}
        
        if any(word in query.lower() for word in factual_words):
            return 0
        elif any(word in query.lower() for word in opinion_words):
            return 1
        return 2

    def get_contextual_weights(self, query: str) -> np.ndarray:
        """
        Calculate contextual weights based on query features
        
        Args:
            query: Input search query
            
        Returns:
            Adjusted weights array
        """
        features = self.calculate_query_features(query)
        
        # Calculate adjustment factors
        length_factor = 1 + (features['query_length'] * self.feature_weights['query_length'])
        entropy_factor = 1 + (features['query_entropy'] * self.feature_weights['query_entropy'])
        type_factor = 1 + (features['query_type'] * self.feature_weights['query_type'])
        
        # Apply adjustments
        adjusted_weights = self.weights.copy()
        adjusted_weights[0] *= length_factor  # BM25 benefits from longer queries
        adjusted_weights[1] *= entropy_factor * type_factor  # SBERT better for complex queries
        
        # Normalize
        return adjusted_weights / np.sum(adjusted_weights)

    def retrieve(self, query: str, bm25_scores: Dict[str, float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform hybrid retrieval with dynamic weighting
        
        Args:
            query: Search query
            bm25_scores: BM25 scores {doc_id: score}
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Get contextual weights
        weights = self.get_contextual_weights(query)
        
        # Get SBERT scores
        query_embedding = self.model.encode(query)
        _, sbert_indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        sbert_scores = {str(i): float(s) for i, s in zip(sbert_indices[0], _[0])}
        
        # Combine scores
        combined_scores = {
            doc_id: (weights[0] * bm25_scores.get(doc_id, 0) + 
                    weights[1] * sbert_scores.get(doc_id, 0))
            for doc_id in set(bm25_scores) | set(sbert_scores)
        }
        
        # Sort and return top results
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def update_weights(self, feedback: Dict[str, float]):
        """
        Update weights based on user feedback
        
        Args:
            feedback: Dictionary of {doc_id: relevance_score}
        """
        # Calculate performance metric (e.g. NDCG)
        current_perf = self._calculate_performance(feedback)
        self.performance_history.append(current_perf)
        
        # Adjust weights if performance drops
        if len(self.performance_history) > 1 and current_perf < self.performance_history[-2] - self.min_improvement:
            self.weights[0] *= 0.9  # Reduce BM25 weight
            self.weights[1] *= 1.1  # Increase SBERT weight
            self.weights /= np.sum(self.weights)  # Renormalize

    def _calculate_performance(self, feedback: Dict[str, float]) -> float:
        """
        Calculate retrieval performance metric
        
        Args:
            feedback: User feedback {doc_id: relevance_score}
            
        Returns:
            Performance score (0-1)
        """
        # Simple implementation - average relevance
        return np.mean(list(feedback.values())) if feedback else 0.5

    def save_config(self):
        """Save current configuration to file"""
        config = {
            'base_weights': self.weights.tolist(),
            'feature_weights': self.feature_weights,
            'performance_history': self.performance_history
        }
        
        with open(CONFIG_DIR / "hybrid_config.json", 'w') as f:
            json.dump(config, f, indent=2)

def main():
    """Main retrieval workflow"""
    retriever = HybridRetriever()
    
    # Example usage
    query = "What are people saying about climate change?"
    bm25_scores = {"doc1": 0.8, "doc2": 0.6, "doc3": 0.4}
    
    results = retriever.retrieve(query, bm25_scores)
    print("Top results:", results)
    
    # Simulate feedback
    retriever.update_weights({"doc1": 1.0, "doc2": 0.5})
    retriever.save_config()

if __name__ == "__main__":
    main()
