#!/usr/bin/env python3
"""
FAISS index construction for efficient vector similarity search.

Key Features:
1. HNSW graph construction with GPU acceleration
2. IVF_PQ quantization for memory efficiency
3. Automatic fp32 to fp16 conversion
4. Index persistence and loading
5. Batch processing for large datasets
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Constants
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = DATA_DIR / "indexes"
EMBEDDING_DIR = DATA_DIR / "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FaissIndexBuilder:
    """Handle FAISS index construction and optimization"""
    
    def __init__(self, model_name: str = "sbert_model"):
        """
        Initialize index builder with SBERT model
        
        Args:
            model_name: Name of fine-tuned SBERT model directory
        """
        self.model = SentenceTransformer(
            str(MODEL_DIR / model_name),
            device=DEVICE
        )
        
        # Create directories
        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(EMBEDDING_DIR, exist_ok=True)

    def generate_embeddings(self, text_file: Path) -> np.ndarray:
        """
        Generate sentence embeddings from text file
        
        Args:
            text_file: Path to text file with one sentence per line
            
        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        with open(text_file) as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        print(f"Generating embeddings for {len(sentences)} sentences...")
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Save embeddings for later use
        emb_path = EMBEDDING_DIR / f"{text_file.stem}_embeddings.npy"
        np.save(emb_path, embeddings)
        
        return embeddings

    def build_hnsw_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build HNSW index with GPU acceleration
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS HNSW index
        """
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        
        # Convert to GPU if available
        if DEVICE == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        print("Building HNSW index...")
        start = time.time()
        index.add(embeddings)
        print(f"Index built in {time.time()-start:.2f}s")
        
        return index

    def optimize_index(self, index: faiss.Index, embeddings: np.ndarray) -> faiss.Index:
        """
        Apply IVF_PQ quantization to optimize index
        
        Args:
            index: Original FAISS index
            embeddings: Original embeddings
            
        Returns:
            Optimized FAISS index
        """
        dim = embeddings.shape[1]
        nlist = 100  # Number of clusters
        m = 8        # Number of subquantizers
        bits = 8      # Bits per subquantizer
        
        # Train IVF_PQ quantizer
        quantizer = faiss.IndexFlatL2(dim)
        index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
        
        print("Training IVF_PQ quantizer...")
        index_ivfpq.train(embeddings)
        
        # Add vectors to index
        index_ivfpq.add(embeddings)
        
        # Convert to fp16 for memory efficiency
        if DEVICE == "cuda":
            index_ivfpq = faiss.index_gpu_to_cpu(index_ivfpq)
        
        return faiss.index_fp16_to_fp32(index_ivfpq)

    def save_index(self, index: faiss.Index, name: str = "social_index"):
        """
        Save FAISS index to disk
        
        Args:
            index: FAISS index to save
            name: Name for index file
        """
        index_path = INDEX_DIR / f"{name}.faiss"
        faiss.write_index(index, str(index_path))
        print(f"Index saved to {index_path}")

    def build_full_pipeline(self, text_file: Path):
        """
        Complete index building pipeline
        
        Args:
            text_file: Path to text file with sentences to index
        """
        # Generate embeddings
        embeddings = self.generate_embeddings(text_file)
        
        # Build HNSW index
        index = self.build_hnsw_index(embeddings)
        
        # Optimize with IVF_PQ
        optimized_index = self.optimize_index(index, embeddings)
        
        # Save final index
        self.save_index(optimized_index)

def main():
    """Main index building workflow"""
    # Initialize builder
    builder = FaissIndexBuilder()
    
    # Build index from processed data
    text_file = Path(__file__).parent.parent / "data/processed/combined.json"
    builder.build_full_pipeline(text_file)
    
    print("\nIndex construction completed successfully!")

if __name__ == "__main__":
    main()
