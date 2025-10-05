"""
HNSW Index Wrapper for Cog Embeddings
Based on the implementation from https://github.com/jina-ai/vectordb

This module provides an efficient HNSW (Hierarchical Navigable Small World) index
for approximate nearest neighbor search of embeddings.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict


class HNSWIndex:
    """
    HNSW Index wrapper for fast approximate nearest neighbor search.
    
    This class wraps the hnswlib library to provide efficient similarity search
    for embeddings stored in the Cog graph database.
    """
    
    def __init__(
        self,
        dim: int,
        space: str = 'cosine',
        ef_construction: int = 200,
        ef: int = 50,
        M: int = 16,
        max_elements: int = 10000,
        allow_replace_deleted: bool = False,
        num_threads: int = 1,
        index_path: Optional[str] = None
    ):
        """
        Initialize HNSW index.
        
        Parameters:
        -----------
        dim : int
            Dimensionality of the embeddings
        space : str
            Distance metric ('cosine', 'l2', or 'ip' for inner product)
        ef_construction : int
            Controls index construction speed/accuracy tradeoff. Higher = more accurate but slower.
        ef : int
            Controls query time/accuracy tradeoff. Higher = more accurate but slower.
        M : int
            Number of bi-directional links per element. Higher = better recall but more memory.
        max_elements : int
            Maximum number of elements in the index
        allow_replace_deleted : bool
            Allow replacing deleted elements with new ones
        num_threads : int
            Number of threads to use for index operations
        index_path : str, optional
            Path to save/load the index
        """
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "hnswlib is required for HNSW indexing. "
                "Install it with: pip install hnswlib"
            )
        
        self.logger = logging.getLogger("cog.hnsw")
        self.dim = dim
        self.space = space
        self.ef_construction = ef_construction
        self.ef = ef
        self.M = M
        self.max_elements = max_elements
        self.allow_replace_deleted = allow_replace_deleted
        self.num_threads = num_threads
        self.index_path = index_path
        
        # Create the HNSW index
        self.index = hnswlib.Index(space=space, dim=dim)
        
        # Track word to index mapping
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.current_idx = 0
        
        # Try to load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            # Initialize a new index
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
                allow_replace_deleted=allow_replace_deleted
            )
            self.index.set_ef(ef)
            self.index.set_num_threads(num_threads)
            
        self.logger.info(
            f"Initialized HNSW index: dim={dim}, space={space}, "
            f"ef={ef}, M={M}, max_elements={max_elements}"
        )
    
    def add_item(self, word: str, embedding: np.ndarray) -> None:
        """
        Add a single embedding to the index.
        
        Parameters:
        -----------
        word : str
            The word/identifier for the embedding
        embedding : np.ndarray
            The embedding vector
        """
        if word in self.word_to_idx:
            # Update existing item
            idx = self.word_to_idx[word]
            self.index.mark_deleted(idx)
        else:
            idx = self.current_idx
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            self.current_idx += 1
        
        # Ensure embedding is a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)
        
        # Add to index
        self.index.add_items(embedding.reshape(1, -1), np.array([idx]))
        self.logger.debug(f"Added item '{word}' at index {idx}")
    
    def add_items_batch(self, words: List[str], embeddings: np.ndarray) -> None:
        """
        Add multiple embeddings to the index in a batch.
        
        Parameters:
        -----------
        words : List[str]
            List of words/identifiers
        embeddings : np.ndarray
            Array of embedding vectors with shape (n, dim)
        """
        if len(words) != len(embeddings):
            raise ValueError("Number of words must match number of embeddings")
        
        indices = []
        valid_embeddings = []
        
        for word, embedding in zip(words, embeddings):
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                self.index.mark_deleted(idx)
            else:
                idx = self.current_idx
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                self.current_idx += 1
            
            indices.append(idx)
            valid_embeddings.append(embedding)
        
        # Convert to numpy arrays
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        indices_array = np.array(indices, dtype=np.int32)
        
        # Add to index
        self.index.add_items(embeddings_array, indices_array)
        self.logger.info(f"Added {len(words)} items in batch")
    
    def search(
        self,
        query_word: str,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors of a query embedding.
        
        Parameters:
        -----------
        query_word : str
            The query word (for logging purposes)
        query_embedding : np.ndarray
            The query embedding vector
        k : int
            Number of nearest neighbors to return
        
        Returns:
        --------
        List[Tuple[str, float]]
            List of (word, distance) tuples for the k nearest neighbors
        """
        if self.get_current_count() == 0:
            self.logger.warning("Index is empty, cannot perform search")
            return []
        
        # Ensure embedding is a numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        else:
            query_embedding = query_embedding.astype(np.float32)
        
        # Perform search
        k = min(k, self.get_current_count())
        labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=k)
        
        # Convert indices back to words
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                results.append((word, float(dist)))
        
        self.logger.debug(f"Search for '{query_word}' returned {len(results)} results")
        return results
    
    def get_current_count(self) -> int:
        """Get the current number of elements in the index."""
        return self.index.get_current_count()
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save the index to disk.
        
        Parameters:
        -----------
        path : str, optional
            Path to save the index. If None, uses self.index_path
        """
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path specified for saving index")
        
        # Save the HNSW index
        self.index.save_index(save_path)
        
        # Save the word mappings
        mapping_path = save_path + '.mappings'
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'current_idx': self.current_idx
            }, f)
        
        self.logger.info(f"Saved HNSW index to {save_path}")
    
    def load_index(self, path: str) -> None:
        """
        Load the index from disk.
        
        Parameters:
        -----------
        path : str
            Path to load the index from
        """
        # Load the HNSW index
        self.index.load_index(path, max_elements=self.max_elements)
        self.index.set_ef(self.ef)
        self.index.set_num_threads(self.num_threads)
        
        # Load the word mappings
        mapping_path = path + '.mappings'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                mappings = pickle.load(f)
                self.word_to_idx = mappings['word_to_idx']
                self.idx_to_word = mappings['idx_to_word']
                self.current_idx = mappings['current_idx']
        
        self.logger.info(f"Loaded HNSW index from {path}")
        self.index_path = path
    
    def delete_item(self, word: str) -> None:
        """
        Mark an item as deleted in the index.
        
        Parameters:
        -----------
        word : str
            The word to delete
        """
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            self.index.mark_deleted(idx)
            self.logger.debug(f"Deleted item '{word}' at index {idx}")
        else:
            self.logger.warning(f"Word '{word}' not found in index")
    
    def resize_index(self, new_max_elements: int) -> None:
        """
        Resize the index to accommodate more elements.
        
        Parameters:
        -----------
        new_max_elements : int
            New maximum number of elements
        """
        self.index.resize_index(new_max_elements)
        self.max_elements = new_max_elements
        self.logger.info(f"Resized index to max_elements={new_max_elements}")
