"""
Example usage of HNSW-enhanced embedding search in Cog.

This example demonstrates:
1. Creating a graph with HNSW indexing enabled
2. Adding embeddings to the graph
3. Performing fast approximate similarity search
4. Comparing HNSW vs. exact search
"""

import numpy as np
from cog.torque import Graph


def example_basic_hnsw():
    """Basic example of HNSW-enabled embedding search."""
    print("=== Basic HNSW Example ===\n")
    
    # Create a graph with HNSW enabled
    graph = Graph(
        graph_name="hnsw_demo",
        use_hnsw=True,
        embedding_dim=128,
        cog_path_prefix="./demo_data"
    )
    
    # Add some sample embeddings
    print("Adding embeddings...")
    words = ["cat", "dog", "kitten", "puppy", "lion", "tiger", "elephant", "mouse"]
    
    for word in words:
        # Generate random embeddings for demonstration
        # In practice, use real embeddings from a model
        embedding = np.random.randn(128)
        graph.put_embedding(word, embedding)
    
    print(f"Added {len(words)} embeddings to the graph\n")
    
    # Perform HNSW-based similarity search
    print("Searching for similar words to 'cat' using HNSW...")
    graph.v(words)  # Start with all words as vertices
    results = graph.sim_hnsw("cat", k=5).all()
    
    print("Results:", results)
    print()


def example_exact_vs_hnsw():
    """Compare exact search vs. HNSW approximate search."""
    print("=== Exact vs. HNSW Comparison ===\n")
    
    # Create graph with HNSW
    graph = Graph(
        graph_name="comparison_demo",
        use_hnsw=True,
        embedding_dim=64,
        cog_path_prefix="./demo_data"
    )
    
    # Add embeddings
    print("Adding 100 embeddings...")
    words = [f"word_{i}" for i in range(100)]
    for word in words:
        embedding = np.random.randn(64)
        graph.put_embedding(word, embedding)
    
    # Exact search (slower but precise)
    print("\n1. Exact search using original sim() method:")
    print("   graph.v(words).sim('word_0', '>=', 0.5, use_hnsw=False)")
    graph_exact = Graph("comparison_demo", cog_path_prefix="./demo_data")
    graph_exact.v(words[:20])  # Search within first 20 words
    results_exact = graph_exact.sim("word_0", ">=", 0.5, use_hnsw=False).all()
    print(f"   Found {len(results_exact['result'])} results")
    
    # HNSW search (faster but approximate)
    print("\n2. HNSW search using enhanced sim() method:")
    print("   graph.v(words).sim('word_0', '>=', 0.5, use_hnsw=True)")
    graph_hnsw = Graph("comparison_demo", use_hnsw=True, embedding_dim=64, cog_path_prefix="./demo_data")
    graph_hnsw.v(words[:20])
    results_hnsw = graph_hnsw.sim("word_0", ">=", 0.5, use_hnsw=True).all()
    print(f"   Found {len(results_hnsw['result'])} results")
    
    print("\nNote: HNSW provides approximate results that are typically")
    print("very close to exact results but computed much faster.")
    print()


def example_custom_config():
    """Example with custom HNSW configuration."""
    print("=== Custom HNSW Configuration ===\n")
    
    # Create graph with custom HNSW parameters
    graph = Graph(
        graph_name="custom_config_demo",
        use_hnsw=True,
        embedding_dim=256,
        cog_path_prefix="./demo_data",
        hnsw_config={
            'space': 'l2',              # Use L2 distance instead of cosine
            'ef': 100,                  # Higher ef = more accurate search
            'M': 32,                    # More links = better recall
            'max_elements': 50000,      # Support more elements
            'ef_construction': 400      # Higher = better index quality
        }
    )
    
    print("Created graph with custom HNSW configuration:")
    print("  - space: l2 (Euclidean distance)")
    print("  - ef: 100 (more accurate queries)")
    print("  - M: 32 (better recall)")
    print("  - max_elements: 50,000")
    print()
    
    # Add some embeddings
    print("Adding embeddings with 256 dimensions...")
    for i in range(20):
        word = f"vec_{i}"
        embedding = np.random.randn(256)
        graph.put_embedding(word, embedding)
    
    print("Embeddings added successfully!")
    print()


def example_batch_operations():
    """Example showing batch operations with HNSW."""
    print("=== Batch Operations Example ===\n")
    
    graph = Graph(
        graph_name="batch_demo",
        use_hnsw=True,
        embedding_dim=128,
        cog_path_prefix="./demo_data"
    )
    
    # Add many embeddings
    print("Adding 1000 embeddings...")
    for i in range(1000):
        word = f"item_{i}"
        embedding = np.random.randn(128)
        graph.put_embedding(word, embedding)
    
    print("Done! HNSW index now contains 1000 embeddings")
    
    # Perform multiple searches
    print("\nPerforming similarity searches...")
    search_words = ["item_0", "item_100", "item_500"]
    
    for search_word in search_words:
        # Use HNSW for fast search
        graph.v([f"item_{i}" for i in range(100)])  # Search within subset
        results = graph.sim_hnsw(search_word, k=5).all()
        print(f"  Top 5 similar to '{search_word}': {len(results['result'])} results")
    
    print("\nNote: With 1000+ embeddings, HNSW provides significant speedup!")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("HNSW-Enhanced Embedding Search Examples")
    print("=" * 60)
    print()
    
    # Note: These examples require numpy and hnswlib to be installed
    print("Prerequisites: pip install numpy hnswlib")
    print()
    
    try:
        # Run examples
        example_basic_hnsw()
        example_exact_vs_hnsw()
        example_custom_config()
        example_batch_operations()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install required packages: pip install numpy hnswlib")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
