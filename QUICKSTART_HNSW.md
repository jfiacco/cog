# Quick Start: HNSW-Enhanced Embedding Search

## Installation

First, install the required dependencies:

```bash
pip install hnswlib numpy
```

Or if using poetry:

```bash
poetry add hnswlib numpy
# Then run
poetry install
```

## Quick Example

```python
from cog.torque import Graph
import numpy as np

# 1. Create a graph with HNSW enabled
graph = Graph(
    graph_name="my_embeddings",
    use_hnsw=True,           # Enable HNSW indexing
    embedding_dim=128,        # Your embedding dimension
    cog_path_prefix="./data"  # Storage location
)

# 2. Add embeddings (they're automatically indexed in HNSW)
words = ["cat", "dog", "kitten", "puppy", "lion"]
for word in words:
    # Use your actual embedding model here
    embedding = np.random.randn(128)  # Replace with real embeddings
    graph.put_embedding(word, embedding)

# 3. Perform fast similarity search
graph.v(words)  # Start with your vertices
results = graph.sim("cat", ">=", 0.7).all()  # Automatically uses HNSW!

# Or explicitly use HNSW
results = graph.sim_hnsw("cat", k=5).all()

print("Similar words:", results)
```

## When to Use HNSW

### Use HNSW When:
- ✅ You have **1,000+ embeddings** (significant speedup)
- ✅ You need **fast queries** (10-100x faster)
- ✅ You can tolerate **approximate results** (typically >95% accurate)
- ✅ You're doing **frequent similarity searches**

### Use Exact Search When:
- ✅ You have **<1,000 embeddings** (HNSW overhead not worth it)
- ✅ You need **guaranteed exact results**
- ✅ You're doing **rare/one-off searches**

## Configuration Tips

### For Large Collections (100K+ embeddings)
```python
graph = Graph(
    "large_db",
    use_hnsw=True,
    embedding_dim=512,
    hnsw_config={
        'ef': 100,              # Higher ef = better accuracy
        'M': 32,                # More links = better recall
        'max_elements': 200000  # Increase capacity
    }
)
```

### For Fast Queries (Lower Accuracy OK)
```python
graph = Graph(
    "fast_db",
    use_hnsw=True,
    embedding_dim=128,
    hnsw_config={
        'ef': 20,               # Lower ef = faster queries
        'M': 8                  # Fewer links = less memory
    }
)
```

### For High Accuracy (Slower Queries)
```python
graph = Graph(
    "accurate_db",
    use_hnsw=True,
    embedding_dim=256,
    hnsw_config={
        'ef': 200,              # Higher ef = better accuracy
        'M': 48,                # More links = better recall
        'ef_construction': 400  # Better index quality
    }
)
```

## Migration from Existing Code

### Before (Exact Search Only)
```python
graph = Graph("my_graph")
graph.put_embedding("word", embedding)
results = graph.v().sim("word", ">=", 0.8).all()
```

### After (With HNSW - No Code Changes!)
```python
# Just enable HNSW, everything else stays the same!
graph = Graph("my_graph", use_hnsw=True, embedding_dim=128)
graph.put_embedding("word", embedding)
results = graph.v().sim("word", ">=", 0.8).all()  # Automatically uses HNSW
```

## Troubleshooting

### Import Error: "No module named 'hnswlib'"
```bash
pip install hnswlib
```

### Import Error: "No module named 'numpy'"
```bash
pip install numpy
```

### Warning: "HNSW enabled but embedding_dim not specified"
**Solution**: Specify `embedding_dim` when creating the Graph:
```python
graph = Graph("my_graph", use_hnsw=True, embedding_dim=128)
```

### Error: "HNSW index is not enabled"
**Solution**: Make sure you created the graph with `use_hnsw=True`:
```python
graph = Graph("my_graph", use_hnsw=True, embedding_dim=128)
```

## Performance Tips

1. **Batch Operations**: When adding many embeddings, add them sequentially - HNSW handles this efficiently
2. **Index Persistence**: HNSW indices are automatically saved to disk and reloaded
3. **Memory Usage**: HNSW uses ~M×dim×4 bytes per element (e.g., 16×128×4 = 8KB per embedding)
4. **Query Speed**: Use lower `ef` for faster queries, higher for better accuracy

## More Information

- Full documentation: `HNSW_ENHANCEMENT.md`
- Example code: `examples/hnsw_example.py`
- Jina vectordb reference: https://github.com/jina-ai/vectordb
- hnswlib library: https://github.com/nmslib/hnswlib

## Support

For issues or questions:
1. Check the documentation in `HNSW_ENHANCEMENT.md`
2. Review example code in `examples/hnsw_example.py`
3. See hnswlib docs for parameter tuning

Happy searching! 🚀
