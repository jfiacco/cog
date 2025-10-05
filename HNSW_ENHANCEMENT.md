# HNSW Enhancement for Cog Embedding Search

This document describes the HNSW (Hierarchical Navigable Small World) enhancement added to the Cog graph database's embedding search functionality. The implementation is based on the approach used in [Jina AI's vectordb](https://github.com/jina-ai/vectordb).

## Overview

HNSW indexing provides **approximate nearest neighbor (ANN) search** for embeddings, which is significantly faster than exact brute-force cosine similarity calculations, especially for large embedding collections. This is ideal for use cases involving thousands or millions of embeddings.

## What Was Changed

### 1. New HNSW Index Module (`lib/cog/cog/hnsw_index.py`)
Created a dedicated `HNSWIndex` class that wraps the `hnswlib` library with features:
- **Initialization**: Configurable HNSW parameters (ef, M, ef_construction, space, etc.)
- **Indexing**: Add single or batch embeddings with word-to-index mapping
- **Search**: Fast k-nearest neighbor search
- **Persistence**: Save and load index to/from disk
- **Deletion**: Mark items as deleted in the index

### 2. Configuration (`lib/cog/cog/config.py`)
Added HNSW configuration constants:
```python
HNSW_ENABLED = False  # Enable by default
HNSW_SPACE = 'cosine'  # Distance metric
HNSW_EF_CONSTRUCTION = 200  # Construction speed/accuracy tradeoff
HNSW_EF = 50  # Query time/accuracy tradeoff
HNSW_M = 16  # Bi-directional links per element
HNSW_MAX_ELEMENTS = 10000  # Maximum elements
HNSW_ALLOW_REPLACE_DELETED = False
HNSW_NUM_THREADS = 1
HNSW_INDEX_DIR = 'hnsw_indices'
```

### 3. Enhanced Graph Class (`lib/cog/cog/torque.py`)

#### Graph Initialization
Added optional HNSW parameters to the `Graph.__init__` method:
```python
graph = Graph(
    graph_name="my_graph",
    use_hnsw=True,           # Enable HNSW indexing
    embedding_dim=128,        # Embedding dimensionality
    hnsw_config={             # Optional custom config
        'ef': 100,
        'M': 32
    }
)
```

#### Updated `put_embedding` Method
Now automatically adds embeddings to both:
1. Cog storage (for exact retrieval)
2. HNSW index (for fast approximate search)

#### New `sim_hnsw` Method
Dedicated method for HNSW-based similarity search:
```python
# Fast approximate search
graph.v().sim_hnsw(
    word="query_word",
    k=10,                    # Number of neighbors
    threshold=0.8,           # Optional distance threshold
    operator='>='            # Optional comparison operator
)
```

#### Enhanced `sim` Method
The existing `sim` method now:
- **Automatically uses HNSW** when available (falling back to exact search)
- Accepts `use_hnsw` parameter to explicitly control behavior
```python
# Automatic (uses HNSW if available)
graph.v().sim("query_word", ">=", 0.8)

# Force exact search
graph.v().sim("query_word", ">=", 0.8, use_hnsw=False)

# Force HNSW (with fallback on error)
graph.v().sim("query_word", ">=", 0.8, use_hnsw=True)
```

### 4. Dependencies (`pyproject.toml`)
Added required packages:
- `hnswlib (>=0.8.0,<1.0.0)` - HNSW index implementation
- `numpy (>=1.24.0,<2.0.0)` - Array operations

## How to Use

### Basic Usage (Exact Search - Default)
```python
from cog.torque import Graph

# Create graph without HNSW
graph = Graph("my_graph")

# Add embeddings
graph.put_embedding("word1", [0.1, 0.2, 0.3, ...])
graph.put_embedding("word2", [0.4, 0.5, 0.6, ...])

# Search using exact cosine similarity (slower but precise)
results = graph.v().sim("word1", ">=", 0.8).all()
```

### Fast Approximate Search with HNSW
```python
from cog.torque import Graph

# Create graph with HNSW enabled
graph = Graph(
    "my_graph",
    use_hnsw=True,
    embedding_dim=128
)

# Add embeddings (automatically indexed in HNSW)
graph.put_embedding("word1", [0.1, 0.2, 0.3, ...])
graph.put_embedding("word2", [0.4, 0.5, 0.6, ...])
# ... add thousands more embeddings ...

# Search using HNSW (much faster for large collections)
results = graph.v().sim("word1", ">=", 0.8).all()
# OR explicitly use HNSW:
results = graph.v().sim_hnsw("word1", k=10, threshold=0.8, operator=">=").all()
```

### Custom HNSW Configuration
```python
graph = Graph(
    "my_graph",
    use_hnsw=True,
    embedding_dim=256,
    hnsw_config={
        'space': 'l2',         # Use L2 distance instead of cosine
        'ef': 100,             # Higher ef = more accurate but slower
        'M': 32,               # More links = better recall but more memory
        'max_elements': 50000  # Support up to 50k embeddings
    }
)
```

## Performance Characteristics

### Exact Search (Original `sim` method without HNSW)
- **Pros**: Guaranteed exact cosine similarity results
- **Cons**: O(n) time complexity - slow for large embedding collections
- **Best for**: Small to medium collections (<10,000 embeddings)

### HNSW Approximate Search
- **Pros**: 
  - Much faster - typically 10-100x speedup
  - Sublinear time complexity - scales well to millions of embeddings
  - Configurable speed/accuracy tradeoff
- **Cons**: 
  - Results are approximate (but typically >95% accurate)
  - Uses more memory for the index
- **Best for**: Large collections (>10,000 embeddings)

## Configuration Parameters Explained

### `ef_construction` (default: 200)
Controls index build speed vs. accuracy. Higher values = more accurate but slower to build.

### `ef` (default: 50)
Controls query speed vs. accuracy. Higher values = more accurate search but slower queries.

### `M` (default: 16)
Number of bi-directional links per node. Higher values = better recall but more memory usage.

### `space` (default: 'cosine')
Distance metric to use:
- `'cosine'`: Cosine distance (1 - cosine similarity)
- `'l2'`: Euclidean (L2) distance
- `'ip'`: Inner product

## Index Persistence

HNSW indices are automatically saved to disk when created/updated:
```
{cog_db_path}/hnsw_indices/{graph_name}.hnsw
{cog_db_path}/hnsw_indices/{graph_name}.hnsw.mappings
```

The index is automatically loaded when you create a Graph instance with `use_hnsw=True`.

## Migration Notes

### Backward Compatibility
The changes are **fully backward compatible**:
- Existing code continues to work without modification
- HNSW is disabled by default (`HNSW_ENABLED = False`)
- The `sim` method works exactly as before when HNSW is not enabled

### Enabling HNSW for Existing Graphs
To add HNSW to an existing graph:
1. Create Graph with `use_hnsw=True` and `embedding_dim`
2. Re-insert embeddings (they'll be automatically added to HNSW index)
3. Or load embeddings and batch-add them to the index:
```python
graph = Graph("existing_graph", use_hnsw=True, embedding_dim=128)

# Rebuild HNSW index from existing embeddings
# (This requires iterating through your embeddings)
```

## Installation

Install the new dependencies:
```bash
pip install hnswlib numpy
```

Or with poetry:
```bash
poetry install
```

## References

- [Jina AI vectordb](https://github.com/jina-ai/vectordb) - Reference implementation
- [hnswlib](https://github.com/nmslib/hnswlib) - HNSW library
- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Original research paper
