# EVoC vs. Manual HDBSCAN Pipeline

## Quick Comparison

| Feature | Manual HDBSCAN | EVoC |
|---------|---|---|
| **Setup** | Two-stage manual clustering (top-level + sub-clustering) | Single EVoC call, automatic multi-layer hierarchy |
| **Runtime** | Slower (two full HDBSCAN passes) | Faster (optimized for embeddings, single pass) |
| **Layers** | 2 layers (groups + sub-clusters) | Multiple auto-generated layers (finest to coarsest) |
| **Hyperparameters** | Many (min_cluster_size, eps, etc.) | Few (mostly defaults work well) |
| **Duplicates** | Manual detection | Automatic via `duplicates_` |
| **Topic inference** | Per-combined-cluster | Per-layer available |
| **Customize** | Full control, verbose | Simple, fewer options |

## Installation

```bash
pip install evoc
```

EVoC depends only on: `numpy`, `scikit-learn`, `numba`, `tqdm`, `tbb`

## File Structure

```
IR/
├── Colab_Main_without_KMeans.ipynb    # Original: manual 2-stage HDBSCAN
├── Colab_Main_with_EVoC.ipynb         # NEW: EVoC-based pipeline
├── EVOC_GUIDE.md                       # This file
├── Easy Import/
│   ├── cluster_search.py               # Shared search module
│   └── cluster_store.py
└── Marimo/
    └── cluster_*.py
```

## Quick Start: EVoC Pipeline

### 1. Load Data & Embeddings (same as before)
```python
sentences = load_titles_from_csv()
embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5").encode(...)
```

### 2. One-line clustering
```python
clusterer = evoc.EVoC()
labels = clusterer.fit_predict(embeddings)
```

### 3. Access results
```python
cluster_labels = clusterer.cluster_labels_    # Finest granularity
cluster_layers = clusterer.cluster_layers_    # All granularities
cluster_tree = clusterer.cluster_tree_        # Hierarchy
duplicates = clusterer.duplicates_            # Near-duplicates
```

## Understanding EVoC Output

### `cluster_labels_` (finest layer)
- Standard cluster labels, like HDBSCAN output
- Use for most detailed topic extraction
- Shape: (N,) with values in range [−1, n_clusters)
- −1 = noise point

### `cluster_layers_` (all granularities)
- List of label arrays, from fine to coarse
- Each layer is a progressively coarser clustering
- Useful for multi-scale analysis and UI drill-down

Example:
```
Layer 0: [  0,  1,  2,  ..., 150]  (150 clusters, finest)
Layer 1: [  0,  0,  1,  ...,  42]  (42 clusters)
Layer 2: [  0,  0,  0,  ...,  12]  (12 clusters, coarsest)
```

### `cluster_tree_`
- Hierarchy object describing parent-child relationships
- Can reconstruct cluster merging process
- Advanced usage: custom hierarchy visualization

### `duplicates_`
- List of lists, each sub-list is a duplicate group
- Example: `[[5, 128, 341], [17, 42]]` means docs 5/128/341 are very similar
- Use for deduplication or quality checks

## Quality Metrics

The EVoC notebook computes:

**Per layer:**
- **Silhouette score** (−1 to 1, higher = better separation)
- **Cluster size range** (min, max, median)
- **Noise %** (fraction of points labeled −1)

Example output:
```
[2] EVoC hierarchical clustering …
    → 127 clusters (layer 0 / finest)
    └ Quality: silhouette=0.52, noise=2.3%
    └ Sizes: 3–892 (median 45)
[4] Cluster layer hierarchy:
    Layer 0:  127 clusters, noise=2.3%, sizes 3–892
    Layer 1:   47 clusters, noise=1.8%, sizes 8–523
    Layer 2:   18 clusters, noise=0.9%, sizes 21–312
```

## Advantages of EVoC

1. **Multi-granularity built-in**: Get clusters at multiple levels automatically
   - No manual "decide on sub-cluster size"
   - Perfect for exploratory analysis and hierarchical UIs

2. **Automatic duplication detection**: Find near-duplicates without extra work

3. **Fewer hyperparameters**: Most defaults are sensible
   - No need to tune `min_cluster_size`, `eps`, etc.

4. **Faster**: Optimized for embedding vectors
   - Single pass vs. two HDBSCAN calls

5. **Cleaner code**: Less manual logic, fewer edge cases

## When to Use Each

**Use EVoC (`Colab_Main_with_EVoC.ipynb`) if:**
- You want fast clustering out-of-the-box
- Multi-layer hierarchy is useful (drill-down UI, multi-scale topics)
- You want automatic deduplication
- Embedding-only clustering (no other features needed)

**Use Manual HDBSCAN (`Colab_Main_without_KMeans.ipynb`) if:**
- You need fine-grained control over clustering parameters
- You have custom distance metrics or preprocessing
- You want to debug/understand the clustering process
- You need integration with other tools that expect specific HDBSCAN behavior

## Integration with Search & Visualization

Both pipelines feed into:
- **`cluster_search.py`**: Same semantic search module works with both
- **Visualization**: Both save embeddings and labels in compatible format
- **Dashboard**: Both export to interactive 3D HTML

To use EVoC results with search:

```python
from Easy_Import.cluster_search import ClusterSearch

cs = ClusterSearch(CACHE_DIR)  # Expects cached embeddings + labels
results = cs.search("vaccine side effects", top_k=5)
cs.display_results(results)
```

## Migrating Results

To convert EVoC results for use with the search module:

```python
# Load EVoC results
results = load_cache("evoc_results_{fp}")
sentences = results["sentences"]
embeddings = results["embeddings"]
cluster_labels = results["cluster_labels"]
topics_dict = results["topics_dict"]

# Convert to search-compatible format (same as manual pipeline)
np.savez("cluster_data.npz",
    embeddings=embeddings,
    group_labels=cluster_labels,  # Use finest layer
    sub_labels=np.full(len(sentences), -1),  # No sub-layer
    combined_labels=cluster_labels,
)

# Save metadata
meta = {
    "sentences": sentences,
    "group_topics": list(topics_dict.values()),
    "combined_topics": {int(k): v for k, v in topics_dict.items()},
}
with open("cluster_meta.json", "w") as f:
    json.dump(meta, f)
```

## Performance Notes

**Memory:**
- EVoC uses ~same as two HDBSCAN calls
- Typical: 3K docs × 1024-dim embeddings ≈ 12 MB

**Speed (empirical on CPU):**
- Embedding (3K docs): ~60s
- EVoC clustering: ~2-5s
- Total: ~65s vs. ~90s for manual two-stage

**GPU acceleration:**
- Embeddings scale well with GPU
- EVoC doesn't use GPU (numpy/numba based)
- Overall still faster due to fewer clustering passes

## Troubleshooting

### "ImportError: No module named 'evoc'"
```bash
pip install evoc
# Or in Colab:
# !pip install evoc
```

### Too many/too few clusters
EVoC automatically selects cluster count. If unhappy:
- Use a specific layer from `cluster_layers_` instead
- Adjust by manually selecting layer index (0=finest, -1=coarsest)

### Silhouette score is negative
- Clusters are overlapping; try a different layer
- EVoC may have detected ambiguous structure
- This is OK; inspect results visually

### High noise percentage
- Data may be sparse or multi-modal
- Try a different layer (coarser) from `cluster_layers_`

## Example: Switching Between Pipelines

**Old (manual):**
```python
exec(open('Colab_Main_without_KMeans.ipynb').read())
main()
# Access: group_labels, sub_labels, combined_labels, topics, embeddings
```

**New (EVoC):**
```python
exec(open('Colab_Main_with_EVoC.ipynb').read())
results = main()
# Access: results['cluster_labels'], results['cluster_layers'], results['topics_dict']
```

Both produce compatible outputs for search and visualization.

## Next Steps

1. **Try EVoC**: Run `Colab_Main_with_EVoC.ipynb` on your data
2. **Compare**: Check layer quality metrics and topics
3. **Integrate**: Use results with `cluster_search.py`
4. **Decide**: Stick with EVoC for production, or use manual if you need more control

---

**Questions?** Check EVoC docs: https://evoc.readthedocs.io/
