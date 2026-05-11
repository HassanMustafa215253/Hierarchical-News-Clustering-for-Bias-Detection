# Hierarchical News Title Clustering Pipeline

A two-level hierarchical clustering pipeline for news headlines, designed to run in Google Colab. Groups titles first by shared keywords, then sub-clusters each group by semantic meaning, producing clusters-within-clusters that reflect both topic and nuance.

---

## How It Works

The pipeline has two levels of clustering:

**Level 1 — Keyword Groups (TF-IDF + KMeans)**
Titles are vectorised using TF-IDF and grouped into broad keyword-based buckets. This is fast and purely lexical — it separates "space station" articles from "oil prices" articles without caring about meaning.

**Level 2 — Semantic Sub-clusters (Embeddings + PCA + UMAP + HDBSCAN)**
Within each keyword group, titles are encoded using a sentence transformer model (BGE-large) to produce 1024-dimensional semantic embeddings. These are then reduced and clustered:

```
Raw embeddings (1024D)
  → PCA (150D)          linear noise removal; makes UMAP's nearest-neighbour fast
  → UMAP (50D)          nonlinear manifold structure; the actual clustering space
  → HDBSCAN             density-based clustering; no need to specify cluster count
```

The final output is a set of clusters labelled as `[keyword group] › semantic sub-topic`, for example:

```
[space, space station] › space, shuttle, space shuttle launch
[bomb, dead, car bomb] › bomb, madrid, threat, blast
```

**Visualisation**
A separate, cheaper UMAP pass reduces all embeddings to 3D for an interactive Plotly scatter plot. This 3D reduction is independent of the clustering — accuracy here doesn't matter, only visual interpretability.

---

## Dimensionality Reduction Rationale

BGE-large embeddings are already L2-normalised unit vectors. **StandardScaler is intentionally not applied** — scaling destroys the unit norm and degrades nearest-neighbour quality.

| Step | Input | Output | Why |
|---|---|---|---|
| PCA | 1024D | 150D | Removes redundancy linearly; UMAP's approximate-NN breaks above ~200D |
| UMAP (clustering) | 150D | 50D | Captures nonlinear manifold; `min_dist=0.0` for tight density clusters |
| UMAP (visualisation) | 150D | 3D | Separate pass; `min_dist=0.1` for readable spread; accuracy irrelevant |

After PCA the data is no longer on the unit sphere, so `metric="euclidean"` is used for both UMAP and HDBSCAN (not cosine).

---

## Requirements

### Python packages

```bash
# Always required
pip install numpy torch scikit-learn sentence-transformers plotly

# CPU-only path
pip install umap-learn hdbscan

# GPU path (RAPIDS) — Colab with A100/T4 + RAPIDS runtime
# Install via: https://rapids.ai/start.html
# cupy, cuml are auto-detected at runtime
```

### Data format

A CSV file with at least a `Title` column:

```
Title,Category,...
"Stocks Fall After Fed Meeting - www.reuters.com",Business,...
"Space Shuttle Launch Delayed - ap.org",Science,...
```

The pipeline automatically strips:
- Source suffixes (`Title - www.site.com` → `Title`)
- HTML entities (`&#39;` → `'`, `&lt;` → `<`)
- HTML tags (`<b>text</b>` → `text`)
- Non-alphanumeric punctuation (preserving apostrophes and hyphens)

---

## Setup & Usage

### Step 1 — Mount Google Drive (run first, every session)

```python
from google.colab import drive
drive.mount('/content/drive')
```

This must be the first cell. Without it, all computed results save to `/tmp` and are lost when the session ends.

### Step 2 — Upload your data

Upload `train.csv` to `/content/train.csv` in Colab, or change `CSV_FILE` at the top of the script.

### Step 3 — Run the pipeline

```python
main()
```

Or from terminal:

```bash
python clustering_pipeline.py
```

---

## Caching

Every expensive computation is saved to disk as a `.pkl` file and reloaded automatically on subsequent runs. The cache key includes an MD5 fingerprint of the input data, so if your titles change the cache invalidates itself.

| File | Contents | Typical size |
|---|---|---|
| `embeddings_{fp}.pkl` | BGE-large vectors (N × 1024) | ~60 MB for 15k titles |
| `groups_{fp}.pkl` | TF-IDF KMeans group labels | small |
| `sub_labels_{fp}.pkl` | HDBSCAN sub-cluster labels | small |
| `combined_{fp}.pkl` | Merged labels + topic strings | small |
| `umap3d_{fp}.pkl` | 3D visualisation coordinates | small |

**Cache location:**
- Google Drive mounted → `/content/drive/MyDrive/clustering_cache/` (persists across sessions)
- Drive not mounted → `/tmp/clustering_cache/` (lost on session end)

**To clear the cache** (required after changing `clean_title` or clustering parameters):

```python
import os, glob
for f in glob.glob("/content/drive/MyDrive/clustering_cache/*.pkl"):
    os.remove(f)
    print(f"Deleted {f}")
```

---

## Configuration

All tunable parameters are constants at the top of the script:

```python
CSV_FILE          = Path("/content/train.csv")   # path to your data
PCA_DIMS          = 150    # intermediate PCA target before UMAP
CLUSTER_UMAP_DIMS = 50     # UMAP target for clustering (fed into HDBSCAN)
```

HDBSCAN parameters inside `semantic_subclusters()`:

```python
mcs = max(10, min(group_size // 20, 50))  # min cluster size — raise to get fewer, larger clusters
min_samples          = 5                   # raise to require denser cores; reduces noise
cluster_selection_method = "eom"           # "eom" merges micro-clusters; "leaf" splits to maximum granularity
```

Number of Level 1 keyword groups in `main()`:

```python
n_groups = max(5, min(50, int(np.sqrt(N / 10))))  # auto-scales with dataset size
```

---

## GPU vs CPU

The script detects the available hardware at startup and prints which path it takes:

```
✓ RAPIDS / cuML detected — using GPU pipeline
```
or
```
✓ No GPU — using CPU pipeline (scikit-learn / umap-learn / hdbscan)
```

The same functions run on both paths — there are no commented-out blocks to toggle. On GPU, PCA, UMAP, and HDBSCAN all run via cuML (RAPIDS). On CPU they run via scikit-learn, umap-learn, and hdbscan respectively.

Approximate runtimes for 15k titles:

| Step | GPU (T4) | CPU |
|---|---|---|
| Embeddings (BGE-large) | ~3 min | ~25 min |
| Level 1 KMeans | <1 min | <1 min |
| Level 2 UMAP + HDBSCAN (all groups) | ~5 min | ~45 min |
| 3D UMAP for visualisation | ~1 min | ~10 min |

After the first run everything loads from cache in seconds.

---

## Output

**Console** — per-group sub-cluster summary with noise percentage:

```
Group  0  'space, space station, sta'  →  12 sub-clusters,  45 noise pts  (14.2%)
Group  1  'oil prices, prices, oil'    →   8 sub-clusters,  23 noise pts   (9.1%)
...
Final : 187 clusters,  1243 noise points
```

**Interactive 3D plot** — rendered inline in Colab via Plotly WebGL. Rotate to find angles where clusters separate cleanly. Legend entries are clickable to toggle individual clusters on/off. Also saved as a self-contained HTML file:

```
/content/clusters_3d.html
```

**Noise points** — plotted as semi-transparent grey. These are titles HDBSCAN could not assign to any cluster — too far from any dense region. Under 10% noise is healthy; over 20% suggests `min_cluster_size` or `min_samples` should be lowered.

---

## Project Structure

```
clustering_pipeline.py   main script — all logic in one file
README.md                this file
```

Cache files (generated at runtime, not committed):
```
clustering_cache/
  embeddings_{fp}.pkl
  groups_{fp}.pkl
  sub_labels_{fp}.pkl
  combined_{fp}.pkl
  umap3d_{fp}.pkl
```