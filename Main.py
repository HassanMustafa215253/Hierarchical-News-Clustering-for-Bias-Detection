# -*- coding: utf-8 -*-
"""
Hierarchical Sentence Clustering Pipeline
==========================================
Level 1 : TF-IDF + KMeans  (keyword groups)
Level 2 : Embeddings → PCA (1024D → 150D) → UMAP (150D → 50D) → HDBSCAN
Visualization : Plotly  (interactive 3-D scatter — works in Colab without a display server)

Dimensionality reduction rationale
------------------------------------
BGE-large embeddings are 1024-dimensional unit vectors (already L2-normalised).
StandardScaler is intentionally NOT applied — it would destroy the unit-norm
property and hurt nearest-neighbour quality.

  PCA  : 1024D → PCA_DIMS (default 150)
    Removes noise / redundancy linearly; cheap and lossless for this range.
    Must run before UMAP because UMAP's approximate-NN degrades sharply
    above ~200 dimensions.

  UMAP : PCA_DIMS → CLUSTER_UMAP_DIMS (default 50) for clustering
    Captures nonlinear manifold structure; feeds directly into HDBSCAN.
    min_dist=0.0 → tightest possible packing for density-based clustering.

  UMAP : PCA_DIMS → 3  for visualisation only (separate, cheaper pass)
    Accuracy here is irrelevant — just needs to look interpretable.
    min_dist=0.1 → slight spread so points don't all overlap.

GPU / CPU auto-detection
------------------------
If a CUDA GPU is available the script uses cupy / cuml (rapids).
Otherwise it falls back to scikit-learn / umap-learn / hdbscan (pure CPU).
Only ONE set of imports is active at runtime — no commented-out blocks.
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
import csv
import hashlib
import html
import json
import os
import pickle
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")          # keep Colab output clean

# ── Third-party (always available) ───────────────────────────────────────────
import numpy as np
import torch
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# ── GPU / CPU conditional imports ─────────────────────────────────────────────
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        import cupy as cp
        from cuml.decomposition import PCA as _PCA
        from cuml.manifold import UMAP as _UMAP
        from cuml.cluster import HDBSCAN as _HDBSCAN
        USE_GPU = True
        print("✓ RAPIDS / cuML detected — using GPU pipeline")
    except ImportError:
        USE_GPU = False
        print("⚠ CUDA found but cuML not installed — falling back to CPU pipeline")
else:
    USE_GPU = False
    print("✓ No GPU — using CPU pipeline (scikit-learn / umap-learn / hdbscan)")

if not USE_GPU:
    from sklearn.decomposition import PCA as _PCA
    # umap-learn  →  pip install umap-learn
    from umap import UMAP as _UMAP
    # hdbscan     →  pip install hdbscan
    from hdbscan import HDBSCAN as _HDBSCAN

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_FILE = Path("/content/train.csv")

# Dimensionality reduction settings
# PCA_DIMS      : intermediate target after PCA (noise removal before UMAP)
#                 150 retains ~95%+ variance for BGE-large while making
#                 UMAP's approximate-NN fast and accurate.
# CLUSTER_UMAP_DIMS : final dims fed into HDBSCAN — 50 is the sweet spot;
#                 too low loses structure, too high and HDBSCAN's
#                 "curse of dimensionality" degrades density estimates.
PCA_DIMS          = 150
CLUSTER_UMAP_DIMS = 50

# Cache is stored on Google Drive when mounted, otherwise falls back to /tmp
_DRIVE_CACHE = "/content/drive/MyDrive/clustering_cache"
_LOCAL_CACHE = "/tmp/clustering_cache"

# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_cache_dir() -> str:
    """
    Return the cache directory to use.
    Prefer Google Drive (persistent across sessions) when it is mounted.
    Fall back to /tmp (lost on session restart, but no error).
    """
    if os.path.isdir("/content/drive/MyDrive"):
        cache_dir = _DRIVE_CACHE
    else:
        cache_dir = _LOCAL_CACHE
        print(
            "⚠  Google Drive not mounted — cache will be stored in /tmp "
            "(lost when session ends).\n"
            "   Run: from google.colab import drive; drive.mount('/content/drive')"
        )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# Resolve once at import time so every call is consistent.
CACHE_DIR = _resolve_cache_dir()


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def save_cache(name: str, obj) -> None:
    path = _cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ saved  → {path}")


def load_cache(name: str):
    path = _cache_path(name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✓ loaded ← {path}")
        return obj
    return None          # caller checks for None


def data_fingerprint(sentences: list[str]) -> str:
    """MD5 of the serialised sentence list — cache is invalidated if input changes."""
    raw = json.dumps(sentences, ensure_ascii=False).encode()
    return hashlib.md5(raw).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

# Compiled once at module load — cheap to reuse
_RE_SOURCE_SUFFIX = re.compile(r'\s+-\s+\S+\.\S+$')   # " - www.site.com" at end of title
_RE_HTML_TAG      = re.compile(r'<[^>]+>')
_RE_NONALPHA      = re.compile(r'[^a-zA-Z0-9\s\'\-]')
_RE_SPACES        = re.compile(r'\s+')


def clean_title(raw: str) -> str:
    """
    Normalise a news headline for both TF-IDF and embedding.

    Steps (in order):
      1. html.unescape  : &#39; → '   &amp; → &   &lt; → <   &gt; → >
      2. Strip source suffix: "Some Title - www.washingtonpost.com" → "Some Title"
         Matches " - " followed by a token containing a dot at the very end of
         the string — precise enough to avoid stripping legitimate mid-title dashes.
      3. Strip residual HTML tags (<b>, <i>, etc.)
      4. Remove punctuation that isn't apostrophe or hyphen
         (keeps contractions and hyphenated words intact)
      5. Collapse whitespace and strip
    """
    s = html.unescape(raw)                  # &#39; → '   &lt; → <   &gt; → >
    s = _RE_SOURCE_SUFFIX.sub('', s)        # "Title - site.com" → "Title"
    s = _RE_HTML_TAG.sub(' ', s)            # <b>foo</b> → foo
    s = _RE_NONALPHA.sub(' ', s)            # remove remaining punctuation / symbols
    s = _RE_SPACES.sub(' ', s)              # collapse whitespace
    return s.strip()


def load_titles_from_csv(csv_file: Path = CSV_FILE) -> list[str]:
    """Load and clean non-empty values from the 'Title' column of a CSV file."""
    titles: list[str] = []
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("Title") or "").strip()
            if not raw:
                continue
            cleaned = clean_title(raw)
            if cleaned:                 # skip titles that become empty after cleaning
                titles.append(cleaned)
    return titles


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def choose_umap_neighbors(num_samples: int, preferred: int = 30) -> int:
    """Clamp n_neighbors so it is always < num_samples."""
    if num_samples <= 2:
        return 1
    return max(2, min(preferred, num_samples - 1))


def infer_topic(sentences: list[str], top_terms: int = 5) -> str:
    """Best-effort cluster label derived from TF-IDF term frequencies."""
    if not sentences:
        return "miscellaneous"
    analyzer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    ).build_analyzer()
    counts: Counter = Counter()
    for s in sentences:
        counts.update(analyzer(s))
    terms = [t for t, _ in counts.most_common(top_terms)]
    return ", ".join(terms) if terms else "miscellaneous"


def _to_numpy(arr) -> np.ndarray:
    """Convert cupy array → numpy, or pass numpy through unchanged."""
    if USE_GPU:
        import cupy as cp          # already imported at top but re-import is cheap
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


def _free_gpu() -> None:
    """Release cupy memory pool blocks (no-op on CPU)."""
    if USE_GPU:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — keyword groups (TF-IDF + KMeans)
# ─────────────────────────────────────────────────────────────────────────────

def assign_keyword_groups(
    sentences: list[str],
    n_groups: int,
) -> tuple[np.ndarray, list[str]]:
    """
    TF-IDF vectorisation followed by KMeans grouping.

    Returns
    -------
    group_labels : np.ndarray shape (N,)
    group_topics : list[str]  — top TF-IDF terms for each centroid
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,   # log(1+tf) — better for short texts
        min_df=2,            # remove terms appearing in only 1 document
    )
    tfidf_matrix  = vectorizer.fit_transform(sentences)
    feature_names = np.array(vectorizer.get_feature_names_out())

    km = KMeans(
        n_clusters=n_groups,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=42,
    )
    group_labels = km.fit_predict(tfidf_matrix)

    group_topics: list[str] = []
    for centroid in km.cluster_centers_:
        top_idx   = centroid.argsort()[::-1][:5]
        top_words = ", ".join(feature_names[top_idx])
        group_topics.append(top_words)

    return group_labels.astype(int), group_topics


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — semantic sub-clustering (GPU or CPU unified path)
# ─────────────────────────────────────────────────────────────────────────────

def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    PCA dimensionality reduction.
    Input  : L2-normalised embeddings — StandardScaler is intentionally skipped
             because scaling unit vectors destroys their norm and degrades
             nearest-neighbour quality.
    Output : plain numpy array of shape (N, n_components).
    """
    M          = embeddings.shape[0]
    pca_dims   = min(n_components, M - 1, embeddings.shape[1])

    if USE_GPU:
        import cupy as cp
        arr = cp.asarray(embeddings)
    else:
        arr = embeddings

    arr = _PCA(n_components=pca_dims).fit_transform(arr)
    return _to_numpy(arr)


def _umap_reduce(
    pca_output  : np.ndarray,
    n_components: int,
    n_neighbors : int,
    min_dist    : float,
    n_epochs    : int,
    metric      : str = "euclidean",   # euclidean is correct AFTER PCA
) -> np.ndarray:
    """
    UMAP dimensionality reduction on already-PCA-reduced data.
    Always returns a plain numpy array.

    Note: metric="euclidean" is correct here. The original L2-normalised
    embeddings used cosine similarity, but after PCA the data is no longer
    on the unit sphere, so euclidean is the right choice.
    """
    M         = pca_output.shape[0]
    umap_dims = min(n_components, pca_output.shape[1], M - 1)

    if USE_GPU:
        import cupy as cp
        arr        = cp.asarray(pca_output)
        umap_model = _UMAP(
            n_components=umap_dims,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            n_epochs=n_epochs,
        )
    else:
        arr        = pca_output
        umap_model = _UMAP(
            n_components=umap_dims,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            n_epochs=n_epochs,
            low_memory=True,    # CPU-only; keeps RAM usage manageable
        )

    arr = umap_model.fit_transform(arr)
    return _to_numpy(arr)


def reduce_for_clustering(
    embeddings: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """
    Accurate reduction pipeline for HDBSCAN input:
      raw embeddings (1024D)
        → PCA  (→ PCA_DIMS, default 150D)    [linear, fast, removes noise]
        → UMAP (→ CLUSTER_UMAP_DIMS, 50D)    [nonlinear manifold structure]

    Returns float32 numpy array of shape (N, CLUSTER_UMAP_DIMS).
    """
    pca_out = _pca_reduce(embeddings, n_components=PCA_DIMS)
    reduced = _umap_reduce(
        pca_out,
        n_components=CLUSTER_UMAP_DIMS,
        n_neighbors=choose_umap_neighbors(group_size, preferred=30),
        min_dist=0.0,       # 0.0 = maximally tight — best for density clustering
        n_epochs=500,
    )
    _free_gpu()
    return reduced


def reduce_for_visualisation(embeddings: np.ndarray, N: int) -> np.ndarray:
    """
    Cheap 3-D reduction for plotting — accuracy is not critical here.
      raw embeddings (1024D)
        → PCA  (→ PCA_DIMS, 150D)   [reuses same intermediate as clustering]
        → UMAP (→ 3D)               [separate pass; min_dist=0.1 for readability]

    Returns float32 numpy array of shape (N, 3).
    """
    pca_out = _pca_reduce(embeddings, n_components=PCA_DIMS)
    coords  = _umap_reduce(
        pca_out,
        n_components=3,
        n_neighbors=choose_umap_neighbors(N, preferred=30),
        min_dist=0.1,       # slight spread so points don't overlap in 3D view
        n_epochs=300,       # fewer epochs fine — visual fidelity, not accuracy
    )
    _free_gpu()
    return coords


def semantic_subclusters(
    embeddings: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """
    Per-group semantic sub-clustering:
      raw embeddings → PCA (PCA_DIMS) → UMAP (CLUSTER_UMAP_DIMS=50) → HDBSCAN(leaf)

    The two-step reduction (PCA then UMAP) is deliberate:
      - PCA linearises the high-dimensional embedding space cheaply
      - UMAP then finds the nonlinear manifold structure in 50D
      - HDBSCAN clusters in that 50D space (no curse of dimensionality)

    Returns cluster labels array of shape (M,). -1 = noise.
    """
    M = embeddings.shape[0]
    if M < 4:
        return np.zeros(M, dtype=int)

    # Accurate two-stage reduction: PCA_DIMS → CLUSTER_UMAP_DIMS
    reduced = reduce_for_clustering(embeddings, group_size=M)

    # min_cluster_size controls cluster granularity — this is the most
    # important HDBSCAN parameter.
    #
    # Old formula (group_size // 100, max 10) was producing mcs=3..10 for
    # groups of 300–1000 titles, which with cluster_selection_method="leaf"
    # created hundreds of micro-clusters per group (5001 total).
    #
    # New formula targets ~15–30 clusters per group:
    #   mcs = group_size / 20  → roughly 20 points per cluster minimum
    #   clamped: at least 10, at most 50
    #
    # cluster_selection_method="eom" (excess of mass) merges micro-clusters
    # up into stable parent clusters — produces far fewer, more meaningful
    # clusters than "leaf" which splits to maximum granularity.
    mcs = max(10, min(group_size // 20, 50))

    if USE_GPU:
        import cupy as cp
        labels = _HDBSCAN(
            min_cluster_size=mcs,
            min_samples=5,                      # raised from 1: requires denser cores
            cluster_selection_method="eom",     # changed from "leaf": merges micro-clusters
            cluster_selection_epsilon=0.0,
            metric="euclidean",
        ).fit_predict(cp.asarray(reduced))
        labels = _to_numpy(labels)
    else:
        labels = _HDBSCAN(
            min_cluster_size=mcs,
            min_samples=5,
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.0,
            metric="euclidean",
            core_dist_n_jobs=-1,
        ).fit_predict(reduced)

    _free_gpu()
    return labels.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation  —  two-level drill-down (outer groups → inner sub-clusters)
# ─────────────────────────────────────────────────────────────────────────────
#
# Performance problem with the naive approach:
#   - 120k scatter points in one WebGL scene → laggy even on fast machines
#   - 500 Plotly legend traces → each is a separate DOM element → browser freeze
#
# Solution — lazy drill-down:
#   Level 0 (overview): only ~50 outer-group centroids shown as large labelled
#     markers.  Tiny dataset, instant render.
#   Level 1 (drill-in): clicking a centroid replaces the scene with only that
#     group's points (~300 average), coloured by sub-cluster.  Fast because
#     it's a small slice.
#   Back button returns to overview.
#
# All data is embedded as JSON inside a self-contained HTML file.
# Plotly is loaded from CDN — no server needed, works in Colab iframe.
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def _build_viz_data(
    embeddings_3d   : np.ndarray,
    combined_labels : np.ndarray,
    combined_topics : dict[int, str],
    sentences       : list[str],
    group_labels    : np.ndarray,
    group_topics    : list[str],
) -> dict:
    """
    Build the JSON payload that the HTML visualiser consumes.

    Structure
    ---------
    {
      "overview": [
        { "group_id": 0, "label": "space, station...", "x": 1.2, "y": -0.5, "z": 0.8,
          "count": 312, "n_sub": 12, "color": "#e6194b" },
        ...
      ],
      "groups": {
        "0": {
          "label": "space, station...",
          "color": "#e6194b",
          "noise": [ {"x":..,"y":..,"z":..,"title":"..."}, ... ],
          "subclusters": [
            { "sub_id": 0, "label": "shuttle launch", "color": "#3cb44b",
              "points": [ {"x":..,"y":..,"z":..,"title":"..."}, ... ] },
            ...
          ]
        },
        ...
      }
    }

    overview is tiny (~50 rows) and loaded immediately.
    groups[g] is loaded only when the user clicks group g.
    """
    import json as _json

    n_groups = len(group_topics)

    # ── overview: one centroid per outer group ────────────────────────────────
    overview = []
    for g in range(n_groups):
        g_mask = group_labels == g
        if not g_mask.any():
            continue
        pts    = embeddings_3d[g_mask]
        cx, cy, cz = pts.mean(axis=0).tolist()

        # count only non-noise points in this group
        g_combined = combined_labels[g_mask]
        n_assigned = int((g_combined != -1).sum())
        n_sub      = len(set(g_combined) - {-1})
        color      = _PALETTE[g % len(_PALETTE)]

        overview.append({
            "group_id" : g,
            "label"    : group_topics[g],
            "x"        : round(cx, 4),
            "y"        : round(cy, 4),
            "z"        : round(cz, 4),
            "count"    : n_assigned,
            "n_sub"    : n_sub,
            "color"    : color,
        })

    # ── per-group detail: sub-clusters + noise ────────────────────────────────
    groups_data: dict[str, dict] = {}
    for g in range(n_groups):
        g_mask   = group_labels == g
        g_idx    = np.where(g_mask)[0]
        if len(g_idx) == 0:
            continue

        g_pts      = embeddings_3d[g_idx]
        g_combined = combined_labels[g_idx]
        g_titles   = [sentences[i] for i in g_idx]
        color      = _PALETTE[g % len(_PALETTE)]

        # noise points for this group
        noise_mask = g_combined == -1
        noise_pts  = []
        for i in np.where(noise_mask)[0]:
            noise_pts.append({
                "x": round(float(g_pts[i, 0]), 4),
                "y": round(float(g_pts[i, 1]), 4),
                "z": round(float(g_pts[i, 2]), 4),
                "title": g_titles[i],
            })

        # sub-clusters
        sub_ids = sorted(set(g_combined) - {-1})
        subclusters = []
        for si, sub_id in enumerate(sub_ids):
            sub_mask = g_combined == sub_id
            sub_pts  = []
            clabel   = g * 10_000 + sub_id
            topic    = combined_topics.get(clabel, f"sub {sub_id}")
            # strip the outer group prefix for the inner label
            inner_label = topic.split("›")[-1].strip() if "›" in topic else topic

            for i in np.where(sub_mask)[0]:
                sub_pts.append({
                    "x": round(float(g_pts[i, 0]), 4),
                    "y": round(float(g_pts[i, 1]), 4),
                    "z": round(float(g_pts[i, 2]), 4),
                    "title": g_titles[i],
                })
            subclusters.append({
                "sub_id" : int(sub_id),
                "label"  : inner_label,
                "color"  : _PALETTE[si % len(_PALETTE)],
                "points" : sub_pts,
            })

        groups_data[str(g)] = {
            "label"       : group_topics[g],
            "color"       : color,
            "noise"       : noise_pts,
            "subclusters" : subclusters,
        }

    return {"overview": overview, "groups": groups_data}


def visualize_hierarchy(
    embeddings_3d   : np.ndarray,
    combined_labels : np.ndarray,
    combined_topics : dict[int, str],
    sentences       : list[str],
    group_labels    : np.ndarray,
    group_topics    : list[str],
    output_html     : str = "/content/clusters_3d.html",
) -> None:
    """
    Build and save the two-level drill-down visualisation.

    Overview loads instantly (one point per outer group).
    Clicking a group replaces the scene with only that group's sub-clusters.
    """
    import json as _json

    print("  Building visualisation data …")
    viz = _build_viz_data(
        embeddings_3d, combined_labels, combined_topics,
        sentences, group_labels, group_topics,
    )
    data_json = _json.dumps(viz)
    n_groups  = len(viz["overview"])
    n_pts     = sum(
        len(sc["points"])
        for g in viz["groups"].values()
        for sc in g["subclusters"]
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hierarchical Cluster Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f0f19; color: #e0e0e0; font-family: monospace; height: 100vh;
         display: flex; flex-direction: column; }}
  #topbar {{ padding: 8px 14px; background: #1a1a2e; border-bottom: 1px solid #333;
             display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
  #back-btn {{ display: none; padding: 5px 14px; background: #333; color: #ccc;
               border: 1px solid #555; border-radius: 4px; cursor: pointer;
               font-family: monospace; font-size: 12px; }}
  #back-btn:hover {{ background: #444; }}
  #title-text {{ font-size: 13px; color: #aaa; }}
  #subtitle {{ font-size: 11px; color: #666; margin-left: auto; }}
  #plot {{ flex: 1; min-height: 0; }}
</style>
</head>
<body>
<div id="topbar">
  <button id="back-btn" onclick="showOverview()">← Back to overview</button>
  <span id="title-text">Overview — {n_groups} keyword groups · {n_pts} clustered points</span>
  <span id="subtitle">Click a group to drill in</span>
</div>
<div id="plot"></div>

<script>
const DATA = {data_json};

const LAYOUT_BASE = {{
  paper_bgcolor: "rgb(15,15,25)",
  font: {{ color: "#e0e0e0", family: "monospace", size: 11 }},
  scene: {{
    bgcolor: "rgb(15,15,25)",
    xaxis: {{ title: "UMAP-1", gridcolor: "rgb(50,50,70)", color: "#888" }},
    yaxis: {{ title: "UMAP-2", gridcolor: "rgb(50,50,70)", color: "#888" }},
    zaxis: {{ title: "UMAP-3", gridcolor: "rgb(50,50,70)", color: "#888" }},
  }},
  margin: {{ l:0, r:0, t:0, b:0 }},
  legend: {{ font: {{ size: 10 }}, itemsizing: "constant" }},
  uirevision: "static",   // keeps camera angle when traces update
}};

let currentCamera = null;

// ── Overview: one large marker per outer group ────────────────────────────
function showOverview() {{
  document.getElementById("back-btn").style.display = "none";
  document.getElementById("title-text").textContent =
    "Overview — {n_groups} keyword groups · {n_pts} clustered points";
  document.getElementById("subtitle").textContent = "Click a group to drill in";

  const ov = DATA.overview;
  const trace = {{
    type: "scatter3d",
    mode: "markers+text",
    x: ov.map(d => d.x),
    y: ov.map(d => d.y),
    z: ov.map(d => d.z),
    text: ov.map(d => d.label.split(",")[0].trim()),   // first keyword only
    textposition: "top center",
    textfont: {{ size: 9, color: "#ccc" }},
    marker: {{
      size: ov.map(d => Math.max(8, Math.min(24, d.count / 30))),
      color: ov.map(d => d.color),
      opacity: 0.9,
      line: {{ width: 1, color: "#fff" }},
    }},
    customdata: ov.map(d => [d.group_id, d.label, d.count, d.n_sub]),
    hovertemplate:
      "<b>%{{customdata[1]}}</b><br>" +
      "%{{customdata[2]}} titles · %{{customdata[3]}} sub-clusters<br>" +
      "<i>Click to drill in</i><extra></extra>",
    showlegend: false,
  }};

  const layout = Object.assign({{}}, LAYOUT_BASE, {{
    title: null,
    scene: Object.assign({{}}, LAYOUT_BASE.scene, {{
      camera: currentCamera || undefined,
    }}),
  }});

  Plotly.react("plot", [trace], layout, {{responsive: true}}).then(() => {{
    const gd = document.getElementById("plot");
    gd.on("plotly_click", evt => {{
      if (!evt.points.length) return;
      const [gid] = evt.points[0].customdata;
      showGroup(gid);
    }});
    gd.on("plotly_relayout", evt => {{
      if (evt["scene.camera"]) currentCamera = evt["scene.camera"];
    }});
  }});
}}

// ── Drill-in: all sub-clusters for one outer group ────────────────────────
function showGroup(gid) {{
  const g = DATA.groups[String(gid)];
  if (!g) return;

  document.getElementById("back-btn").style.display = "inline-block";
  document.getElementById("title-text").textContent =
    "[" + g.label + "]  —  " + g.subclusters.length + " sub-clusters";
  document.getElementById("subtitle").textContent =
    g.noise.length + " noise points (grey)";

  const traces = [];

  // noise trace
  if (g.noise.length) {{
    traces.push({{
      type: "scatter3d", mode: "markers",
      name: "noise (" + g.noise.length + ")",
      x: g.noise.map(p => p.x),
      y: g.noise.map(p => p.y),
      z: g.noise.map(p => p.z),
      customdata: g.noise.map(p => p.title),
      marker: {{ size: 2, color: "rgba(160,160,160,0.2)" }},
      hovertemplate: "%{{customdata}}<extra>noise</extra>",
    }});
  }}

  // one trace per sub-cluster
  g.subclusters.forEach(sc => {{
    traces.push({{
      type: "scatter3d", mode: "markers",
      name: sc.label.length > 45 ? sc.label.slice(0,42)+"…" : sc.label,
      x: sc.points.map(p => p.x),
      y: sc.points.map(p => p.y),
      z: sc.points.map(p => p.z),
      customdata: sc.points.map(p => p.title),
      marker: {{ size: 4, color: sc.color, opacity: 0.85 }},
      hovertemplate: "%{{customdata}}<br><extra>" + sc.label + "</extra>",
    }});
  }});

  const layout = Object.assign({{}}, LAYOUT_BASE, {{
    scene: Object.assign({{}}, LAYOUT_BASE.scene, {{
      camera: currentCamera || undefined,
    }}),
  }});

  Plotly.react("plot", traces, layout, {{responsive: true}}).then(() => {{
    const gd = document.getElementById("plot");
    gd.on("plotly_relayout", evt => {{
      if (evt["scene.camera"]) currentCamera = evt["scene.camera"];
    }});
  }});
}}

// ── Boot ──────────────────────────────────────────────────────────────────
showOverview();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  ✓ Interactive drill-down plot saved → {output_html}")
    print(f"    Overview : {n_groups} group centroids  (loads instantly)")
    print(f"    Drill-in : up to ~{n_pts // max(n_groups,1)} points per group  (loaded on click)")

    # Show inline in Colab
    try:
        from IPython.display import IFrame, display
        display(IFrame(src=output_html, width="100%", height="700px"))
    except Exception:
        print("  (open the HTML file directly in your browser for the interactive view)")



# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 0. Load data ──────────────────────────────────────────────────────────
    print("Loading titles …")
    sentences = load_titles_from_csv()
    if len(sentences) < 2:
        raise ValueError("Need at least 2 titles to cluster.")

    N  = len(sentences)
    fp = data_fingerprint(sentences)
    print(f"Dataset : {N} titles  |  fingerprint : {fp}")
    print(f"Cache   : {CACHE_DIR}\n")

    # ── 1. Embeddings ─────────────────────────────────────────────────────────
    emb_key    = f"embeddings_{fp}"
    embeddings = load_cache(emb_key)

    if embeddings is None:
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        print(f"Computing embeddings on {device} …")
        model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64 if CUDA_AVAILABLE else 16,
            show_progress_bar=True,
        )
        del model
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        save_cache(emb_key, embeddings)

    print(f"Embedding shape : {embeddings.shape}")

    # ── 2. Level 1 — keyword groups ───────────────────────────────────────────
    grp_key      = f"groups_{fp}"
    cached_groups = load_cache(grp_key)

    if cached_groups is None:
        n_groups = max(5, min(50, int(np.sqrt(N / 10))))
        print(f"\nLevel 1 — {n_groups} keyword groups via TF-IDF + KMeans …")
        group_labels, group_topics = assign_keyword_groups(sentences, n_groups=n_groups)
        save_cache(grp_key, (group_labels, group_topics))
    else:
        group_labels, group_topics = cached_groups
        n_groups = len(group_topics)
        print(f"\nLevel 1 — loaded {n_groups} keyword groups")

    for g, topic in enumerate(group_topics):
        count = (group_labels == g).sum()
        print(f"  Group {g:2d}  ({count:5d} titles)  {topic}")

    # ── 3. Level 2 — semantic sub-clusters ────────────────────────────────────
    sub_key   = f"sub_labels_{fp}"
    sub_labels = load_cache(sub_key)

    if sub_labels is None:
        print(f"\nLevel 2 — semantic sub-clustering …")
        # Initialise to -2 (sentinel meaning "not yet processed")
        # so we can distinguish unprocessed groups from genuine HDBSCAN noise (-1)
        sub_labels = np.full(N, -2, dtype=int)

        for g in range(n_groups):
            idx = np.where(group_labels == g)[0]
            M   = len(idx)
            if M < 4:
                # Too small to sub-cluster — treat as a single cluster (label 0)
                sub_labels[idx] = 0
                continue

            subs            = semantic_subclusters(embeddings[idx], group_size=M)
            sub_labels[idx] = subs          # -1 = genuine HDBSCAN noise within group

            n_subs = len(set(subs) - {-1})
            noise  = (subs == -1).sum()
            print(
                f"  Group {g:2d}  '{group_topics[g][:40]}'  "
                f"→  {n_subs} sub-clusters,  {noise} noise pts  "
                f"({100*noise/M:.1f}%)"
            )

        # Any remaining -2 (should not happen if group_labels covers all N) → noise
        sub_labels[sub_labels == -2] = -1

        save_cache(sub_key, sub_labels)
    else:
        print(f"\nLevel 2 — loaded sub-cluster labels")

    # ── 4. Combined labels & topics ───────────────────────────────────────────
    combined_key    = f"combined_{fp}"
    cached_combined = load_cache(combined_key)

    if cached_combined is None:
        # Encode as  group_id * 10000 + sub_id  (sub=-1 stays -1)
        combined_labels = np.where(
            sub_labels == -1,
            -1,
            group_labels * 10_000 + sub_labels,
        ).astype(int)

        combined_topics: dict[int, str] = {}
        for clabel in sorted(set(combined_labels)):
            if clabel == -1:
                continue
            mask  = combined_labels == clabel
            topic = infer_topic([sentences[i] for i in np.where(mask)[0]])
            g     = clabel // 10_000
            s     = clabel % 10_000
            combined_topics[clabel] = f"[{group_topics[g][:25]}] › {topic}"

        save_cache(combined_key, (combined_labels, combined_topics))
    else:
        combined_labels, combined_topics = cached_combined
        print("Combined labels — loaded from cache")

    n_clusters = len(combined_topics)
    n_noise    = (combined_labels == -1).sum()
    print(f"\nFinal : {n_clusters} clusters,  {n_noise} noise points")

    # ── 5. 3-D UMAP for visualisation ────────────────────────────────────────
    # This is a completely separate UMAP pass from the clustering one.
    # Accuracy here doesn't matter — it only needs to look interpretable.
    umap_key      = f"umap3d_{fp}"
    embeddings_3d = load_cache(umap_key)

    if embeddings_3d is None:
        print("\nComputing 3-D UMAP for visualisation (separate from clustering pass) …")
        embeddings_3d = reduce_for_visualisation(embeddings, N=N)
        save_cache(umap_key, embeddings_3d)
    else:
        print("3-D UMAP — loaded from cache")

    # ── 6. Visualise ──────────────────────────────────────────────────────────
    print("\nBuilding drill-down visualisation …")
    visualize_hierarchy(
        embeddings_3d, combined_labels, combined_topics,
        sentences, group_labels, group_topics,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()