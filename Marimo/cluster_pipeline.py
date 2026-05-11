"""
cluster_pipeline.py
====================
Hierarchical news-headline clustering pipeline.
Converted from Colab notebook → standalone Python script (evoc-style).

Usage
-----
    python cluster_pipeline.py --csv train.csv
    python cluster_pipeline.py --csv train.csv --cache-dir ./cache
    python cluster_pipeline.py --csv train.csv --search "covid vaccine side effects"

Requirements
------------
    pip install numpy torch scikit-learn sentence-transformers umap-learn hdbscan
    # GPU (optional): pip install cupy-cuda12x cuml-cu12

External modules expected in the same directory:
    cluster_store.py   (save_cluster_data)
    cluster_search.py  (ClusterSearch)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import hashlib
import html
import json
import os
import pickle
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
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
    from umap import UMAP as _UMAP
    from hdbscan import HDBSCAN as _HDBSCAN


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_CSV       = Path("train.csv")
DEFAULT_CACHE_DIR = Path("./clustering_cache")
PCA_DIMS          = 150
CLUSTER_UMAP_DIMS = 50
_MAX_DRILL_POINTS = 3_000
_MAX_NOISE_POINTS = 300

_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

# ── Regex (compiled once) ──────────────────────────────────────────────────────
_RE_SOURCE_SUFFIX = re.compile(r'\s+-\s+\S+\.\S+$')
_RE_HTML_TAG      = re.compile(r'<[^>]+>')
_RE_NONALPHA      = re.compile(r'[^a-zA-Z0-9\s\'\-]')
_RE_SPACES        = re.compile(r'\s+')
_RE_SHORT_NUM     = re.compile(r'(?<![a-zA-Z0-9\-])\d{1,3}(?![a-zA-Z0-9\-])')
_RE_THOUSANDS     = re.compile(r'(\d),(\d)')


# =============================================================================
# Data loading & cleaning
# =============================================================================

def clean_title(raw: str) -> str:
    s = html.unescape(raw)
    s = html.unescape(s)
    s = _RE_SOURCE_SUFFIX.sub('', s)
    s = _RE_HTML_TAG.sub(' ', s)
    s = _RE_THOUSANDS.sub(r'\1\2', s)
    s = _RE_THOUSANDS.sub(r'\1\2', s)
    s = _RE_NONALPHA.sub(' ', s)
    s = _RE_SHORT_NUM.sub(' ', s)
    s = _RE_SPACES.sub(' ', s)
    return s.strip()


def load_titles_from_csv(csv_file: Path) -> list[str]:
    """Load and clean non-empty values from the 'Title' column of a CSV file."""
    titles: list[str] = []
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("Title") or "").strip()
            if not raw:
                continue
            cleaned = clean_title(raw)
            if cleaned:
                titles.append(cleaned)
    return titles


# =============================================================================
# Cache helpers
# =============================================================================

def resolve_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.pkl"


def save_cache(cache_dir: Path, name: str, obj) -> None:
    path = _cache_path(cache_dir, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ saved  → {path}")


def load_cache(cache_dir: Path, name: str):
    path = _cache_path(cache_dir, name)
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✓ loaded ← {path}")
        return obj
    return None


def data_fingerprint(sentences: list[str]) -> str:
    raw = json.dumps(sentences, ensure_ascii=False).encode()
    return hashlib.md5(raw).hexdigest()[:12]


# =============================================================================
# Small helpers
# =============================================================================

def choose_umap_neighbors(num_samples: int, preferred: int = 30) -> int:
    if num_samples <= 2:
        return 1
    return max(2, min(preferred, num_samples - 1))


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen_words: set[str] = set()
    kept: list[str] = []
    for t in terms:
        words = set(t.split())
        if not words.issubset(seen_words):
            kept.append(t)
            seen_words |= words
    return kept


def infer_topic(sentences: list[str], top_terms: int = 5) -> str:
    if not sentences:
        return "miscellaneous"
    try:
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        )
        mat = vec.fit_transform(sentences)
        mean_scores  = np.asarray(mat.mean(axis=0)).ravel()
        feature_names = vec.get_feature_names_out()
        top_idx   = mean_scores.argsort()[::-1][:top_terms * 3]
        raw_terms = [feature_names[i] for i in top_idx]
        terms     = _dedupe_terms(raw_terms)[:top_terms]
        return ", ".join(terms) if terms else "miscellaneous"
    except Exception:
        return "miscellaneous"


def _to_numpy(arr) -> np.ndarray:
    if USE_GPU:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


def _free_gpu() -> None:
    if USE_GPU:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


# =============================================================================
# Level 1 — keyword groups (TF-IDF + KMeans)
# =============================================================================

def assign_keyword_groups(
    sentences: list[str],
    n_groups: int,
) -> tuple[np.ndarray, list[str]]:
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2),
        max_features=20_000, sublinear_tf=True, min_df=2,
    )
    tfidf_matrix  = vectorizer.fit_transform(sentences)
    feature_names = np.array(vectorizer.get_feature_names_out())

    km = KMeans(n_clusters=n_groups, init="k-means++", n_init=20, max_iter=500, random_state=42)
    group_labels = km.fit_predict(tfidf_matrix)

    group_topics: list[str] = []
    for centroid in km.cluster_centers_:
        top_idx   = centroid.argsort()[::-1][:15]
        raw_words = [feature_names[i] for i in top_idx]
        top_words = ", ".join(_dedupe_terms(raw_words)[:5])
        group_topics.append(top_words)

    return group_labels.astype(int), group_topics, tfidf_matrix


# =============================================================================
# Level 2 — semantic sub-clustering (GPU or CPU)
# =============================================================================

def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    M        = embeddings.shape[0]
    pca_dims = min(n_components, M - 1, embeddings.shape[1])
    arr      = (cp.asarray(embeddings) if USE_GPU else embeddings)
    arr      = _PCA(n_components=pca_dims).fit_transform(arr)
    return _to_numpy(arr)


def _umap_reduce(
    pca_output: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    n_epochs: int,
    metric: str = "euclidean",
) -> np.ndarray:
    M         = pca_output.shape[0]
    umap_dims = min(n_components, pca_output.shape[1], M - 1)

    if USE_GPU:
        import cupy as cp
        arr        = cp.asarray(pca_output)
        umap_model = _UMAP(
            n_components=umap_dims, n_neighbors=n_neighbors,
            min_dist=min_dist, metric=metric, random_state=42, n_epochs=n_epochs,
        )
    else:
        arr        = pca_output
        umap_model = _UMAP(
            n_components=umap_dims, n_neighbors=n_neighbors,
            min_dist=min_dist, metric=metric, random_state=42,
            n_epochs=n_epochs, low_memory=True,
        )

    return _to_numpy(umap_model.fit_transform(arr))


def reduce_for_clustering(embeddings: np.ndarray, group_size: int) -> np.ndarray:
    pca_out = _pca_reduce(embeddings, n_components=PCA_DIMS)
    reduced = _umap_reduce(
        pca_out,
        n_components=CLUSTER_UMAP_DIMS,
        n_neighbors=choose_umap_neighbors(group_size, preferred=30),
        min_dist=0.0,
        n_epochs=500,
    )
    _free_gpu()
    return reduced


def reduce_for_visualisation(embeddings: np.ndarray, N: int) -> np.ndarray:
    pca_out = _pca_reduce(embeddings, n_components=PCA_DIMS)
    coords  = _umap_reduce(
        pca_out,
        n_components=3,
        n_neighbors=choose_umap_neighbors(N, preferred=30),
        min_dist=0.1,
        n_epochs=300,
    )
    _free_gpu()
    return coords


def semantic_subclusters(embeddings: np.ndarray, group_size: int) -> np.ndarray:
    M = embeddings.shape[0]
    if M < 4:
        return np.zeros(M, dtype=int)

    reduced  = reduce_for_clustering(embeddings, group_size=M)
    mcs      = max(3, min(group_size // 40, 40))
    min_samp = max(2, mcs // 3)
    eps      = 0.3

    if USE_GPU:
        import cupy as cp
        labels = _HDBSCAN(
            min_cluster_size=mcs, min_samples=min_samp,
            cluster_selection_method="eom", cluster_selection_epsilon=eps,
            metric="euclidean",
        ).fit_predict(cp.asarray(reduced))
        labels = _to_numpy(labels)
    else:
        labels = _HDBSCAN(
            min_cluster_size=mcs, min_samples=min_samp,
            cluster_selection_method="eom", cluster_selection_epsilon=eps,
            metric="euclidean", core_dist_n_jobs=-1,
        ).fit_predict(reduced)

    _free_gpu()
    return labels.astype(int)


# =============================================================================
# Visualisation — two-level drill-down HTML
# =============================================================================

def _build_viz_data(
    embeddings_3d   : np.ndarray,
    combined_labels : np.ndarray,
    combined_topics : dict[int, str],
    sentences       : list[str],
    group_labels    : np.ndarray,
    group_topics    : list[str],
) -> dict:
    n_groups = len(group_topics)
    rng      = np.random.default_rng(42)

    overview = []
    for g in range(n_groups):
        g_mask = group_labels == g
        if not g_mask.any():
            continue
        pts    = embeddings_3d[g_mask]
        cx, cy, cz = pts.mean(axis=0).tolist()
        g_combined = combined_labels[g_mask]
        n_assigned = int((g_combined != -1).sum())
        n_sub      = len(set(g_combined) - {-1})
        overview.append({
            "group_id": g, "label": group_topics[g],
            "x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4),
            "count": n_assigned, "n_sub": n_sub,
            "color": _PALETTE[g % len(_PALETTE)],
        })

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

        noise_all = np.where(g_combined == -1)[0]
        if len(noise_all) > _MAX_NOISE_POINTS:
            noise_all = rng.choice(noise_all, size=_MAX_NOISE_POINTS, replace=False)
        noise_pts = [
            {"x": round(float(g_pts[i, 0]), 4), "y": round(float(g_pts[i, 1]), 4),
             "z": round(float(g_pts[i, 2]), 4), "title": g_titles[i]}
            for i in noise_all
        ]

        sub_ids = sorted(set(g_combined) - {-1})
        sub_raw: list[tuple] = []
        total_assigned = 0
        for sub_id in sub_ids:
            idxs = np.where(g_combined == sub_id)[0]
            sub_raw.append((sub_id, idxs))
            total_assigned += len(idxs)

        needs_sample = total_assigned > _MAX_DRILL_POINTS
        subclusters  = []
        for si, (sub_id, idxs) in enumerate(sub_raw):
            if needs_sample and total_assigned > 0:
                budget = max(5, round(_MAX_DRILL_POINTS * len(idxs) / total_assigned))
                if len(idxs) > budget:
                    idxs = rng.choice(idxs, size=budget, replace=False)
            clabel      = g * 10_000 + sub_id
            topic       = combined_topics.get(clabel, f"sub {sub_id}")
            inner_label = topic.split("›")[-1].strip() if "›" in topic else topic
            sub_pts = [
                {"x": round(float(g_pts[i, 0]), 4), "y": round(float(g_pts[i, 1]), 4),
                 "z": round(float(g_pts[i, 2]), 4), "title": g_titles[i]}
                for i in idxs
            ]
            subclusters.append({
                "sub_id": int(sub_id), "label": inner_label,
                "color": _PALETTE[si % len(_PALETTE)], "points": sub_pts,
            })

        groups_data[str(g)] = {
            "label": group_topics[g], "color": color,
            "total": total_assigned,
            "sampled": min(total_assigned, _MAX_DRILL_POINTS),
            "noise": noise_pts, "subclusters": subclusters,
        }

    return {"overview": overview, "groups": groups_data}


def visualize_hierarchy(
    embeddings_3d   : np.ndarray,
    combined_labels : np.ndarray,
    combined_topics : dict[int, str],
    sentences       : list[str],
    group_labels    : np.ndarray,
    group_topics    : list[str],
    output_html     : Path = Path("clusters_3d.html"),
) -> None:
    print("  Building visualisation data …")
    viz       = _build_viz_data(
        embeddings_3d, combined_labels, combined_topics,
        sentences, group_labels, group_topics,
    )
    data_json = json.dumps(viz)
    n_groups  = len(viz["overview"])
    n_pts     = sum(
        len(sc["points"]) for g in viz["groups"].values() for sc in g["subclusters"]
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
const REV_OVERVIEW = "overview-v1";
let   revCounter   = 0;

function makeLayout(uirev) {{
  return {{
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
    uirevision: uirev,
  }};
}}

function showOverview() {{
  document.getElementById("back-btn").style.display = "none";
  document.getElementById("title-text").textContent =
    "Overview — {n_groups} keyword groups · {n_pts} clustered points";
  document.getElementById("subtitle").textContent = "Click a group to drill in";
  const ov = DATA.overview;
  const trace = {{
    type: "scatter3d", mode: "markers+text",
    x: ov.map(d => d.x), y: ov.map(d => d.y), z: ov.map(d => d.z),
    text: ov.map(d => d.label.split(",")[0].trim()),
    textposition: "top center",
    textfont: {{ size: 9, color: "#ccc" }},
    marker: {{
      size: ov.map(d => Math.max(8, Math.min(24, d.count / 30))),
      color: ov.map(d => d.color), opacity: 0.9,
      line: {{ width: 1, color: "#fff" }},
    }},
    customdata: ov.map(d => [d.group_id, d.label, d.count, d.n_sub]),
    hovertemplate:
      "<b>%{{customdata[1]}}</b><br>" +
      "%{{customdata[2]}} titles · %{{customdata[3]}} sub-clusters<br>" +
      "<i>Click to drill in</i><extra></extra>",
    showlegend: false,
  }};
  Plotly.newPlot("plot", [trace], makeLayout(REV_OVERVIEW), {{responsive: true}}).then(() => {{
    const gd = document.getElementById("plot");
    gd.on("plotly_click", evt => {{
      if (!evt.points.length) return;
      const [gid] = evt.points[0].customdata;
      showGroup(gid);
    }});
  }});
}}

function showGroup(gid) {{
  const g = DATA.groups[String(gid)];
  if (!g) return;
  const isSampled = g.sampled !== undefined && g.total !== undefined && g.sampled < g.total;
  const sampledNote = isSampled ? ` (showing ${{g.sampled}} of ${{g.total}})` : "";
  document.getElementById("back-btn").style.display = "inline-block";
  document.getElementById("title-text").textContent =
    "[" + g.label + "]  —  " + g.subclusters.length + " sub-clusters" + sampledNote;
  document.getElementById("subtitle").textContent =
    g.noise.length + " noise pts (grey)" +
    (isSampled ? " · downsampled for performance" : "");
  const traces = [];
  if (g.noise.length) {{
    traces.push({{
      type: "scatter3d", mode: "markers",
      name: "noise (" + g.noise.length + ")",
      x: g.noise.map(p => p.x), y: g.noise.map(p => p.y), z: g.noise.map(p => p.z),
      customdata: g.noise.map(p => p.title),
      marker: {{ size: 2, color: "rgba(160,160,160,0.2)" }},
      hovertemplate: "%{{customdata}}<extra>noise</extra>",
    }});
  }}
  g.subclusters.forEach(sc => {{
    traces.push({{
      type: "scatter3d", mode: "markers",
      name: sc.label.length > 45 ? sc.label.slice(0,42)+"…" : sc.label,
      x: sc.points.map(p => p.x), y: sc.points.map(p => p.y), z: sc.points.map(p => p.z),
      customdata: sc.points.map(p => p.title),
      marker: {{ size: 4, color: sc.color, opacity: 0.85 }},
      hovertemplate: "%{{customdata}}<br><extra>" + sc.label + "</extra>",
    }});
  }});
  revCounter++;
  Plotly.newPlot("plot", traces, makeLayout("drill-" + revCounter), {{responsive: true}});
}}

showOverview();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  ✓ Interactive drill-down plot saved → {output_html}")
    print(f"    Overview : {n_groups} group centroids  (loads instantly)")
    print(f"    Drill-in : up to {_MAX_DRILL_POINTS} pts per group  (downsampled if larger)")


# =============================================================================
# Main pipeline
# =============================================================================

def main(
    csv_file  : Path = DEFAULT_CSV,
    cache_dir : Path = DEFAULT_CACHE_DIR,
    output_html: Path = Path("clusters_3d.html"),
    search_query: str | None = None,
) -> None:

    cache_dir = resolve_cache_dir(cache_dir)

    # ── 0. Load data ──────────────────────────────────────────────────────────
    print("Loading titles …")
    sentences = load_titles_from_csv(csv_file)
    if len(sentences) < 2:
        raise ValueError("Need at least 2 titles to cluster.")

    N  = len(sentences)
    fp = data_fingerprint(sentences)
    print(f"Dataset : {N} titles  |  fingerprint : {fp}")
    print(f"Cache   : {cache_dir}\n")

    # ── 1. Embeddings ─────────────────────────────────────────────────────────
    emb_key    = f"embeddings_{fp}"
    embeddings = load_cache(cache_dir, emb_key)

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
        save_cache(cache_dir, emb_key, embeddings)

    print(f"Embedding shape : {embeddings.shape}")

    # ── 2. Level 1 — keyword groups ───────────────────────────────────────────
    grp_key       = f"groups_{fp}"
    cached_groups = load_cache(cache_dir, grp_key)

    if cached_groups is None:
        n_groups = max(5, min(100, int(np.sqrt(N / 10))))
        print(f"\nLevel 1 — {n_groups} keyword groups via TF-IDF + KMeans …")
        group_labels, group_topics, tfidf_matrix = assign_keyword_groups(sentences, n_groups=n_groups)
        save_cache(cache_dir, grp_key, (group_labels, group_topics, tfidf_matrix))
    else:
        group_labels, group_topics, tfidf_matrix = cached_groups
        n_groups = len(group_topics)
        print(f"\nLevel 1 — loaded {n_groups} keyword groups")

    for g, topic in enumerate(group_topics):
        count = (group_labels == g).sum()
        print(f"  Group {g:2d}  ({count:5d} titles)  {topic}")

    # ── 3. Level 2 — semantic sub-clusters ────────────────────────────────────
    sub_key    = f"sub_labels_{fp}"
    sub_labels = load_cache(cache_dir, sub_key)

    if sub_labels is None:
        print(f"\nLevel 2 — semantic sub-clustering …")
        sub_labels = np.full(N, -2, dtype=int)
        MAX_GROUP_SIZE = 5000

        for g in range(n_groups):
            idx = np.where(group_labels == g)[0]
            M   = len(idx)

            if M < 4:
                sub_labels[idx] = 0
                continue

            if M > MAX_GROUP_SIZE:
                n_splits   = max(2, M // 2000)
                sub_km     = KMeans(n_clusters=n_splits, n_init=5, random_state=42)
                local_lbls = sub_km.fit_predict(tfidf_matrix[idx])
                for new_g in range(n_splits):
                    split_mask = local_lbls == new_g
                    split_idx  = idx[split_mask]
                    split_M    = len(split_idx)
                    if split_M < 4:
                        sub_labels[split_idx] = 0
                    else:
                        subs = semantic_subclusters(embeddings[split_idx], group_size=split_M)
                        sub_labels[split_idx] = subs
                continue

            subs            = semantic_subclusters(embeddings[idx], group_size=M)
            sub_labels[idx] = subs
            n_subs = len(set(subs) - {-1})
            noise  = (subs == -1).sum()
            print(
                f"  Group {g:2d}  '{group_topics[g][:40]}'  "
                f"→  {n_subs} sub-clusters,  {noise} noise pts  "
                f"({100*noise/M:.1f}%)"
            )

        sub_labels[sub_labels == -2] = -1
        save_cache(cache_dir, sub_key, sub_labels)
    else:
        print(f"\nLevel 2 — loaded sub-cluster labels")

    # ── 4. Combined labels & topics ───────────────────────────────────────────
    combined_key    = f"combined_{fp}"
    cached_combined = load_cache(cache_dir, combined_key)

    if cached_combined is None:
        combined_labels = np.where(
            sub_labels == -1, -1, group_labels * 10_000 + sub_labels,
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
        save_cache(cache_dir, combined_key, (combined_labels, combined_topics))
    else:
        combined_labels, combined_topics = cached_combined
        print("Combined labels — loaded from cache")

    n_clusters = len(combined_topics)
    n_noise    = (combined_labels == -1).sum()
    print(f"\nFinal : {n_clusters} clusters,  {n_noise} noise points")

    # ── 5. 3-D UMAP for visualisation ─────────────────────────────────────────
    umap_key      = f"umap3d_{fp}"
    embeddings_3d = load_cache(cache_dir, umap_key)

    if embeddings_3d is None:
        print("\nComputing 3-D UMAP for visualisation …")
        embeddings_3d = reduce_for_visualisation(embeddings, N=N)
        save_cache(cache_dir, umap_key, embeddings_3d)
    else:
        print("3-D UMAP — loaded from cache")

    # ── 6. Visualise ──────────────────────────────────────────────────────────
    print("\nBuilding drill-down visualisation …")
    visualize_hierarchy(
        embeddings_3d, combined_labels, combined_topics,
        sentences, group_labels, group_topics,
        output_html=output_html,
    )

    # ── 7. Save cluster store for search ──────────────────────────────────────
    try:
        from cluster_store import save_cluster_data
        print("\nSaving cluster store for search …")
        save_cluster_data(
            sentences=sentences, embeddings=embeddings,
            group_labels=group_labels, group_topics=group_topics,
            sub_labels=sub_labels, combined_labels=combined_labels,
            combined_topics=combined_topics, out_dir=str(cache_dir),
        )
    except ImportError:
        print("\n⚠  cluster_store.py not found — skipping search-store save.")

    # ── 8. Optional CLI search ─────────────────────────────────────────────────
    if search_query:
        try:
            from cluster_search import ClusterSearch
            searcher = ClusterSearch(cache_dir=str(cache_dir))
            results  = searcher.search(
                query=search_query, top_k=5, min_similarity=0.2,
                max_depth=3, min_leaf_size=10, refine=True, top_k_per_level=3,
            )
            searcher.display_results(results[:5], max_sents=5)
        except ImportError:
            print("\n⚠  cluster_search.py not found — skipping search.")

    return dict(
        sentences       = sentences,
        embeddings      = embeddings,
        group_labels    = group_labels,
        group_topics    = group_topics,
        sub_labels      = sub_labels,
        combined_labels = combined_labels,
        combined_topics = combined_topics,
    )


# =============================================================================
# Cache inspection helper
# =============================================================================

def inspect_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    import glob
    existing = (
        glob.glob(str(cache_dir / "*.pkl"))
        + glob.glob(str(cache_dir / "*.npz"))
        + glob.glob(str(cache_dir / "*.json"))
    )
    if existing:
        print(f"Cache contains {len(existing)} file(s) in {cache_dir}:")
        for p in sorted(existing):
            print(f"  {Path(p).name:40s}  {Path(p).stat().st_size/1e6:.1f} MB")
    else:
        print(f"Cache is empty — everything will be computed fresh.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical headline clustering pipeline")
    parser.add_argument("--csv",          type=Path, default=DEFAULT_CSV,            help="Path to train.csv")
    parser.add_argument("--cache-dir",    type=Path, default=DEFAULT_CACHE_DIR,      help="Cache directory")
    parser.add_argument("--output",       type=Path, default=Path("clusters_3d.html"), help="3-D drill-down HTML path")
    parser.add_argument("--search",       type=str,  default=None,                   help="Run a search query after clustering")
    parser.add_argument("--inspect-cache",action="store_true",                       help="Show cache contents and exit")

    # ── Visualisation flags ────────────────────────────────────────────────────
    parser.add_argument("--viz",          action="store_true",
                        help="Build all cluster visualisations after clustering")
    parser.add_argument("--viz-dashboard",action="store_true",
                        help="Build a single tabbed dashboard (all 6 views, one file)")
    parser.add_argument("--viz-dir",      type=Path, default=Path("."),
                        help="Output directory for visualisation HTML files (default: .)")
    parser.add_argument("--viz-parallel", action="store_true", help="Parallel coordinates only")
    parser.add_argument("--viz-heatmap",  action="store_true", help="Centroid heatmap only")
    parser.add_argument("--viz-scatter",  action="store_true", help="Scatter matrix only")
    parser.add_argument("--viz-treemap",  action="store_true", help="Treemap only")
    parser.add_argument("--viz-ridgeline",action="store_true", help="Density ridgeline only")
    parser.add_argument("--viz-sunburst", action="store_true", help="Sunburst only")
    args = parser.parse_args()

    if args.inspect_cache:
        inspect_cache(args.cache_dir)
    else:
        pipeline_data = main(
            csv_file    = args.csv,
            cache_dir   = args.cache_dir,
            output_html = args.output,
            search_query= args.search,
        )

        # ── Visualisation ──────────────────────────────────────────────────────
        any_viz = any([
            args.viz, args.viz_dashboard,
            args.viz_parallel, args.viz_heatmap, args.viz_scatter,
            args.viz_treemap, args.viz_ridgeline, args.viz_sunburst,
        ])

        if any_viz:
            try:
                from cluster_viz import ClusterViz, build_all_visualisations

                if args.viz or args.viz_dashboard:
                    # Bulk render
                    build_all_visualisations(
                        **pipeline_data,
                        out_dir        = str(args.viz_dir),
                        dashboard_only = args.viz_dashboard,
                    )
                else:
                    # Individual views
                    viz = ClusterViz(**pipeline_data)
                    out = args.viz_dir
                    out.mkdir(parents=True, exist_ok=True)
                    if args.viz_parallel:
                        viz.parallel_coords (output_html=str(out / "viz_parallel.html"))
                    if args.viz_heatmap:
                        viz.heatmap         (output_html=str(out / "viz_heatmap.html"))
                    if args.viz_scatter:
                        viz.scatter_matrix  (output_html=str(out / "viz_scatter.html"))
                    if args.viz_treemap:
                        viz.cluster_treemap (output_html=str(out / "viz_treemap.html"))
                    if args.viz_ridgeline:
                        viz.density_ridgeline(output_html=str(out / "viz_ridgeline.html"))
                    if args.viz_sunburst:
                        viz.sunburst        (output_html=str(out / "viz_sunburst.html"))

            except ImportError:
                print("\n⚠  cluster_viz.py not found — place it in the same directory.")
