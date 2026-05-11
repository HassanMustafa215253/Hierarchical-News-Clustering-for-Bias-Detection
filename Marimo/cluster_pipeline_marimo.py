"""
cluster_pipeline_marimo.py
===========================
Hierarchical news-headline clustering pipeline — Marimo notebook.

Run with:
    marimo run  cluster_pipeline_marimo.py   # read-only app mode
    marimo edit cluster_pipeline_marimo.py   # editable notebook mode

Requirements:
    pip install marimo numpy torch scikit-learn sentence-transformers umap-learn hdbscan
    # GPU (optional): pip install cupy-cuda12x cuml-cu12

External modules expected in the same directory:
    cluster_store.py   (save_cluster_data)
    cluster_search.py  (ClusterSearch)
"""

import marimo

__generated_with = "0.6.0"
app = marimo.App(width="wide", app_title="Hierarchical Cluster Explorer")


# ── Cell 1: Imports & GPU detection ──────────────────────────────────────────
@app.cell
def imports():
    from __future__ import annotations
    import csv, hashlib, html, json, os, pickle, re, warnings
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sentence_transformers import SentenceTransformer

    warnings.filterwarnings("ignore")

    CUDA_AVAILABLE = torch.cuda.is_available()

    if CUDA_AVAILABLE:
        try:
            import cupy as cp
            from cuml.decomposition import PCA as _PCA
            from cuml.manifold import UMAP as _UMAP
            from cuml.cluster import HDBSCAN as _HDBSCAN
            USE_GPU = True
            _gpu_status = "✓ RAPIDS / cuML detected — using GPU pipeline"
        except ImportError:
            USE_GPU = False
            _gpu_status = "⚠ CUDA found but cuML not installed — CPU fallback"
    else:
        USE_GPU = False
        _gpu_status = "✓ No GPU — using CPU pipeline"

    if not USE_GPU:
        from sklearn.decomposition import PCA as _PCA
        from umap import UMAP as _UMAP
        from hdbscan import HDBSCAN as _HDBSCAN

    mo.md(f"## ⚙️ Environment\n```\n{_gpu_status}\n```")
    return (
        mo, np, torch, Path, csv, hashlib, html, json, os, pickle, re,
        TfidfVectorizer, KMeans, SentenceTransformer,
        CUDA_AVAILABLE, USE_GPU, _PCA, _UMAP, _HDBSCAN,
    )


# ── Cell 2: Configuration UI ─────────────────────────────────────────────────
@app.cell
def config(mo):
    csv_path_input = mo.ui.text(
        value="/content/train.csv",
        label="CSV file path",
        full_width=True,
    )
    cache_dir_input = mo.ui.text(
        value="/content/drive/MyDrive/clustering_cache",
        label="Cache directory",
        full_width=True,
    )
    output_html_input = mo.ui.text(
        value="/content/clusters_3d.html",
        label="Output HTML path",
        full_width=True,
    )
    run_btn = mo.ui.run_button(label="▶ Run Pipeline")

    mo.md(f"""
## 📁 Configuration
{mo.hstack([csv_path_input, cache_dir_input, output_html_input], justify="start")}

{run_btn}
""")
    return csv_path_input, cache_dir_input, output_html_input, run_btn


# ── Cell 3: Constants & compiled regexes ─────────────────────────────────────
@app.cell
def constants(re):
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

    _RE_SOURCE_SUFFIX = re.compile(r'\s+-\s+\S+\.\S+$')
    _RE_HTML_TAG      = re.compile(r'<[^>]+>')
    _RE_NONALPHA      = re.compile(r'[^a-zA-Z0-9\s\'\-]')
    _RE_SPACES        = re.compile(r'\s+')
    _RE_SHORT_NUM     = re.compile(r'(?<![a-zA-Z0-9\-])\d{1,3}(?![a-zA-Z0-9\-])')
    _RE_THOUSANDS     = re.compile(r'(\d),(\d)')

    return (
        PCA_DIMS, CLUSTER_UMAP_DIMS, _MAX_DRILL_POINTS, _MAX_NOISE_POINTS,
        _PALETTE, _RE_SOURCE_SUFFIX, _RE_HTML_TAG, _RE_NONALPHA,
        _RE_SPACES, _RE_SHORT_NUM, _RE_THOUSANDS,
    )


# ── Cell 4: Helper functions ──────────────────────────────────────────────────
@app.cell
def helpers(
    html, json, hashlib, os, pickle,
    np, TfidfVectorizer,
    USE_GPU, _PCA, _UMAP, _HDBSCAN,
    PCA_DIMS, CLUSTER_UMAP_DIMS,
    _RE_SOURCE_SUFFIX, _RE_HTML_TAG, _RE_NONALPHA,
    _RE_SPACES, _RE_SHORT_NUM, _RE_THOUSANDS,
):
    # ── Text cleaning ──────────────────────────────────────────────────────────
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

    # ── Cache helpers ──────────────────────────────────────────────────────────
    def resolve_cache_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def _cache_path(cache_dir: str, name: str) -> str:
        return os.path.join(cache_dir, f"{name}.pkl")

    def save_cache(cache_dir: str, name: str, obj) -> None:
        with open(_cache_path(cache_dir, name), "wb") as f:
            pickle.dump(obj, f)

    def load_cache(cache_dir: str, name: str):
        path = _cache_path(cache_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def data_fingerprint(sentences: list) -> str:
        raw = json.dumps(sentences, ensure_ascii=False).encode()
        return hashlib.md5(raw).hexdigest()[:12]

    # ── Misc helpers ───────────────────────────────────────────────────────────
    def choose_umap_neighbors(num_samples: int, preferred: int = 30) -> int:
        if num_samples <= 2:
            return 1
        return max(2, min(preferred, num_samples - 1))

    def _dedupe_terms(terms: list) -> list:
        seen_words: set = set()
        kept: list = []
        for t in terms:
            words = set(t.split())
            if not words.issubset(seen_words):
                kept.append(t)
                seen_words |= words
        return kept

    def infer_topic(sentences: list, top_terms: int = 5) -> str:
        if not sentences:
            return "miscellaneous"
        try:
            vec = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            )
            mat           = vec.fit_transform(sentences)
            mean_scores   = np.asarray(mat.mean(axis=0)).ravel()
            feature_names = vec.get_feature_names_out()
            top_idx       = mean_scores.argsort()[::-1][:top_terms * 3]
            raw_terms     = [feature_names[i] for i in top_idx]
            terms         = _dedupe_terms(raw_terms)[:top_terms]
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

    return (
        clean_title, resolve_cache_dir, save_cache, load_cache,
        data_fingerprint, choose_umap_neighbors, _dedupe_terms,
        infer_topic, _to_numpy, _free_gpu,
    )


# ── Cell 5: Clustering algorithms ────────────────────────────────────────────
@app.cell
def algorithms(
    np, TfidfVectorizer, KMeans,
    USE_GPU, _PCA, _UMAP, _HDBSCAN,
    PCA_DIMS, CLUSTER_UMAP_DIMS,
    choose_umap_neighbors, _dedupe_terms, _to_numpy, _free_gpu,
):
    def assign_keyword_groups(sentences, n_groups):
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            max_features=20_000, sublinear_tf=True, min_df=2,
        )
        tfidf_matrix  = vectorizer.fit_transform(sentences)
        feature_names = np.array(vectorizer.get_feature_names_out())

        km = KMeans(
            n_clusters=n_groups, init="k-means++",
            n_init=20, max_iter=500, random_state=42,
        )
        group_labels = km.fit_predict(tfidf_matrix)

        group_topics = []
        for centroid in km.cluster_centers_:
            top_idx   = centroid.argsort()[::-1][:15]
            raw_words = [feature_names[i] for i in top_idx]
            top_words = ", ".join(_dedupe_terms(raw_words)[:5])
            group_topics.append(top_words)

        return group_labels.astype(int), group_topics, tfidf_matrix

    def _pca_reduce(embeddings, n_components):
        M        = embeddings.shape[0]
        pca_dims = min(n_components, M - 1, embeddings.shape[1])
        arr      = (cp.asarray(embeddings) if USE_GPU else embeddings)  # type: ignore
        arr      = _PCA(n_components=pca_dims).fit_transform(arr)
        return _to_numpy(arr)

    def _umap_reduce(pca_output, n_components, n_neighbors, min_dist, n_epochs, metric="euclidean"):
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

    def reduce_for_clustering(embeddings, group_size):
        pca_out = _pca_reduce(embeddings, n_components=PCA_DIMS)
        reduced = _umap_reduce(
            pca_out,
            n_components=CLUSTER_UMAP_DIMS,
            n_neighbors=choose_umap_neighbors(group_size, preferred=30),
            min_dist=0.0, n_epochs=500,
        )
        _free_gpu()
        return reduced

    def reduce_for_visualisation(embeddings, N):
        pca_out = _pca_reduce(embeddings, n_components=PCA_DIMS)
        coords  = _umap_reduce(
            pca_out, n_components=3,
            n_neighbors=choose_umap_neighbors(N, preferred=30),
            min_dist=0.1, n_epochs=300,
        )
        _free_gpu()
        return coords

    def semantic_subclusters(embeddings, group_size):
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

    return (
        assign_keyword_groups, reduce_for_clustering,
        reduce_for_visualisation, semantic_subclusters,
    )


# ── Cell 6: Load & display data ───────────────────────────────────────────────
@app.cell
def load_data(
    mo, csv, Path,
    csv_path_input, run_btn,
    clean_title, data_fingerprint,
    resolve_cache_dir, cache_dir_input,
    load_cache, save_cache,
    np,
    SentenceTransformer, torch, CUDA_AVAILABLE,
):
    mo.stop(not run_btn.value, mo.md("*Configure settings above and click **▶ Run Pipeline**.*"))

    csv_file  = Path(csv_path_input.value)
    cache_dir = resolve_cache_dir(cache_dir_input.value)

    with mo.status.spinner("Loading titles from CSV …"):
        titles: list = []
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get("Title") or "").strip()
                if not raw:
                    continue
                cleaned = clean_title(raw)
                if cleaned:
                    titles.append(cleaned)

    if len(titles) < 2:
        mo.stop(True, mo.callout(mo.md("**Error:** CSV needs at least 2 titles."), kind="danger"))

    N  = len(titles)
    fp = data_fingerprint(titles)

    # Embeddings
    emb_key    = f"embeddings_{fp}"
    embeddings = load_cache(cache_dir, emb_key)
    emb_source = "cache"

    if embeddings is None:
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        with mo.status.spinner(f"Computing embeddings on {device} (first run only) …"):
            model      = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
            embeddings = model.encode(
                titles, convert_to_numpy=True, normalize_embeddings=True,
                batch_size=64 if CUDA_AVAILABLE else 16, show_progress_bar=False,
            )
            del model
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
            save_cache(cache_dir, emb_key, embeddings)
        emb_source = "computed"

    mo.md(f"""
## 📊 Dataset Loaded

| Stat | Value |
|---|---|
| Titles | **{N:,}** |
| Fingerprint | `{fp}` |
| Embedding shape | `{embeddings.shape}` |
| Embeddings | {emb_source} |
| Cache dir | `{cache_dir}` |
""")
    return titles, N, fp, embeddings, cache_dir


# ── Cell 7: Level 1 — keyword groups ─────────────────────────────────────────
@app.cell
def keyword_groups(
    mo, np,
    titles, N, fp, embeddings, cache_dir,
    assign_keyword_groups,
    load_cache, save_cache,
    KMeans,
):
    grp_key       = f"groups_{fp}"
    cached_groups = load_cache(cache_dir, grp_key)

    if cached_groups is None:
        n_groups = max(5, min(100, int(np.sqrt(N / 10))))
        with mo.status.spinner(f"Level 1 — {n_groups} keyword groups via TF-IDF + KMeans …"):
            group_labels, group_topics, tfidf_matrix = assign_keyword_groups(titles, n_groups=n_groups)
            save_cache(cache_dir, grp_key, (group_labels, group_topics, tfidf_matrix))
        grp_source = "computed"
    else:
        group_labels, group_topics, tfidf_matrix = cached_groups
        n_groups   = len(group_topics)
        grp_source = "cache"

    # Build display table
    rows = []
    for _g, _topic in enumerate(group_topics):
        count = int((group_labels == _g).sum())
        rows.append({"Group": _g, "Titles": count, "Top Keywords": _topic})

    mo.md(f"""
## 🗂️ Level 1 — Keyword Groups ({grp_source})

{mo.ui.table(rows, selection=None)}
""")
    return group_labels, group_topics, tfidf_matrix, n_groups


# ── Cell 8: Level 2 — semantic sub-clusters ───────────────────────────────────
@app.cell
def sub_clustering(
    mo, np,
    titles, N, fp, embeddings, cache_dir,
    group_labels, group_topics, tfidf_matrix, n_groups,
    semantic_subclusters,
    load_cache, save_cache, KMeans,
):
    sub_key    = f"sub_labels_{fp}"
    sub_labels = load_cache(cache_dir, sub_key)
    sub_source = "cache"

    if sub_labels is None:
        sub_source  = "computed"
        sub_labels  = np.full(N, -2, dtype=int)
        MAX_GROUP   = 5000
        _prog_steps = []

        with mo.status.spinner("Level 2 — semantic sub-clustering (may take a while) …"):
            for _g in range(n_groups):
                idx = np.where(group_labels == _g)[0]
                M   = len(idx)
                if M < 4:
                    sub_labels[idx] = 0
                    continue
                if M > MAX_GROUP:
                    n_splits   = max(2, M // 2000)
                    sub_km     = KMeans(n_clusters=n_splits, n_init=5, random_state=42)
                    local_lbls = sub_km.fit_predict(tfidf_matrix[idx])
                    for new_g in range(n_splits):
                        split_idx = idx[local_lbls == new_g]
                        if len(split_idx) < 4:
                            sub_labels[split_idx] = 0
                        else:
                            subs = semantic_subclusters(embeddings[split_idx], group_size=len(split_idx))
                            sub_labels[split_idx] = subs
                    continue
                subs            = semantic_subclusters(embeddings[idx], group_size=M)
                sub_labels[idx] = subs
                n_subs = len(set(subs) - {-1})
                noise  = int((subs == -1).sum())
                _prog_steps.append(f"Group {_g:2d} '{group_topics[_g][:35]}' → {n_subs} sub-clusters, {noise} noise")

        sub_labels[sub_labels == -2] = -1
        save_cache(cache_dir, sub_key, sub_labels)

    total_noise  = int((sub_labels == -1).sum())
    total_signal = int((sub_labels >= 0).sum())
    mo.md(f"""
## 🔬 Level 2 — Semantic Sub-clusters ({sub_source})

| | Count |
|---|---|
| Assigned points | **{total_signal:,}** |
| Noise points | **{total_noise:,}** ({100*total_noise/N:.1f}%) |
""")
    return sub_labels,


# ── Cell 9: Combined labels & topics ─────────────────────────────────────────
@app.cell
def combined(
    mo, np,
    titles, fp, cache_dir,
    group_labels, group_topics, sub_labels,
    infer_topic, load_cache, save_cache,
):
    combined_key    = f"combined_{fp}"
    cached_combined = load_cache(cache_dir, combined_key)

    if cached_combined is None:
        with mo.status.spinner("Building combined cluster labels …"):
            combined_labels = np.where(
                sub_labels == -1, -1, group_labels * 10_000 + sub_labels,
            ).astype(int)
            combined_topics: dict = {}
            for clabel in sorted(set(combined_labels)):
                if clabel == -1:
                    continue
                mask  = combined_labels == clabel
                topic = infer_topic([titles[i] for i in np.where(mask)[0]])
                g     = clabel // 10_000
                s     = clabel % 10_000
                combined_topics[clabel] = f"[{group_topics[g][:25]}] › {topic}"
            save_cache(cache_dir, combined_key, (combined_labels, combined_topics))
    else:
        combined_labels, combined_topics = cached_combined

    n_clusters = len(combined_topics)
    n_noise    = int((combined_labels == -1).sum())
    mo.md(f"""
## 🏷️ Combined Cluster Labels

**{n_clusters} total clusters**, **{n_noise} noise points**
""")
    return combined_labels, combined_topics


# ── Cell 10: 3-D UMAP ─────────────────────────────────────────────────────────
@app.cell
def umap_3d(
    mo, fp, N, embeddings, cache_dir,
    reduce_for_visualisation,
    load_cache, save_cache,
):
    umap_key      = f"umap3d_{fp}"
    embeddings_3d = load_cache(cache_dir, umap_key)
    umap_source   = "cache"

    if embeddings_3d is None:
        with mo.status.spinner("Computing 3-D UMAP for visualisation …"):
            embeddings_3d = reduce_for_visualisation(embeddings, N=N)
            save_cache(cache_dir, umap_key, embeddings_3d)
        umap_source = "computed"

    mo.md(f"## 🗺️ 3-D UMAP ({umap_source}) — shape `{embeddings_3d.shape}`")
    return embeddings_3d,


# ── Cell 11: Build & display the HTML visualisation ───────────────────────────
@app.cell
def visualise(
    mo, json, np,
    embeddings_3d, combined_labels, combined_topics,
    titles, group_labels, group_topics,
    output_html_input,
    _PALETTE, _MAX_DRILL_POINTS, _MAX_NOISE_POINTS,
):
    # ── Build viz data ────────────────────────────────────────────────────────
    n_groups_viz = len(group_topics)
    rng          = np.random.default_rng(42)
    overview     = []

    for _g in range(n_groups_viz):
        g_mask = group_labels == _g
        if not g_mask.any():
            continue
        pts    = embeddings_3d[g_mask]
        cx, cy, cz = pts.mean(axis=0).tolist()
        g_combined = combined_labels[g_mask]
        overview.append({
            "group_id": _g, "label": group_topics[_g],
            "x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4),
            "count": int((g_combined != -1).sum()),
            "n_sub": len(set(g_combined) - {-1}),
            "color": _PALETTE[_g % len(_PALETTE)],
        })

    groups_data: dict = {}
    for _g in range(n_groups_viz):
        g_mask   = group_labels == _g
        g_idx    = np.where(g_mask)[0]
        if len(g_idx) == 0:
            continue
        g_pts      = embeddings_3d[g_idx]
        g_combined = combined_labels[g_idx]
        g_titles   = [titles[i] for i in g_idx]
        color      = _PALETTE[_g % len(_PALETTE)]

        noise_all = np.where(g_combined == -1)[0]
        if len(noise_all) > _MAX_NOISE_POINTS:
            noise_all = rng.choice(noise_all, size=_MAX_NOISE_POINTS, replace=False)
        noise_pts = [
            {"x": round(float(g_pts[i, 0]), 4), "y": round(float(g_pts[i, 1]), 4),
             "z": round(float(g_pts[i, 2]), 4), "title": g_titles[i]}
            for i in noise_all
        ]

        sub_ids = sorted(set(g_combined) - {-1})
        sub_raw = [(sid, np.where(g_combined == sid)[0]) for sid in sub_ids]
        total_assigned = sum(len(x[1]) for x in sub_raw)
        needs_sample   = total_assigned > _MAX_DRILL_POINTS

        subclusters = []
        for si, (sub_id, idxs) in enumerate(sub_raw):
            if needs_sample and total_assigned > 0:
                budget = max(5, round(_MAX_DRILL_POINTS * len(idxs) / total_assigned))
                if len(idxs) > budget:
                    idxs = rng.choice(idxs, size=budget, replace=False)
            viz_clabel = _g * 10_000 + sub_id
            viz_topic = combined_topics.get(viz_clabel, f"sub {sub_id}")
            inner_label = (
                viz_topic.split("›")[-1].strip()
                if "›" in viz_topic
                else viz_topic
            )
            sub_pts     = [
                {"x": round(float(g_pts[i, 0]), 4), "y": round(float(g_pts[i, 1]), 4),
                 "z": round(float(g_pts[i, 2]), 4), "title": g_titles[i]}
                for i in idxs
            ]
            subclusters.append({
                "sub_id": int(sub_id), "label": inner_label,
                "color": _PALETTE[si % len(_PALETTE)], "points": sub_pts,
            })
        groups_data[str(_g)] = {
            "label": group_topics[_g], "color": color,
            "total": total_assigned,
            "sampled": min(total_assigned, _MAX_DRILL_POINTS),
            "noise": noise_pts, "subclusters": subclusters,
        }

    viz = {"overview": overview, "groups": groups_data}

    # ── Save HTML ──────────────────────────────────────────────────────────────
    data_json  = json.dumps(viz)
    n_grp_ov   = len(overview)
    n_pts_ov   = sum(len(sc["points"]) for g in groups_data.values() for sc in g["subclusters"])
    output_html = output_html_input.value

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
               border: 1px solid #555; border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 12px; }}
  #back-btn:hover {{ background: #444; }}
  #title-text {{ font-size: 13px; color: #aaa; }}
  #subtitle {{ font-size: 11px; color: #666; margin-left: auto; }}
  #plot {{ flex: 1; min-height: 0; }}
</style>
</head>
<body>
<div id="topbar">
  <button id="back-btn" onclick="showOverview()">← Back to overview</button>
  <span id="title-text">Overview — {n_grp_ov} groups · {n_pts_ov} pts</span>
  <span id="subtitle">Click a group to drill in</span>
</div>
<div id="plot"></div>
<script>
const DATA={data_json};const REV_OVERVIEW="overview-v1";let revCounter=0;
function makeLayout(r){{return{{paper_bgcolor:"rgb(15,15,25)",font:{{color:"#e0e0e0",family:"monospace",size:11}},scene:{{bgcolor:"rgb(15,15,25)",xaxis:{{title:"UMAP-1",gridcolor:"rgb(50,50,70)",color:"#888"}},yaxis:{{title:"UMAP-2",gridcolor:"rgb(50,50,70)",color:"#888"}},zaxis:{{title:"UMAP-3",gridcolor:"rgb(50,50,70)",color:"#888"}}}},margin:{{l:0,r:0,t:0,b:0}},legend:{{font:{{size:10}},itemsizing:"constant"}},uirevision:r}};}}
function showOverview(){{document.getElementById("back-btn").style.display="none";document.getElementById("title-text").textContent="Overview — {n_grp_ov} groups · {n_pts_ov} pts";document.getElementById("subtitle").textContent="Click a group to drill in";const ov=DATA.overview;const trace={{type:"scatter3d",mode:"markers+text",x:ov.map(d=>d.x),y:ov.map(d=>d.y),z:ov.map(d=>d.z),text:ov.map(d=>d.label.split(",")[0].trim()),textposition:"top center",textfont:{{size:9,color:"#ccc"}},marker:{{size:ov.map(d=>Math.max(8,Math.min(24,d.count/30))),color:ov.map(d=>d.color),opacity:0.9,line:{{width:1,color:"#fff"}}}},customdata:ov.map(d=>[d.group_id,d.label,d.count,d.n_sub]),hovertemplate:"<b>%{{customdata[1]}}</b><br>%{{customdata[2]}} titles · %{{customdata[3]}} sub-clusters<br><i>Click to drill in</i><extra></extra>",showlegend:false}};Plotly.newPlot("plot",[trace],makeLayout(REV_OVERVIEW),{{responsive:true}}).then(()=>{{const gd=document.getElementById("plot");gd.on("plotly_click",evt=>{{if(!evt.points.length)return;const[gid]=evt.points[0].customdata;showGroup(gid);}});}});}}
function showGroup(gid){{const g=DATA.groups[String(gid)];if(!g)return;const isSampled=g.sampled!==undefined&&g.total!==undefined&&g.sampled<g.total;document.getElementById("back-btn").style.display="inline-block";document.getElementById("title-text").textContent="["+g.label+"] — "+g.subclusters.length+" sub-clusters"+(isSampled?` (showing ${{g.sampled}} of ${{g.total}})`:"");document.getElementById("subtitle").textContent=g.noise.length+" noise pts"+(isSampled?" · downsampled":"");const traces=[];if(g.noise.length){{traces.push({{type:"scatter3d",mode:"markers",name:"noise ("+g.noise.length+")",x:g.noise.map(p=>p.x),y:g.noise.map(p=>p.y),z:g.noise.map(p=>p.z),customdata:g.noise.map(p=>p.title),marker:{{size:2,color:"rgba(160,160,160,0.2)"}},hovertemplate:"%{{customdata}}<extra>noise</extra>"}});}}g.subclusters.forEach(sc=>{{traces.push({{type:"scatter3d",mode:"markers",name:sc.label.length>45?sc.label.slice(0,42)+"…":sc.label,x:sc.points.map(p=>p.x),y:sc.points.map(p=>p.y),z:sc.points.map(p=>p.z),customdata:sc.points.map(p=>p.title),marker:{{size:4,color:sc.color,opacity:0.85}},hovertemplate:"%{{customdata}}<br><extra>"+sc.label+"</extra>"}});}});revCounter++;Plotly.newPlot("plot",traces,makeLayout("drill-"+revCounter),{{responsive:true}});}}
showOverview();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as _f:
        _f.write(html_content)

    mo.md(f"""
## 🌐 Visualisation Built

Interactive HTML saved to `{output_html}`. Open it in your browser for the 3-D drill-down explorer.

> **Tip:** `python -m http.server 8080` then visit `http://localhost:8080/{output_html}`
""")
    return output_html,


# ── Cell 12: Save cluster store ───────────────────────────────────────────────
@app.cell
def save_store(
    mo,
    titles, embeddings,
    group_labels, group_topics, sub_labels,
    combined_labels, combined_topics, cache_dir,
):
    try:
        from cluster_store import save_cluster_data
        with mo.status.spinner("Saving cluster store …"):
            save_cluster_data(
                sentences=titles, embeddings=embeddings,
                group_labels=group_labels, group_topics=group_topics,
                sub_labels=sub_labels, combined_labels=combined_labels,
                combined_topics=combined_topics, out_dir=str(cache_dir),
            )
        _store_msg = mo.callout(
            mo.md(f"✅ Cluster store saved to `{cache_dir}`"), kind="success"
        )
    except ImportError:
        _store_msg = mo.callout(
            mo.md("⚠️ `cluster_store.py` not found — skipping store save."), kind="warn"
        )

    _store_msg
    return


# ── Cell 13: Interactive semantic search ─────────────────────────────────────
@app.cell
def search_ui(mo, cache_dir):
    _search_query = mo.ui.text(
        placeholder="e.g. covid vaccine side effects",
        label="Search query",
        full_width=True,
    )
    _top_k     = mo.ui.slider(1, 20,   value=5,  label="top_k")
    _max_depth = mo.ui.slider(0, 5,    value=3,  label="max_depth")
    _min_sim   = mo.ui.slider(0.0, 0.9, value=0.2, step=0.05, label="min_similarity")
    _search_btn = mo.ui.run_button(label="🔍 Search")

    mo.md(f"""
## 🔍 Semantic Search

{_search_query}

{mo.hstack([_top_k, _max_depth, _min_sim])}

{_search_btn}
""")
    return _search_query, _top_k, _max_depth, _min_sim, _search_btn, cache_dir


@app.cell
def search_results(
    mo,
    _search_query, _top_k, _max_depth, _min_sim, _search_btn, cache_dir,
):
    mo.stop(not _search_btn.value, None)
    mo.stop(not _search_query.value.strip(), mo.md("*Enter a query above.*"))

    try:
        from cluster_search import ClusterSearch
        with mo.status.spinner(f'Searching for "{_search_query.value}" …'):
            _searcher = ClusterSearch(cache_dir=str(cache_dir))
            _results  = _searcher.search(
                query=_search_query.value.strip(),
                top_k=_top_k.value,
                min_similarity=_min_sim.value,
                max_depth=_max_depth.value,
                refine=True,
            )[:5]

        _rows = []
        for _r in _results:
            _rows.append({
                "Cluster": _r.get("label", ""),
                "Similarity": f"{_r.get('similarity', 0):.3f}",
                "Size": _r.get("size", 0),
                "Sample": " | ".join(_r.get("sentences", [])[:3]),
            })

        mo.md(f"### Results for *{_search_query.value}*") if _rows else None
        mo.ui.table(_rows, selection=None) if _rows else mo.md("*No results above threshold.*")

    except ImportError:
        mo.callout(mo.md("⚠️ `cluster_search.py` not found — search unavailable."), kind="warn")


if __name__ == "__main__":
    app.run()
