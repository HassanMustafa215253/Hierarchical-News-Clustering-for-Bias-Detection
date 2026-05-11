import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="News Bias Detector")


# ============================================================
# CELL 0 — Install dependencies (run once in Colab)
# ============================================================
@app.cell
def install_deps():
    import subprocess, sys
    packages = [
        "marimo", "evoc", "sentence-transformers", "scikit-learn",
        "numpy", "pandas", "plotly", "transformers", "torch",
        "keybert", "feedparser", "requests", "joblib",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All packages installed.")
    return


# ============================================================
# CELL 1 — Imports + GPU detection
# ============================================================
@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline
    from keybert import KeyBERT
    import evoc
    import feedparser
    import requests
    import json
    import time
    import os
    import joblib
    import pickle
    from collections import defaultdict
    from pathlib import Path
    import warnings
    import torch

    warnings.filterwarnings("ignore")

    # ── GPU detection ──────────────────────────────────────────────────────────
    cuda_ok      = torch.cuda.is_available()
    device_id    = 0 if cuda_ok else -1           # HuggingFace pipeline device
    torch_device = torch.device("cuda" if cuda_ok else "cpu")
    st_device    = "cuda" if cuda_ok else "cpu"   # sentence-transformers

    gpu_info = (
        f"🟢 **GPU detected:** {torch.cuda.get_device_name(0)}"
        if cuda_ok
        else "🟡 **No GPU found — running on CPU**"
    )
    print(f"Device: {torch_device}  |  {gpu_info}")
    print("✅ Imports OK")

    return (
        KeyBERT, PCA, Path, defaultdict, evoc, feedparser, go, hf_pipeline,
        json, joblib, mo, np, normalize, os, pd, pickle, px, requests,
        time, torch, warnings, SentenceTransformer,
        cuda_ok, device_id, torch_device, st_device, gpu_info,
    )


# ============================================================
# CELL 2 — Header + GPU banner
# ============================================================
@app.cell
def config_ui(mo, gpu_info):
    mo.md(f"""
# 🔍 News Bias Detector
### Hierarchical Semantic Clustering + NLI Bias Detection

{gpu_info}

**Pipeline:**
1. Fetch news titles from RSS feeds (or manual input)
2. Encode with **SentenceTransformer** on GPU/CPU
3. PCA reduction → **EVōC** hierarchical clustering (2 pre-built layers)
4. On query: cosine routing top-k per layer → deep sub-cluster → **NLI bias labels**
5. **Save / Load** the full index so expensive steps run only once

---
""")
    return


# ============================================================
# CELL 3 — Save/Load path UI
# ============================================================
@app.cell
def save_load_config(mo):
    mo.md("## 💾 Save / Load Index")
    return


@app.cell
def save_load_ui(mo):
    save_dir_input = mo.ui.text(
        label="Index save directory (created if missing)",
        value="./bias_detector_index",
        full_width=True,
    )
    return (save_dir_input,)


@app.cell
def show_save_dir(mo, save_dir_input):
    mo.vstack([save_dir_input])
    return


# ============================================================
# CELL 4 — Persistence helpers
# ============================================================
@app.cell
def persistence_helpers(np, pd, joblib, pickle, json, Path):
    """
    Saved files in <save_dir>/:
      articles.parquet        — raw article DataFrame
      embeddings.npy          — L2-normed embeddings      float32 (N, 384)
      reduced.npy             — PCA-reduced embeddings    float32 (N, 64)
    tsne.npy                — cached 2D t-SNE coords    float32 (N, 2)
      pca.joblib              — fitted sklearn PCA
      cluster_layers.pkl      — list[np.ndarray] from EVoC
      layer_maps.pkl          — list[dict{cid: [idx, …]}]
      layer_centroids.pkl     — list[dict{cid: np.ndarray}]
      cluster_keywords.json   — {str(cid): [kw, …]}
      noise_idxs.npy          — int32 array of noise article indices
      meta.json               — {n_layers, titles}
    """

    def save_index(cluster_index: dict, save_dir: str) -> str:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        cluster_index["df"].to_parquet(d / "articles.parquet", index=False)
        np.save(d / "embeddings.npy", cluster_index["embeddings"].astype("float32"))
        np.save(d / "reduced.npy",    cluster_index["reduced"].astype("float32"))
        np.save(d / "noise_idxs.npy", np.array(cluster_index["noise_idxs"], dtype="int32"))
        if cluster_index.get("tsne") is not None:
            np.save(d / "tsne.npy", cluster_index["tsne"].astype("float32"))
        joblib.dump(cluster_index["pca"], d / "pca.joblib")

        with open(d / "cluster_layers.pkl",  "wb") as f: pickle.dump(cluster_index["layers"],          f)
        with open(d / "layer_maps.pkl",      "wb") as f: pickle.dump(cluster_index["layer_maps"],      f)
        with open(d / "layer_centroids.pkl", "wb") as f: pickle.dump(cluster_index["layer_centroids"], f)

        kw_json = {str(k): v for k, v in cluster_index["cluster_keywords"].items()}
        with open(d / "cluster_keywords.json", "w") as f:
            json.dump(kw_json, f, ensure_ascii=False)

        meta = {"n_layers": cluster_index["n_layers"], "titles": cluster_index["titles"]}
        with open(d / "meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        return str(d.resolve())

    def load_index(save_dir: str) -> dict:
        d = Path(save_dir)
        required = [
            "articles.parquet", "embeddings.npy", "reduced.npy", "pca.joblib",
            "cluster_layers.pkl", "layer_maps.pkl", "layer_centroids.pkl",
            "cluster_keywords.json", "noise_idxs.npy", "meta.json",
        ]
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Index incomplete — missing: {missing}")

        df         = pd.read_parquet(d / "articles.parquet")
        embeddings = np.load(d / "embeddings.npy")
        reduced    = np.load(d / "reduced.npy")
        noise_idxs = np.load(d / "noise_idxs.npy").tolist()
        pca        = joblib.load(d / "pca.joblib")

        with open(d / "cluster_layers.pkl",  "rb") as f: layers          = pickle.load(f)
        with open(d / "layer_maps.pkl",      "rb") as f: layer_maps      = pickle.load(f)
        with open(d / "layer_centroids.pkl", "rb") as f: layer_centroids = pickle.load(f)

        with open(d / "cluster_keywords.json") as f:
            cluster_keywords = {int(k): v for k, v in json.load(f).items()}

        tsne = None
        if (d / "tsne.npy").exists():
            tsne = np.load(d / "tsne.npy")

        layer_sizes = [{cid: len(idxs) for cid, idxs in cmap.items()} for cmap in layer_maps]

        with open(d / "meta.json") as f:
            meta = json.load(f)

        return {
            "df": df, "embeddings": embeddings, "reduced": reduced,
            "noise_idxs": noise_idxs, "pca": pca,
            "layers": layers, "layer_maps": layer_maps,
            "layer_centroids": layer_centroids,
            "cluster_keywords": cluster_keywords,
            "n_layers": meta["n_layers"], "titles": meta["titles"],
            "layer_sizes": layer_sizes, "tsne": tsne,
        }

    def index_exists(save_dir: str) -> bool:
        return (Path(save_dir) / "meta.json").exists()

    return save_index, load_index, index_exists


# ============================================================
# CELL 4B — CSV cleaning + loading
# ============================================================
@app.cell
def csv_cleaning_helpers(Path):
    import csv
    import html
    import re

    # Compiled once at module load to keep per-row cleaning fast.
    _RE_SOURCE_SUFFIX = re.compile(r"\s+-\s+\S+\.\S+$")   # " - www.site.com" at end of title
    _RE_HTML_TAG      = re.compile(r"<[^>]+>")
    _RE_NONALPHA      = re.compile(r"[^a-zA-Z0-9\s\'\-]")
    _RE_SPACES        = re.compile(r"\s+")
    # Strip bare short-digit tokens (1–3 digits) that are space-bounded.
    # These come from HTML entity residue (&#39; → ' → 39 after apostrophe removal)
    # or formatting fragments. Hyphen-attached digits like COVID-19 and G7 are kept.
    # 4-digit numbers (years, etc.) are never touched.
    _RE_SHORT_NUM     = re.compile(r"(?<![a-zA-Z0-9\-])\d{1,3}(?![a-zA-Z0-9\-])")
    # Collapse thousands-separator commas BEFORE punctuation stripping:
    # "1,000" → "1000", "50,000" → "50000". Applied twice for millions.
    _RE_THOUSANDS     = re.compile(r"(\d),(\d)")

    def clean_title(raw: str) -> str:
        """
        Normalise a news headline for both TF-IDF and embedding.

        Steps (in order):
          1. html.unescape x2 : handles double-encoded entities (common in RSS)
             &#39; → '   &amp; → &   &lt; → <   &gt; → >
             &amp;#39; → &#39; → ' (double-encoded)
          2. Strip source suffix: "Some Title - www.washingtonpost.com" → "Some Title"
          3. Strip residual HTML tags (<b>, <i>, etc.)
          4. Collapse formatted numbers: "1 000 jobs" → "1000 jobs"
             Prevents "000" token surviving after comma-stripping.
          5. Remove punctuation that isn't apostrophe or hyphen
             (keeps contractions and hyphenated words intact)
          6. Remove bare short-digit tokens (1–3 digits) left by HTML entity
             residue, e.g. the "39" in "don 39t" that comes from &#39; → ' → 39
             after apostrophe stripping. Real years (4 digits) are kept.
          7. Collapse whitespace and strip
        """
        s = html.unescape(raw)                  # first pass: &amp;#39; → &#39;
        s = html.unescape(s)                    # second pass: &#39; → '
        s = _RE_SOURCE_SUFFIX.sub("", s)        # "Title - site.com" → "Title"
        s = _RE_HTML_TAG.sub(" ", s)            # <b>foo</b> → foo
        s = _RE_THOUSANDS.sub(r"\1\2", s)      # "1,000" → "1000" before comma-strip
        s = _RE_THOUSANDS.sub(r"\1\2", s)      # second pass for "1,000,000"
        s = _RE_NONALPHA.sub(" ", s)            # remove remaining punctuation / symbols
        s = _RE_SHORT_NUM.sub(" ", s)           # remove bare 1-3 digit tokens (entity residue)
        s = _RE_SPACES.sub(" ", s)              # collapse whitespace
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
                if cleaned:  # skip titles that become empty after cleaning
                    titles.append(cleaned)
        return titles

    return clean_title, load_titles_from_csv


# ============================================================
# CELL 5 — RSS sources
# ============================================================
@app.cell
def rss_sources(mo):
    mo.md("## Step 1 — Load Articles")
    rss_feeds = {
        "BBC News":     "http://feeds.bbci.co.uk/news/rss.xml",
        "Reuters":      "https://feeds.reuters.com/reuters/topNews",
        "Al Jazeera":   "https://www.aljazeera.com/xml/rss/all.xml",
        "Fox News":     "https://moxie.foxnews.com/google-publisher/latest.xml",
        "The Guardian": "https://www.theguardian.com/world/rss",
        "NPR":          "https://feeds.npr.org/1001/rss.xml",
        "CNN":          "http://rss.cnn.com/rss/edition.rss",
        "NY Times":     "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    }
    selected_sources = mo.ui.multiselect(
        options=list(rss_feeds.keys()),
        value=["BBC News", "Al Jazeera", "Reuters"],
        label="Select RSS sources",
    )
    max_articles_slider = mo.ui.slider(
        start=20, stop=300, step=10, value=80, label="Max articles to fetch",
    )
    return rss_feeds, selected_sources, max_articles_slider


@app.cell
def show_source_controls(mo, selected_sources, max_articles_slider):
    mo.vstack([selected_sources, max_articles_slider])
    return


# ============================================================
# CELL 6 — Manual article input
# ============================================================
@app.cell
def manual_input_ui(mo):
    manual_titles_area = mo.ui.text_area(
        label="(Optional) Paste additional article titles — one per line",
        placeholder="Scientists discover new species in Amazon\nUS Senate passes new budget bill\n...",
        rows=5,
    )
    return (manual_titles_area,)


@app.cell
def show_manual_input(mo, manual_titles_area):
    mo.vstack([mo.md("**Or add your own titles:**"), manual_titles_area])
    return


# ============================================================
# CELL 6B — CSV input
# ============================================================
@app.cell
def csv_input_ui(mo):
    # Optional local CSV path to merge into fetched titles.
    csv_path_input = mo.ui.text(
        label="(Optional) CSV file path with a 'Title' column",
        placeholder="./archive/train.csv",
        full_width=True,
    )
    return (csv_path_input,)


@app.cell
def show_csv_input(mo, csv_path_input):
    mo.vstack([mo.md("**Or load titles from CSV:**"), csv_path_input])
    return


# ============================================================
# CELL 7 — Fetch articles
# ============================================================
@app.cell
def fetch_button(mo):
    fetch_btn = mo.ui.button(label="📡 Fetch Articles", kind="success")
    return (fetch_btn,)


@app.cell
def show_fetch_button(mo, fetch_btn):
    mo.hstack([fetch_btn])
    return


@app.cell
def fetch_articles(
    fetch_btn, selected_sources, max_articles_slider,
    manual_titles_area, csv_path_input, rss_feeds,
    feedparser, load_titles_from_csv, Path, pd, mo,
):
    fetch_btn  # reactive
    articles = []
    for _src in selected_sources.value:
        _url = rss_feeds[_src]
        try:
            _feed    = feedparser.parse(_url)
            _per_src = max_articles_slider.value // max(len(selected_sources.value), 1)
            for _entry in _feed.entries[:_per_src]:
                _t = _entry.get("title", "").strip()
                if _t:
                    articles.append({
                        "title":   _t,
                        "summary": _entry.get("summary", "")[:300],
                        "source":  _src,
                        "url":     _entry.get("link", ""),
                    })
        except Exception as _e:
            mo.callout(mo.md(f"⚠️ Could not fetch **{_src}**: {_e}"), kind="warn")

    if manual_titles_area.value.strip():
        for _line in manual_titles_area.value.strip().split("\n"):
            _line = _line.strip()
            if _line:
                articles.append({"title": _line, "summary": "", "source": "Manual", "url": ""})

    csv_path = csv_path_input.value.strip()
    if csv_path:
        _path = Path(csv_path)
        if not _path.exists():
            mo.callout(mo.md(f"⚠️ CSV not found: `{_path}`"), kind="warn")
        else:
            try:
                # Clean titles and attach them as a separate source bucket.
                _titles = load_titles_from_csv(_path)
                for _t in _titles:
                    articles.append({"title": _t, "summary": "", "source": "CSV", "url": ""})
            except Exception as _e:
                mo.callout(mo.md(f"⚠️ CSV load failed: `{_e}`"), kind="warn")

    df_articles = pd.DataFrame(articles).drop_duplicates(subset="title").reset_index(drop=True)
    mo.md(f"✅ **{len(df_articles)}** unique articles from **{len(df_articles['source'].unique())}** sources")
    return (df_articles,)


@app.cell
def show_articles_table(mo, df_articles):
    mo.ui.table(df_articles[["source", "title"]].head(30), label="Preview (first 30)", selection=None)
    return


# ============================================================
# CELL 8 — Load models (GPU-aware)
# ============================================================
@app.cell
def load_models_header(mo):
    mo.md("## Step 2 — Models *(first run downloads weights, ~1–2 min)*")
    return


@app.cell
def model_choice_ui(mo):
    model_options = {
        "all-mpnet-base-v2 (default)": "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "bge-large-en-v1.5 (GPU-heavy)": "BAAI/bge-large-en-v1.5",
    }
    model_choice = mo.ui.dropdown(
        options=model_options,
        value="all-mpnet-base-v2 (default)",
        label="SentenceTransformer model",
        full_width=True,
    )
    return (model_choice,)


@app.cell
def show_model_choice(mo, model_choice):
    mo.vstack([model_choice])
    return


@app.cell
def models(
    SentenceTransformer, KeyBERT, hf_pipeline,
    st_device, device_id, cuda_ok, torch, mo, model_choice,
):
    """
    SentenceTransformer → device=st_device, fp16 on GPU
    KeyBERT             → reuses same ST model
    NLI pipeline        → device=device_id, fp16 on GPU
    """
    _model_id = model_choice.value
    with mo.status.spinner(title=f"Loading SentenceTransformer on {st_device.upper()}…"):
        embedder = SentenceTransformer(_model_id, device=st_device)
        if cuda_ok:
            embedder = embedder.half()   # fp16 for speed + memory savings

    with mo.status.spinner(title="Loading KeyBERT (shares encoder)…"):
        kw_model = KeyBERT(model=embedder)

    with mo.status.spinner(title=f"Loading NLI classifier on {'GPU' if cuda_ok else 'CPU'}…"):
        nli_classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_id,
            torch_dtype=torch.float16 if cuda_ok else torch.float32,
        )

    _label = "GPU fp16" if cuda_ok else "CPU fp32"
    mo.callout(
        mo.md(f"✅ All models loaded on **{_label}** (`{_model_id}`)."),
        kind="success",
    )
    return embedder, kw_model, nli_classifier


# ============================================================
# CELL 9 — Build / Load / Save controls
# ============================================================
@app.cell
def build_load_header(mo):
    mo.md("## Step 3 — Build or Load Cluster Index")
    return


@app.cell
def build_load_controls(mo):
    build_btn = mo.ui.button(label="🔬 Build Cluster Index",  kind="success")
    load_btn  = mo.ui.button(label="📂 Load Saved Index",      kind="neutral")
    save_btn  = mo.ui.button(label="💾 Save Current Index",    kind="warn")
    return build_btn, load_btn, save_btn


@app.cell
def show_build_load_controls(mo, build_btn, load_btn, save_btn):
    mo.hstack([build_btn, load_btn, save_btn])
    return


# ============================================================
# CELL 10 — Build cluster index
# ============================================================
@app.cell
def build_clusters(
    build_btn, df_articles, embedder, kw_model, evoc,
    PCA, normalize, np, defaultdict, st_device, cuda_ok, mo,
):
    """
    1. Encode titles → 384-dim (GPU, fp16→fp32 cast for sklearn/EVōC)
    2. L2-normalise
    3. PCA → 64-dim
    4. EVōC max_layers=2
    5. Centroid index + KeyBERT keywords
    """
    build_btn  # reactive

    cluster_index = None

    if len(df_articles) < 5:
        mo.callout(mo.md("⚠️ Fetch at least 5 articles first."), kind="warn")
    else:
        titles = df_articles["title"].tolist()

        # 1. Encode ──────────────────────────────────────────────────────────────
        with mo.status.spinner(title=f"Encoding {len(titles)} titles on {st_device.upper()}…"):
            raw_embeddings = embedder.encode(
                titles,
                show_progress_bar=False,
                batch_size=128 if cuda_ok else 64,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype("float32")   # cast fp16→fp32 for sklearn/numba
            _embeddings = normalize(raw_embeddings)

        # 2. PCA ─────────────────────────────────────────────────────────────────
        with mo.status.spinner(title="PCA reduction → 64-dim…"):
            n_pca  = min(64, len(titles) - 1, _embeddings.shape[1])
            pca    = PCA(n_components=n_pca, random_state=42)
            reduced = normalize(pca.fit_transform(_embeddings).astype("float32"))

        # 3. EVōC ────────────────────────────────────────────────────────────────
        with mo.status.spinner(title="Running EVōC (2 layers)…"):
            clusterer = evoc.EVoC(
                max_layers=2,
                base_min_cluster_size=max(3, len(titles) // 30),
                n_neighbors=min(15, len(titles) // 5),
                random_state=42,
            )
            clusterer.fit(reduced)

        layers   = clusterer.cluster_layers_
        n_layers = len(layers)

        # 4. Layer maps ──────────────────────────────────────────────────────────
        layer_maps = []
        for _ll in layers:
            _cmap = defaultdict(list)
            for _i, _lbl in enumerate(_ll):
                if _lbl >= 0:
                    _cmap[int(_lbl)].append(_i)
            layer_maps.append(dict(_cmap))

        # 5. Centroids ───────────────────────────────────────────────────────────
        layer_centroids = []
        for _cmap in layer_maps:
            _cents = {}
            for _cid, _idxs in _cmap.items():
                _cents[_cid] = normalize(_embeddings[_idxs].mean(axis=0, keepdims=True))[0]
            layer_centroids.append(_cents)

        layer_sizes = [{_cid: len(_idxs) for _cid, _idxs in _cmap.items()} for _cmap in layer_maps]

        # 6. Keywords ────────────────────────────────────────────────────────────
        with mo.status.spinner(title="Extracting cluster keywords via KeyBERT…"):
            cluster_keywords = {}
            for _cid, _idxs in layer_maps[0].items():
                _text = " . ".join(titles[_i] for _i in _idxs)
                _kws  = kw_model.extract_keywords(
                    _text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5,
                )
                cluster_keywords[_cid] = [kw for kw, _ in _kws]

        noise_idxs = [i for i, lbl in enumerate(layers[0]) if lbl == -1]

        # 7. Cached t-SNE ───────────────────────────────────────────────────────
        from sklearn.manifold import TSNE as _TSNE
        with mo.status.spinner(title="t-SNE projection (cached)…"):
            _perp = min(30, max(5, len(_embeddings) // 10))
            tsne = _TSNE(
                n_components=2,
                perplexity=_perp,
                random_state=42,
                n_iter=300,
            ).fit_transform(_embeddings)

        cluster_index = {
            "embeddings": _embeddings, "reduced": reduced, "titles": titles,
            "layers": layers, "layer_maps": layer_maps,
            "layer_centroids": layer_centroids,
            "cluster_keywords": cluster_keywords,
            "noise_idxs": noise_idxs, "pca": pca,
            "layer_sizes": layer_sizes, "tsne": tsne,
            "n_layers": n_layers, "df": df_articles,
        }

        mo.callout(
            mo.md(
                f"✅ Cluster index built — **{n_layers} layers**\n\n"
                f"- Layer 0 (fine):   **{len(layer_maps[0])}** clusters\n"
                + (f"- Layer 1 (coarse): **{len(layer_maps[1])}** clusters\n" if n_layers > 1 else "")
                + f"- Noise articles:   **{len(noise_idxs)}**"
            ),
            kind="success",
        )
    return (cluster_index,)


# ============================================================
# CELL 11 — Load index from disk
# ============================================================
@app.cell
def load_index_cell(load_btn, save_dir_input, load_index, index_exists, mo):
    load_btn  # reactive
    _dir = save_dir_input.value.strip()
    loaded_index = None

    if not index_exists(_dir):
        mo.callout(mo.md(f"⚠️ No saved index at `{_dir}`. Build one first."), kind="warn")
    else:
        with mo.status.spinner(title=f"Loading index from `{_dir}`…"):
            try:
                loaded_index = load_index(_dir)
                mo.callout(
                    mo.md(
                        f"✅ Index loaded from `{_dir}`\n\n"
                        f"- Articles: **{len(loaded_index['titles'])}**\n"
                        f"- Layers: **{loaded_index['n_layers']}**\n"
                        f"- Layer-0 clusters: **{len(loaded_index['layer_maps'][0])}**"
                    ),
                    kind="success",
                )
            except Exception as _e:
                mo.callout(mo.md(f"❌ Load failed: `{_e}`"), kind="danger")
                loaded_index = None

    return (loaded_index,)


# ============================================================
# CELL 12 — Merge build + load → active index
# ============================================================
@app.cell
def active_index(cluster_index, loaded_index):
    """
    Prefer the freshly-built index; fall back to the loaded one.
    Either or both may be None until their respective button is clicked.
    """
    if cluster_index is not None:
        _active = cluster_index
    elif loaded_index is not None:
        _active = loaded_index
    else:
        _active = None
    return (_active,)


# ============================================================
# CELL 13 — Save active index to disk
# ============================================================
@app.cell
def save_index_cell(save_btn, _active, save_dir_input, save_index, mo):
    save_btn  # reactive

    if _active is None:
        mo.callout(mo.md("⚠️ No index to save — build or load one first."), kind="warn")
    else:
        _dir = save_dir_input.value.strip()
        with mo.status.spinner(title=f"Saving index to `{_dir}`…"):
            try:
                _saved = save_index(_active, _dir)
                mo.callout(mo.md(f"✅ Index saved to `{_saved}`"), kind="success")
            except Exception as _e:
                mo.callout(mo.md(f"❌ Save failed: `{_e}`"), kind="danger")
    return


# ============================================================
# CELL 14 — Saved-file listing (refreshes on any action)
# ============================================================
@app.cell
def list_saved_files(save_btn, load_btn, build_btn, save_dir_input, Path, mo):
    _d = Path(save_dir_input.value.strip())
    if not _d.exists():
        mo.md(f"*(Directory `{_d}` does not exist yet)*")
    else:
        _files = sorted(_d.iterdir())
        if not _files:
            mo.md(f"*(Directory `{_d}` is empty)*")
        else:
            _rows = "\n".join(
                f"| `{f.name}` | {f.stat().st_size / 1024:.1f} KB |"
                for f in _files if f.is_file()
            )
            mo.md(f"**Saved index files in `{_d}`:**\n\n| File | Size |\n|---|---|\n{_rows}")
    return


# ============================================================
# CELL 15 — Cluster visualisation (t-SNE)
# ============================================================
@app.cell
def visualise_clusters(_active, px, np, pd, mo):
    if _active is None:
        mo.md("*(Build or load an index first.)*")
    else:
        _embeddings  = _active["embeddings"]
        _labels_fine = _active["layers"][0]
        _df          = _active["df"].copy()

        proj = _active.get("tsne")
        if proj is None:
            from sklearn.manifold import TSNE as _TSNE
            with mo.status.spinner(title="t-SNE projection (cached)…"):
                _perp = min(30, max(5, len(_embeddings) // 10))
                proj  = _TSNE(
                    n_components=2,
                    perplexity=_perp,
                    random_state=42,
                    n_iter=300,
                ).fit_transform(_embeddings)
            _active["tsne"] = proj

        _df["x"]       = proj[:, 0]
        _df["y"]       = proj[:, 1]
        _df["cluster"] = [str(l) if l >= 0 else "noise" for l in _labels_fine]
        _df["hover"]   = _df["title"].str[:80]

        _fig = px.scatter(
            _df, x="x", y="y", color="cluster",
            hover_data={"hover": True, "source": True, "x": False, "y": False},
            title="Article Clusters — t-SNE of sentence embeddings",
            template="plotly_dark", height=620,
        )
        _fig.update_traces(marker=dict(size=6, opacity=0.75))
        mo.plotly(_fig)
    return


# ============================================================
# CELL 15B — Cluster specialization (sunburst + drilldown)
# ============================================================
@app.cell
def cluster_specialization_data(_active, pd):
    cluster_specialization_data = None
    if _active is not None:
        _labels_fine = _active["layers"][0]
        labels_coarse = _active["layers"][1] if _active.get("n_layers", 1) > 1 else None
        _df = _active["df"]

        _rows = []
        for _fine_id, _idxs in _active["layer_maps"][0].items():
            _idxs = list(_idxs)
            if not _idxs:
                continue

            if labels_coarse is None:
                coarse_id = "all"
            else:
                coarse_labels = [labels_coarse[i] for i in _idxs if labels_coarse[i] >= 0]
                if coarse_labels:
                    # Most common coarse label for this fine cluster
                    coarse_id = str(max(set(coarse_labels), key=coarse_labels.count))
                else:
                    coarse_id = "noise"

            _sources = list({_df.iloc[i]["source"] for i in _idxs})
            _rows.append({
                "coarse": f"C{coarse_id}",
                "fine": f"F{_fine_id}",
                "size": len(_idxs),
                "source_diversity": len(_sources),
                "keywords": ", ".join(_active.get("cluster_keywords", {}).get(_fine_id, [])),
            })

        cluster_specialization_data = pd.DataFrame(_rows)

    return (cluster_specialization_data,)


@app.cell
def show_cluster_specialization_viz(cluster_specialization_data, px, mo):
    if cluster_specialization_data is not None and not cluster_specialization_data.empty:
        _fig = px.sunburst(
            cluster_specialization_data,
            path=["coarse", "fine"],
            values="size",
            color="source_diversity",
            color_continuous_scale="Viridis",
            title="Cluster Specialization (Coarse → Fine)",
            template="plotly_dark",
            height=520,
        )
        _fig.update_traces(textinfo="label+value")
        mo.plotly(_fig)
    return


@app.cell
def cluster_specialization_controls(cluster_specialization_data, mo):
    coarse_select = None
    if cluster_specialization_data is not None and not cluster_specialization_data.empty:
        coarse_opts = sorted(cluster_specialization_data["coarse"].unique())
        coarse_select = mo.ui.dropdown(
            options=coarse_opts,
            value=coarse_opts[0],
            label="Coarse cluster",
        )
    return (coarse_select,)


@app.cell
def cluster_specialization_fine_select(cluster_specialization_data, coarse_select, mo):
    fine_select = None
    if cluster_specialization_data is not None and coarse_select is not None:
        subset = cluster_specialization_data[
            cluster_specialization_data["coarse"] == coarse_select.value
        ]
        fine_opts = sorted(subset["fine"].unique())
        fine_select = mo.ui.dropdown(
            options=fine_opts,
            value=fine_opts[0],
            label="Fine cluster",
        )
    return (fine_select,)


@app.cell
def show_cluster_specialization_controls(mo, coarse_select, fine_select):
    if coarse_select is not None and fine_select is not None:
        mo.hstack([coarse_select, fine_select])
    return


@app.cell
def show_cluster_specialization_table(
    _active, cluster_specialization_data, coarse_select, fine_select, pd, mo,
):
    if (
        _active is not None
        and cluster_specialization_data is not None
        and coarse_select is not None
        and fine_select is not None
    ):
        _fine_id = int(fine_select.value.replace("F", ""))
        _idxs = _active["layer_maps"][0].get(_fine_id, [])
        if _idxs:
            _df = _active["df"].iloc[list(_idxs)].copy()
            _df = _df[["source", "title"]].reset_index(drop=True)

            kw = cluster_specialization_data[
                cluster_specialization_data["fine"] == fine_select.value
            ]["keywords"].iloc[0]

            mo.md(f"### Cluster Details — {coarse_select.value} / {fine_select.value}")
            if kw:
                mo.md(f"**Keywords:** {kw}")
            mo.ui.table(_df, selection=None)
    return


# ============================================================
# CELL 16 — Cluster keyword table
# ============================================================
@app.cell
def show_keyword_table(_active, pd, mo):
    if _active is not None:
        _rows = []
        for _cid, _kws in _active["cluster_keywords"].items():
            _idxs   = _active["layer_maps"][0][_cid]
            _srcs   = list({_active["df"].iloc[_i]["source"] for _i in _idxs})
            _rows.append({
                "Cluster": _cid, "# Articles": len(_idxs),
                "Keywords": ", ".join(_kws), "Sources": ", ".join(_srcs[:4]),
            })

        mo.md("### Cluster Keyword Summary")
        mo.ui.table(pd.DataFrame(_rows).sort_values("# Articles", ascending=False), selection=None)
    return


# ============================================================
# CELL 17 — Query controls
# ============================================================
@app.cell
def query_ui(mo):
    mo.md("## Step 4 — Query & Bias Detection")
    return


@app.cell
def query_controls(mo):
    query_input = mo.ui.text(
        label="🔎 Enter a news topic / query",
        placeholder="e.g. climate change legislation",
        full_width=True,
    )
    top_k_slider = mo.ui.slider(start=1, stop=10, step=1, value=3,
                                 label="Top-k clusters per layer")
    max_articles_bias = mo.ui.slider(start=5, stop=50, step=5, value=20,
                                      label="Max articles for bias analysis")
    refine_depth = mo.ui.slider(start=1, stop=6, step=1, value=3,
                                 label="Max refinement depth")
    min_cluster_size = mo.ui.slider(start=2, stop=10, step=1, value=3,
                                    label="Min dynamic cluster size")
    divergence_threshold = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.35,
        label="Divergence threshold (cross-source)"
    )
    bias_labels = mo.ui.text(
        label="Bias hypothesis labels (comma-separated)",
        value="pro-government, anti-government, neutral, sensationalist, fear-inducing",
        full_width=True,
    )
    run_query_btn = mo.ui.button(label="🚀 Analyse Bias", kind="success")
    return (
        query_input, top_k_slider, max_articles_bias,
        refine_depth, min_cluster_size, divergence_threshold,
        bias_labels, run_query_btn,
    )


@app.cell
def show_query_controls(
    mo, query_input, top_k_slider, max_articles_bias,
    refine_depth, min_cluster_size, divergence_threshold, bias_labels, run_query_btn,
):
    mo.vstack([
        query_input,
        mo.hstack([top_k_slider, max_articles_bias]),
        mo.hstack([refine_depth, min_cluster_size, divergence_threshold]),
        bias_labels,
        run_query_btn,
    ])
    return


# ============================================================
# CELL 18 — Routing helper
# ============================================================
@app.cell
def routing_fn(np, normalize):
    def cosine_route(query_emb, layer_centroids, top_k, cluster_sizes=None):
        q    = normalize(query_emb.reshape(1, -1))[0]
        sims = {}
        for cid, c in layer_centroids.items():
            score = float(np.dot(q, c))
            if cluster_sizes:
                score *= float(np.log1p(cluster_sizes.get(cid, 1)))
            sims[cid] = score
        return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return (cosine_route,)


# ============================================================
# CELL 19 — On-the-fly deep sub-clustering
# ============================================================
@app.cell
def deep_cluster_fn(evoc, normalize, np):
    def deep_cluster(embeddings_subset, min_cluster_size=3):
        if len(embeddings_subset) < 6:
            return np.zeros(len(embeddings_subset), dtype=int)
        sub = normalize(embeddings_subset.astype("float32"))
        c   = evoc.EVoC(
            max_layers=1,
            base_min_cluster_size=max(2, min_cluster_size),
            n_neighbors=min(10, len(sub) // 3),
        )
        return c.fit_predict(sub)
    return (deep_cluster_fn,)


# ============================================================
# CELL 19B — Progressive refinement (recursive)
# ============================================================
@app.cell
def progressive_refine_fn(evoc, normalize, np):
    def progressive_refine(
        embeddings, titles, idxs, query_emb, top_k, max_depth, min_cluster_size,
        keyword_top_n=5,
    ):
        import re

        stopwords = {
            "the", "and", "for", "with", "from", "into", "this", "that",
            "about", "after", "before", "over", "under", "between", "against",
            "their", "there", "where", "while", "would", "could", "should",
            "these", "those", "your", "ours", "have", "has", "had", "will",
            "its", "are", "was", "were", "been", "being", "than", "then",
            "they", "them", "his", "her", "she", "him", "you", "our", "out",
            "who", "what", "when", "why", "how", "also", "more", "most",
            "new", "news", "says", "say",
        }

        def extract_keywords(idxs_subset):
            counts = {}
            for i in idxs_subset:
                text = titles[i].lower()
                for tok in re.findall(r"[a-zA-Z]{3,}", text):
                    if tok in stopwords:
                        continue
                    counts[tok] = counts.get(tok, 0) + 1
            ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            return [w for w, _ in ranked[:keyword_top_n]]

        if not idxs:
            return [], []

        current_idxs = idxs
        history = []
        min_for_split = max(6, min_cluster_size * 2)

        for depth in range(max_depth):
            if len(current_idxs) < min_for_split:
                break

            sub_embs = normalize(embeddings[current_idxs].astype("float32"))
            clusterer = evoc.EVoC(
                max_layers=1,
                base_min_cluster_size=max(2, min_cluster_size),
                n_neighbors=min(10, max(3, len(sub_embs) // 3)),
            )
            labels = clusterer.fit_predict(sub_embs)

            clusters = {}
            for i, lbl in enumerate(labels):
                if lbl >= 0:
                    clusters.setdefault(int(lbl), []).append(current_idxs[i])

            if len(clusters) <= 1:
                break

            cluster_keywords = {
                cid: extract_keywords(cidxs) for cid, cidxs in clusters.items()
            }
            keyword_signatures = {tuple(v) for v in cluster_keywords.values() if v}
            all_have_keywords = all(cluster_keywords[cid] for cid in clusters)
            if all_have_keywords and len(keyword_signatures) == 1:
                history.append({
                    "depth": depth + 1,
                    "clusters": len(clusters),
                    "kept_clusters": len(clusters),
                    "kept_cluster_ids": list(clusters.keys()),
                    "articles": len(current_idxs),
                    "cluster_keywords": cluster_keywords,
                    "stop_reason": "keywords identical",
                })
                break

            q = normalize(query_emb.reshape(1, -1))[0]
            centroids = {
                cid: normalize(embeddings[cidxs].mean(axis=0, keepdims=True))[0]
                for cid, cidxs in clusters.items()
            }
            scored = sorted(
                ((cid, float(np.dot(q, c))) for cid, c in centroids.items()),
                key=lambda x: x[1],
                reverse=True,
            )

            keep = [cid for cid, _ in scored[: min(top_k, len(scored))]]
            new_idxs = []
            for cid in keep:
                new_idxs.extend(clusters[cid])
            new_idxs = list(dict.fromkeys(new_idxs))

            history.append({
                "depth": depth + 1,
                "clusters": len(clusters),
                "kept_clusters": len(keep),
                "kept_cluster_ids": keep,
                "articles": len(new_idxs),
                "cluster_keywords": cluster_keywords,
            })

            if not new_idxs or len(new_idxs) == len(current_idxs):
                break

            current_idxs = new_idxs

        return current_idxs, history

    return (progressive_refine,)


# ============================================================
# CELL 20 — Main query runner
# ============================================================
@app.cell
def run_query(
    run_query_btn, query_input, top_k_slider, max_articles_bias,
    refine_depth, min_cluster_size, divergence_threshold, bias_labels, _active,
    embedder, nli_classifier, cosine_route, deep_cluster_fn,
    progressive_refine,
    normalize, np, pd, cuda_ok, st_device, mo,
):
    run_query_btn  # reactive

    query_results = None

    if _active is None:
        mo.callout(mo.md("⚠️ Build or load an index first."), kind="warn")
    elif not query_input.value.strip():
        mo.callout(mo.md("⚠️ Please enter a query."), kind="warn")
    else:
        query_text = query_input.value.strip()
        top_k      = top_k_slider.value
        max_art    = max_articles_bias.value
        _hyp_labels = [l.strip() for l in bias_labels.value.split(",") if l.strip()]

        # Encode query on GPU ────────────────────────────────────────────────────
        with mo.status.spinner(title=f"Encoding query on {st_device.upper()}…"):
            q_raw = embedder.encode(
                [query_text], show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False,
            ).astype("float32")
            q_emb = normalize(q_raw)[0]

        # Layer-0 routing ────────────────────────────────────────────────────────
        with mo.status.spinner(title="Routing through Layer 0 (fine clusters)…"):
            candidate_idxs = []
            _layer_sizes = _active.get("layer_sizes")
            _size0 = _layer_sizes[0] if _layer_sizes else None
            for _cid, _ in cosine_route(
                q_emb,
                _active["layer_centroids"][0],
                top_k,
                cluster_sizes=_size0,
            ):
                candidate_idxs.extend(_active["layer_maps"][0][_cid])

        # Layer-1 routing ────────────────────────────────────────────────────────
        if _active["n_layers"] > 1:
            with mo.status.spinner(title="Routing through Layer 1 (coarse clusters)…"):
                _size1 = _layer_sizes[1] if _layer_sizes and len(_layer_sizes) > 1 else None
                for _cid, _ in cosine_route(
                    q_emb,
                    _active["layer_centroids"][1],
                    top_k,
                    cluster_sizes=_size1,
                ):
                    candidate_idxs.extend(_active["layer_maps"][1][_cid])

        # Noise bucket (direct cosine) ───────────────────────────────────────────
        _noise = _active["noise_idxs"]
        if _noise:
            _nsims = _active["embeddings"][_noise] @ q_emb
            for _ni in np.argsort(_nsims)[::-1][: top_k * 3]:
                candidate_idxs.append(_noise[_ni])

        candidate_idxs = list(dict.fromkeys(candidate_idxs))

        # Progressive refinement (recursive) ────────────────────────────────────
        with mo.status.spinner(title="Progressive refinement…"):
            refined_idxs, refine_hist = progressive_refine(
                _active["embeddings"],
                _active["titles"],
                candidate_idxs,
                q_emb,
                top_k=top_k,
                max_depth=refine_depth.value,
                min_cluster_size=min_cluster_size.value,
            )

        if not refined_idxs:
            mo.callout(mo.md("⚠️ No articles survived refinement."), kind="warn")
        else:
            if refine_hist:
                _lines = "\n".join(
                    f"- Depth {h['depth']}: {h['articles']} articles from {h['clusters']} clusters"
                    for h in refine_hist
                )
                _stop = next((h.get("stop_reason") for h in refine_hist if h.get("stop_reason")), None)
                _stop_line = f"\n- Stopped: {_stop}" if _stop else ""
                mo.callout(mo.md(f"🔍 Refinement path:\n{_lines}{_stop_line}"), kind="info")

            # Rank and trim ──────────────────────────────────────────────────────────
            _cand_embs  = _active["embeddings"][refined_idxs]
            _cand_sims  = _cand_embs @ q_emb
            _ranked     = np.argsort(_cand_sims)[::-1][:max_art]
            final_idxs  = [refined_idxs[i] for i in _ranked]
            final_sims  = [float(_cand_sims[i]) for i in _ranked]

            # Deep sub-cluster ───────────────────────────────────────────────────────
            with mo.status.spinner(title="Deep sub-clustering candidates…"):
                sub_labels = deep_cluster_fn(_active["embeddings"][final_idxs], min_cluster_size=3)

            # Cross-source divergence per fine cluster ──────────────────────────────
            with mo.status.spinner(title="Cross-source divergence…"):
                texts = []
                for idx in final_idxs:
                    _row = _active["df"].iloc[idx]
                    _summary = _row["summary"] if "summary" in _active["df"].columns else ""
                    title = _active["titles"][idx]
                    texts.append(f"{title}. {_summary}" if _summary else title)

                div_embs = embedder.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=64 if cuda_ok else 16,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                ).astype("float32")
                div_embs = normalize(div_embs)

                fine_labels = [_active["layers"][0][i] for i in final_idxs]
                cluster_map = {}
                for pos, (idx, cl) in enumerate(zip(final_idxs, fine_labels)):
                    if cl < 0:
                        continue
                    cluster_map.setdefault(int(cl), []).append((pos, idx))

                divergence_rows = []
                for cl, items in cluster_map.items():
                    if len(items) < 2:
                        continue
                    _sources = [
                        _active["df"].iloc[idx]["source"]
                        for _, idx in items
                    ]
                    unique_sources = sorted(set(_sources))
                    if len(unique_sources) < 2:
                        continue

                    src_to_vecs = {}
                    for (pos, idx), src in zip(items, _sources):
                        src_to_vecs.setdefault(src, []).append(div_embs[pos])

                    src_centroids = {}
                    for src, vecs in src_to_vecs.items():
                        src_centroids[src] = normalize(
                            np.mean(np.vstack(vecs), axis=0, keepdims=True)
                        )[0]

                    src_list = sorted(src_centroids.keys())
                    pair_dists = []
                    for i in range(len(src_list)):
                        for j in range(i + 1, len(src_list)):
                            v1 = src_centroids[src_list[i]]
                            v2 = src_centroids[src_list[j]]
                            pair_dists.append(1.0 - float(np.dot(v1, v2)))

                    if not pair_dists:
                        continue

                    avg_dist = float(sum(pair_dists) / len(pair_dists))
                    divergence_rows.append({
                        "fine_cluster": cl,
                        "sources": ", ".join(unique_sources),
                        "num_sources": len(unique_sources),
                        "num_articles": len(items),
                        "avg_pairwise_distance": round(avg_dist, 4),
                        "flagged": "yes" if avg_dist >= divergence_threshold.value else "no",
                    })

                divergence_df = pd.DataFrame(divergence_rows).sort_values(
                    "avg_pairwise_distance",
                    ascending=False,
                ) if divergence_rows else pd.DataFrame(
                    columns=[
                        "fine_cluster", "sources", "num_sources", "num_articles",
                        "avg_pairwise_distance", "flagged",
                    ]
                )

            # NLI bias detection — batched on GPU ────────────────────────────────────
            final_titles  = [_active["titles"][i] for i in final_idxs]
            final_sources = [_active["df"].iloc[i]["source"] for i in final_idxs]

            with mo.status.spinner(
                title=f"NLI bias detection — {len(final_titles)} articles "
                      f"({'GPU' if cuda_ok else 'CPU'})…"
            ):
                _raw = nli_classifier(
                    final_titles,
                    candidate_labels=_hyp_labels,
                    hypothesis_template="This news article is {}.",
                    batch_size=16 if cuda_ok else 4,
                    multi_label=False,
                )
                # pipeline returns a list when given a list
                if isinstance(_raw, dict):
                    _raw = [_raw]
                _bias_results = [
                    {
                        "top_bias_label": r["labels"][0],
                        "confidence":     round(r["scores"][0], 3),
                        "all_scores":     dict(zip(r["labels"], [round(s, 3) for s in r["scores"]])),
                    }
                    for r in _raw
                ]

            # Assemble dataframe ─────────────────────────────────────────────────────
            _rows = [
                {
                    "rank":        rk + 1,
                    "title":       _active["titles"][idx],
                    "source":      src,
                    "similarity":  round(sim, 4),
                    "fine_cluster": int(_active["layers"][0][idx]),
                    "sub_cluster": int(sl),
                    "bias_label":  bres["top_bias_label"],
                    "confidence":  bres["confidence"],
                    "url":         _active["df"].iloc[idx]["url"],
                }
                for rk, (idx, sim, bres, src, sl) in enumerate(
                    zip(final_idxs, final_sims, _bias_results, final_sources, sub_labels)
                )
            ]

            query_results = {
                "df":           pd.DataFrame(_rows),
                "bias_results": _bias_results,
                "query":        query_text,
                "hyp_labels":   _hyp_labels,
                "refine_history": refine_hist,
                "cross_source_divergence": divergence_df,
            }

            mo.callout(
                mo.md(f"✅ **{len(_rows)}** relevant articles for *'{query_text}'*"),
                kind="success",
            )
    return (query_results,)


# ============================================================
# CELL 21 — Results table
# ============================================================
@app.cell
def show_results_table(query_results, mo):
    if query_results is not None:
        mo.md(f"### Results — *{query_results['query']}*")
        mo.ui.table(
            query_results["df"][
                [
                    "rank", "source", "title", "similarity",
                    "fine_cluster", "sub_cluster", "bias_label", "confidence",
                ]
            ],
            selection=None,
        )
    return


# ============================================================
# CELL 21P — Parallel coordinates (brush to filter)
# ============================================================
@app.cell
def parallel_coords_data(query_results, pd):
    par_df = None
    source_map = None
    bias_map = None
    if query_results is not None:
        par_df = query_results["df"].copy()

        source_vals = sorted(par_df["source"].unique().tolist())
        source_map = {s: i for i, s in enumerate(source_vals)}
        par_df["source_code"] = par_df["source"].map(source_map)

        bias_vals = sorted(par_df["bias_label"].unique().tolist())
        bias_map = {b: i for i, b in enumerate(bias_vals)}
        par_df["bias_code"] = par_df["bias_label"].map(bias_map)

        div_df = query_results.get("cross_source_divergence")
        if div_df is not None and not div_df.empty:
            div_map = {
                int(r["fine_cluster"]): float(r["avg_pairwise_distance"])
                for _, r in div_df.iterrows()
            }
            par_df["cluster_divergence"] = par_df["fine_cluster"].map(div_map).fillna(0.0)
        else:
            par_df["cluster_divergence"] = 0.0

    return par_df, source_map, bias_map


@app.cell
def parallel_coords_plot(par_df, source_map, bias_map, go, mo):
    plot = None
    par_dims = None
    if par_df is not None:
        dimensions = [
            dict(label="Similarity", values=par_df["similarity"]),
            dict(label="Confidence", values=par_df["confidence"]),
            dict(label="Cluster Divergence", values=par_df["cluster_divergence"]),
            dict(label="Fine Cluster", values=par_df["fine_cluster"]),
            dict(label="Sub Cluster", values=par_df["sub_cluster"]),
            dict(
                label="Source",
                values=par_df["source_code"],
                tickvals=list(source_map.values()),
                ticktext=list(source_map.keys()),
            ),
            dict(
                label="Bias Label",
                values=par_df["bias_code"],
                tickvals=list(bias_map.values()),
                ticktext=list(bias_map.keys()),
            ),
        ]

        _fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=par_df["bias_code"],
                    colorscale="Viridis",
                    showscale=False,
                ),
                dimensions=dimensions,
            )
        )
        _fig.update_layout(
            title="Parallel Coordinates — Brush to Filter",
            template="plotly_dark",
            height=520,
        )

        par_dims = [
            "similarity",
            "confidence",
            "cluster_divergence",
            "fine_cluster",
            "sub_cluster",
            "source_code",
            "bias_code",
        ]

        if hasattr(mo.ui, "plotly"):
            plot = mo.ui.plotly(_fig)
            mo.vstack([plot])
        else:
            mo.plotly(_fig)
    return plot, par_dims


@app.cell
def parallel_coords_selection_table(par_df, par_plot, par_dims, mo):
    if par_df is not None:
        selected_df = par_df

        payload = None
        if par_plot is not None:
            payload = par_plot.value

        relayout = None
        if isinstance(payload, dict):
            relayout = payload.get("relayoutData") or payload.get("relayout") or payload

        if isinstance(relayout, dict):
            import re as _re

            constraints = {}
            for key, val in relayout.items():
                m = _re.match(r"dimensions\[(\d+)\]\.constraintrange", str(key))
                if m:
                    constraints[int(m.group(1))] = val

            def in_ranges(value, ranges):
                if isinstance(ranges, list) and len(ranges) == 2 and all(
                    isinstance(x, (int, float)) for x in ranges
                ):
                    return ranges[0] <= value <= ranges[1]
                if isinstance(ranges, list) and ranges and isinstance(ranges[0], list):
                    return any(r[0] <= value <= r[1] for r in ranges)
                return True

            for dim_idx, ranges in constraints.items():
                if dim_idx < len(par_dims):
                    col = par_dims[dim_idx]
                    selected_df = selected_df[selected_df[col].apply(lambda v: in_ranges(v, ranges))]

        mo.md(f"### Parallel Coordinates Selection — {len(selected_df)} rows")
        mo.ui.table(
            selected_df[[
                "rank", "source", "title", "similarity", "confidence",
                "fine_cluster", "sub_cluster", "bias_label", "cluster_divergence",
            ]],
            selection=None,
        )
    return


# ============================================================
# CELL 21A — Cross-source divergence
# ============================================================
@app.cell
def show_cross_source_divergence(query_results, mo):
    if query_results is not None:
        _df = query_results.get("cross_source_divergence")
        mo.md("### Cross-Source Divergence")
        if _df is None or _df.empty:
            mo.md("*(No clusters with 2+ sources in the current results.)*")
        else:
            mo.ui.table(_df, selection=None)
    return


# ============================================================
# CELL 21B — Refinement keywords
# ============================================================
@app.cell
def show_refinement_keywords(query_results, pd, mo):
    if query_results is not None:
        hist = query_results.get("refine_history")
        if hist:
            _rows = []
            for h in hist:
                kept = set(h.get("kept_cluster_ids", []))
                for cid, kws in h.get("cluster_keywords", {}).items():
                    _rows.append({
                        "Depth": h["depth"],
                        "Cluster": cid,
                        "Kept": "yes" if cid in kept else "no",
                        "Keywords": ", ".join(kws),
                    })
                if h.get("stop_reason"):
                    _rows.append({
                        "Depth": h["depth"],
                        "Cluster": "stop",
                        "Kept": "",
                        "Keywords": f"Stop reason: {h['stop_reason']}",
                    })

            mo.md("### Refinement Cluster Keywords")
            mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


# ============================================================
# CELL 22 — Bias charts
# ============================================================
@app.cell
def bias_chart(query_results, px, pd, mo):
    if query_results is not None:
        _df           = query_results["df"]
        _hyp_labels   = query_results["hyp_labels"]
        _bias_results = query_results["bias_results"]

        _label_scores = {lbl: [] for lbl in _hyp_labels}
        for _bres in _bias_results:
            for _lbl in _hyp_labels:
                _label_scores[_lbl].append(_bres["all_scores"].get(_lbl, 0.0))

        agg = pd.DataFrame({
            "Bias Label":      list(_label_scores.keys()),
            "Mean Confidence": [sum(v) / len(v) if v else 0 for v in _label_scores.values()],
        }).sort_values("Mean Confidence", ascending=False)

        fig1 = px.bar(
            agg, x="Bias Label", y="Mean Confidence",
            color="Mean Confidence", color_continuous_scale="RdYlGn_r",
            title=f"Avg Bias Scores — '{query_results['query']}'",
            template="plotly_dark",
        )
        fig1.update_layout(showlegend=False, height=350)

        _sb = _df.groupby(["source", "bias_label"]).size().reset_index(name="count")
        fig2 = px.bar(
            _sb, x="source", y="count", color="bias_label",
            barmode="stack", title="Bias Labels by Source",
            template="plotly_dark", height=380,
        )

        fig3 = px.scatter(
            _df, x="similarity", y="confidence",
            color="bias_label", symbol="source",
            hover_data={"title": True},
            title="Similarity vs Bias Confidence",
            template="plotly_dark", height=380,
        )

        mo.vstack([
            mo.plotly(fig1),
            mo.hstack([mo.plotly(fig2), mo.plotly(fig3)]),
        ])
    return


# ============================================================
# CELL 23 — Source-level summary
# ============================================================
@app.cell
def source_bias_summary(query_results, pd, mo):
    if query_results is not None:
        _df = query_results["df"]
        mo.md("### Source-Level Bias Summary")
        _summary = (
            _df.groupby("source")
            .agg(
                articles       =("title",      "count"),
                avg_similarity =("similarity", "mean"),
                dominant_bias  =("bias_label", lambda x: x.value_counts().index[0]),
                avg_confidence =("confidence", "mean"),
            )
            .reset_index()
            .sort_values("avg_similarity", ascending=False)
        )
        _summary["avg_similarity"] = _summary["avg_similarity"].round(4)
        _summary["avg_confidence"] = _summary["avg_confidence"].round(3)
        mo.ui.table(_summary, selection=None)
    return


# ============================================================
# CELL 24 — Per-article deep-dive
# ============================================================
@app.cell
def article_deepdive(query_results, mo):
    article_options = None
    article_select = None
    if query_results is not None:
        _df = query_results["df"]
        article_options = {
            f"[{r['source']}] {r['title'][:80]}": r for _, r in _df.iterrows()
        }
        article_select = mo.ui.dropdown(
            options=list(article_options.keys()),
            label="Deep-dive into a specific article",
        )
    return article_options, article_select


@app.cell
def show_article_select(mo, article_select):
    if article_select is not None:
        mo.vstack([article_select])
    return


@app.cell
def show_article_details(article_select, article_options, query_results, mo):
    if article_select is not None and article_select.value is not None and query_results is not None:
        _row   = article_options[article_select.value]
        bres  = query_results["bias_results"][int(_row["rank"]) - 1]
        _rows = "\n".join(
            f"| {lbl} | {score:.3f} |"
            for lbl, score in sorted(bres["all_scores"].items(), key=lambda x: -x[1])
        )
        mo.callout(mo.md(f"""
    **Title:** {_row['title']}

    **Source:** {_row['source']} | **Similarity:** {_row['similarity']} | **Sub-cluster:** {_row['sub_cluster']}

    **Top Bias Label:** `{_row['bias_label']}` (confidence: {_row['confidence']})

| Bias Label | Score |
|---|---|
{_rows}

{'[🔗 Read article](' + _row['url'] + ')' if _row['url'] else ''}
"""), kind="info")
    return


# ============================================================
# CELL 25 — Export CSV
# ============================================================
@app.cell
def export_ui(query_results, mo):
    if query_results is not None:
        mo.md("### Export Query Results")
        _csv = query_results["df"].to_csv(index=False).encode()
        mo.download(
            data=_csv,
            filename=f"bias_results_{query_results['query'][:30].replace(' ', '_')}.csv",
            mimetype="text/csv",
            label="⬇️ Download Results CSV",
        )
    return


if __name__ == "__main__":
    app.run()
