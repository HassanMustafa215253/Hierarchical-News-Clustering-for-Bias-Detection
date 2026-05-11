from collections import Counter
import json
import csv
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# from newsapi import NewsApiClient

# API_KEY = "9149735d77014a2ebbf75f14a59ef9ea"
# CACHE_FILE = Path("top_headlines_cache.json")


# def extract_titles(payload: dict[str, Any]) -> list[str]:
#     """Extract cleaned title strings from a NewsAPI response payload."""
#     titles = [
#         (article.get("title") or "").split("-", 1)[0].strip()
#         for article in payload.get("articles", [])
#     ]
#     return [title for title in titles if title]


# def load_cached_payload(cache_file: Path = CACHE_FILE) -> dict[str, Any] | None:
#     """Load cached NewsAPI payload if available and valid."""
#     if not cache_file.exists():
#         return None

#     try:
#         with cache_file.open("r", encoding="utf-8") as file:
#             payload = json.load(file)
#     except (json.JSONDecodeError, OSError):
#         return None

#     return payload if isinstance(payload, dict) else None


# def fetch_and_cache_payload(country: str = "us", cache_file: Path = CACHE_FILE) -> dict[str, Any]:
#     """Fetch headlines from NewsAPI and save the full payload to disk."""
#     newsapi = NewsApiClient(api_key=API_KEY)
#     payload = newsapi.get_top_headlines(country=country)

#     with cache_file.open("w", encoding="utf-8") as file:
#         json.dump(payload, file, ensure_ascii=False, indent=2)

#     return payload


# def get_sentences(country: str = "us") -> list[str]:
#     """Use cached headlines when present; otherwise fetch once and cache."""
#     payload = load_cached_payload()
#     if payload is None:
#         payload = fetch_and_cache_payload(country=country)
#         print(f"Fetched fresh headlines and cached them in {CACHE_FILE}.")
#     else:
#         print(f"Loaded headlines from cache file {CACHE_FILE}.")

#     return extract_titles(payload)


CSV_FILE = Path(__file__).resolve().parent / "archive" / "train.csv"


def load_titles_from_csv(csv_file: Path = CSV_FILE) -> list[str]:
    """Load titles from the second column named `Title` in archive/train.csv."""
    titles: list[str] = []
    with csv_file.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = (row.get("Title") or "").strip()
            if title:
                titles.append(title)
    return titles



# sentences = [
#     "I have been building AI systems",
#     "I love building AI systems",
#     "AI systems are loved by me",
#     "I use  AI systems",
#     "I don't use AI systems",
#     "AI systems are not loved by me",
#     "I don't love building AI systems",
#     "I haven't been building AI systems",
# ]


def choose_dim_components(num_samples: int, num_features: int, target_components: int = 50) -> int:
    """Pick a safe dimensionality for reduction (UMAP/PCA).

    Ensures at most `target_components` and less than number of samples/features.
    """
    return max(1, min(target_components, num_samples - 1, num_features))


def choose_umap_neighbors(num_samples: int, preferred_neighbors: int = 15) -> int:
    """Pick a valid UMAP neighbor count for the current dataset size."""
    if num_samples <= 2:
        return 1
    return max(2, min(preferred_neighbors, num_samples - 1))


def get_embedding_device() -> str:
    """Use GPU for embeddings when available, otherwise fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_cluster_topics(sentences: list[str], labels: list[int], top_terms: int = 3) -> dict[int, str]:
    """Infer a short topic for each cluster from the cluster's titles."""
    cluster_topics: dict[int, str] = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        cluster_sentences = [sentence for sentence, cluster_label in zip(sentences, labels) if cluster_label == label]
        if not cluster_sentences:
            cluster_topics[label] = "miscellaneous"
            continue

        analyzer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).build_analyzer()
        term_counts: Counter[str] = Counter()
        for sentence in cluster_sentences:
            term_counts.update(analyzer(sentence))

        terms = [term for term, _ in term_counts.most_common(top_terms)]
        cluster_topics[label] = ", ".join(terms) if terms else "miscellaneous"

    return cluster_topics


def add_hover_tooltips(fig: Any, ax: Any, scatter: Any, titles: list[str]) -> None:
    """Show a title tooltip when the pointer hovers over a scatter point."""
    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    annotation.set_visible(False)

    offsets = cast(Any, scatter.get_offsets())

    def on_move(event: Any) -> None:
        if event.inaxes != ax:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        contains, info = scatter.contains(event)
        if not contains or not info.get("ind"):
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        index = int(info["ind"][0])
        x_value = float(offsets[index][0])
        y_value = float(offsets[index][1])
        annotation.xy = (x_value, y_value)
        annotation.set_text(titles[index])
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def main() -> None:
    sentences = load_titles_from_csv()
    if len(sentences) < 2:
        print("Need at least 2 titles to cluster. CSV may be empty or missing titles.")
        return

    device = get_embedding_device()
    print(f"Using embedding device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    embeddings = model.encode(sentences, convert_to_numpy=True)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Reduce dimensionality with UMAP before clustering
    umap_components = choose_dim_components(embeddings_scaled.shape[0], embeddings_scaled.shape[1], target_components=5)
    umap_neighbors = choose_umap_neighbors(embeddings_scaled.shape[0], preferred_neighbors=15)
    umap_reducer = umap.UMAP(n_components=umap_components, n_neighbors=umap_neighbors, min_dist=0.0, metric="cosine")
    embeddings_umap = umap_reducer.fit_transform(embeddings_scaled)

    # Cluster using HDBSCAN for hierarchical/precise clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method='leaf', metric='euclidean')
    labels = clusterer.fit_predict(embeddings_umap)

    print("Cluster labels:", labels.tolist())
    print("Cluster counts:", dict(Counter(labels.tolist())))
    cluster_topics = infer_cluster_topics(sentences, labels.tolist())
    for label, topic in cluster_topics.items():
        print(f"Cluster {label} topic: {topic}")

    visualization_components = min(2, embeddings_umap.shape[0], embeddings_umap.shape[1])
    if visualization_components < 2:
        print("Not enough samples to create a 2D UMAP visualization.")
        return

    visualization_umap = umap.UMAP(n_components=2, n_neighbors=umap_neighbors, min_dist=0.0, metric="cosine")
    embeddings_2d = visualization_umap.fit_transform(embeddings_umap)

    unique_labels = sorted(set(labels.tolist()))
    plt.figure(figsize=(15, 6))
    figure = plt.gcf()
    axes = plt.gca()
    scatter_handles: list[tuple[Any, list[str]]] = []

    for label in unique_labels:
        mask = labels == label
        display_label = "Noise" if label == -1 else f"Cluster {label}: {cluster_topics.get(label, 'miscellaneous')}"
        scatter = plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=display_label,
            s=80,
            alpha=0.85,
        )
        scatter_titles = [sentence for sentence, is_selected in zip(sentences, mask) if is_selected]
        scatter_handles.append((scatter, scatter_titles))

    for scatter, titles in scatter_handles:
        add_hover_tooltips(figure, axes, scatter, titles)

    plt.title("HDBSCAN Clustering on UMAP-Reduced Embeddings")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hdbscan_umap_clusters.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

